"""Synthetic tests of keyed joins, exclusion policy and group leakage guards."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTPlanStorage, generate_uid

from ai4ct.reproduction.cohort import (
    assign_groups, audit_legacy_644, delivery_geometry_hash, parse_label_rows,
    prepare_taichi_202609, validated_members,
)


def plan_bytes(patient_id="1", mu=100.0, explicit_second_gantry=True):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = RTPlanStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = "RTPLAN"
    ds.PatientID = patient_id
    ds.PatientName = "PRIVATE^SYNTHETIC"
    reference = Dataset()
    reference.ReferencedBeamNumber = 1
    reference.BeamMeterset = mu
    fraction = Dataset()
    fraction.ReferencedBeamSequence = [reference]
    ds.FractionGroupSequence = [fraction]
    beam = Dataset()
    beam.BeamNumber = 1
    beam.TreatmentDeliveryType = "TREATMENT"
    beam.BeamType = "DYNAMIC"
    beam.RadiationType = "PHOTON"
    beam.FinalCumulativeMetersetWeight = 1.0
    device = Dataset()
    device.RTBeamLimitingDeviceType = "MLCX"
    device.NumberOfLeafJawPairs = 1
    device.LeafPositionBoundaries = [-5, 5]
    beam.BeamLimitingDeviceSequence = [device]
    cps = []
    for index in range(2):
        cp = Dataset()
        cp.ControlPointIndex = index
        cp.CumulativeMetersetWeight = float(index)
        if index == 0 or explicit_second_gantry:
            cp.GantryAngle = 10.0
        if index == 0:
            position = Dataset()
            position.RTBeamLimitingDeviceType = "MLCX"
            position.LeafJawPositions = [-10, 10]
            cp.BeamLimitingDevicePositionSequence = [position]
        cps.append(cp)
    beam.ControlPointSequence = cps
    ds.BeamSequence = [beam]
    output = io.BytesIO()
    pydicom.dcmwrite(output, ds, enforce_file_format=True)
    return output.getvalue()


class LabelContractTests(unittest.TestCase):
    headers = ["序号", "病案号", "2%/2mm", "3%/2mm", "3%/3mm"]

    def test_join_keys_are_explicit_and_preserve_source_row(self):
        labels, audit = parse_label_rows([
            ["Title"], self.headers,
            ["2", "private-b", 89.0, 96.0, 99.0],
            ["1", "private-a", 97.0, 99.0, 100.0],
            ["", "", "", "", ""],
        ], "Measurements")
        self.assertEqual(labels[1]["gpr"], 97.0)
        self.assertEqual(labels[1]["label_row"], 4)
        self.assertEqual(audit["ignored_blank_serial_rows"], 1)

    def test_duplicate_serial_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            parse_label_rows([self.headers, [1, "a", 90, 95, 99], [1, "b", 91, 95, 99]], "S")

    def test_missing_serial_does_not_silently_drop_measurement(self):
        with self.assertRaisesRegex(ValueError, "without a serial"):
            parse_label_rows([self.headers, ["", "a", 90, 95, 99]], "S")

    def test_invalid_gamma_is_rejected(self):
        for value in (None, "90", float("nan"), 101, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_label_rows([self.headers, [1, "a", value, 95, 99]], "S")


class ArchiveAndGroupingTests(unittest.TestCase):
    def test_archive_traversal_and_absolute_paths_rejected(self):
        for name in ("../RP_pID001.dcm", "/RP_pID001.dcm", "C:\\RP_pID001.dcm"):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr(name, b"x")
            with zipfile.ZipFile(data) as archive, self.assertRaises(ValueError):
                validated_members(archive)

    def test_group_union_is_transitive_and_outcome_independent(self):
        rows = [
            {"sample_id": "s1", "_patient_key": "p1", "geometry_sha256": "g1", "gpr": 1},
            {"sample_id": "s2", "_patient_key": "p1", "geometry_sha256": "g2", "gpr": 99},
            {"sample_id": "s3", "_patient_key": "p2", "geometry_sha256": "g2", "gpr": 50},
        ]
        assign_groups(rows)
        self.assertEqual(len({row["group_id"] for row in rows}), 1)
        self.assertEqual(len({row["patient_group_id"] for row in rows}), 2)

    def test_geometry_hash_inherits_state_ignores_identity_and_detects_mu(self):
        first = pydicom.dcmread(io.BytesIO(plan_bytes("1", 100, True)))
        inherited = pydicom.dcmread(io.BytesIO(plan_bytes("2", 100, False)))
        different = pydicom.dcmread(io.BytesIO(plan_bytes("1", 101, True)))
        self.assertEqual(delivery_geometry_hash(first), delivery_geometry_hash(inherited))
        self.assertNotEqual(delivery_geometry_hash(first), delivery_geometry_hash(different))

    def test_strict_cohort_excludes_id_conflict_and_reuses_identical_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                # Reverse physical archive order to prove that order is not the join.
                archive.writestr("plans/RP_pID002.dcm", plan_bytes("1"))
                archive.writestr("plans/RP_pID001.dcm", plan_bytes("1"))
                archive.writestr("labels.xls", b"synthetic workbook")
                archive.writestr("__MACOSX/._labels.xls", b"metadata")
            labels = {
                2: {"_patient_key": "patient-b", "gpr": 98.0, "gpr_3pct_2mm": 99.0,
                    "gpr_3pct_3mm": 100.0, "label_row": 3, "label_sheet": "S"},
                1: {"_patient_key": "patient-a", "gpr": 87.0, "gpr_3pct_2mm": 99.0,
                    "gpr_3pct_3mm": 100.0, "label_row": 4, "label_sheet": "S"},
            }
            with patch("ai4ct.reproduction.cohort._read_labels", return_value=(labels, {"label_records": 2})):
                cohort = prepare_taichi_202609(source, root / "prepared")
                repeated = prepare_taichi_202609(source, root / "prepared")
            self.assertEqual(len(cohort.samples), 1)
            self.assertEqual(cohort.samples[0]["sample_id"], "tc202609-0001")
            self.assertEqual(cohort.samples[0]["gpr"], 87.0)
            self.assertEqual(cohort.audit["excluded_count"], 1)
            self.assertEqual(repeated.audit, cohort.audit)
            manifest_text = cohort.manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("patient-a", manifest_text)
            self.assertNotIn("patient-b", manifest_text)
            self.assertNotIn("_patient_key", manifest_text)
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["excluded"][0]["mapping_status"], "internal_id_conflict")
            self.assertEqual(source.read_bytes()[:2], b"PK")

    def test_legacy_audit_does_not_turn_ordinals_into_a_training_manifest(self):
        class Sheet:
            def __init__(self, rows):
                self.values = rows

        class Book(dict):
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RP_pID001.dcm").write_bytes(plan_bytes("1"))
            (root / "匿名化—太极ArcCheck验证患者.xlsx").write_bytes(b"mock source hash")
            (root / "arccheck_labels.json").write_text(json.dumps([
                {"patient_id": "private-record-key", "patient_name": "PRIVATE-NAME", "gpr_2mm_2pct": 95.0}
            ]), encoding="utf-8")
            book = Book({
                "Sheet1": Sheet([[None] * 12, [1, "PRIVATE-NAME", "private-record-key", "", "", "", "", "", "", "", 95.0, 99.0]]),
                "Sheet2": Sheet([[None] * 7, [1, "", "", "", "", 95.0, 99.0]]),
            })
            with patch("openpyxl.load_workbook", return_value=book):
                audit = audit_legacy_644(root, root / "audit")
            self.assertEqual(audit["alignment_status"], "unverified")
            self.assertFalse(audit["training_manifest_created"])
            self.assertTrue(audit["sheet2_first_column_is_sequential_ordinal"])
            self.assertEqual(audit["positional_gamma_comparison"]["equal_values"], 1)
            self.assertFalse((root / "audit" / "manifest.json").exists())
            self.assertNotIn("private-record-key", (root / "audit" / "audit.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
