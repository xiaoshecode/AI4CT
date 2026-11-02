"""Keyed, audited loading of the September 2026 TaiChi source archive.

The source workbook's explicit serial key joins the RP_pID filename key. The
independent DICOM PatientID check must agree unless a caller deliberately selects
the documented filename-only sensitivity policy. Workbook row order is never a
join key. No patient names or medical record numbers are persisted in manifests.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import zipfile

from ai4ct import paths


ARCHIVE_NAME = "TaiChi 加速器PSQA 2026年9月测试版- RT Plan文件(1).zip"
SCHEMA_VERSION = "taichi-202609-keyed-v1"
GEOMETRY_VERSION = "delivery-geometry-v1"
REQUIRED_HEADERS = ("序号", "病案号", "2%/2mm", "3%/2mm", "3%/3mm")


@dataclass(frozen=True)
class Cohort:
    samples: list[dict]
    audit: dict
    manifest_path: Path
    audit_path: Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _integer_key(value, context: str) -> int:
    text = str(value).strip()
    if isinstance(value, bool) or not re.fullmatch(r"\d+(?:\.0+)?", text):
        raise ValueError(f"{context}: expected a positive integer key.")
    number = int(float(text))
    if number <= 0:
        raise ValueError(f"{context}: key must be positive.")
    return number


def _gpr(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: GPR must be numeric, not missing or text.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError(f"{context}: GPR must be finite and in 0..100 percent.")
    return number


def parse_label_rows(rows: list[list], sheet_name: str) -> tuple[dict, dict]:
    """Parse explicit serial IDs and three distinct Gamma measurement columns.

    Public for small synthetic tests; returns private patient grouping keys only
    to the caller. These private keys must never be logged or serialized.
    """
    candidates = []
    for index, row in enumerate(rows[:10]):
        headers = [str(value).strip() for value in row]
        if all(name in headers for name in REQUIRED_HEADERS):
            candidates.append((index, headers))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one unambiguous required header row.")
    header_index, headers = candidates[0]
    if any(headers.count(name) != 1 for name in REQUIRED_HEADERS):
        raise ValueError("Required workbook column names must be unique.")
    columns = {name: headers.index(name) for name in REQUIRED_HEADERS}
    labels = {}
    ignored_blank_serial_rows = 0
    unnumbered_unmeasured_rows = []
    for row_index, values in enumerate(rows[header_index + 1:], header_index + 1):
        row = list(values) + [""] * max(0, len(headers) - len(values))
        serial = row[columns["序号"]]
        if serial is None or str(serial).strip() == "":
            if any(row[columns[name]] not in (None, "") for name in REQUIRED_HEADERS[2:]):
                raise ValueError(f"Workbook row {row_index + 1}: data without a serial key.")
            ignored_blank_serial_rows += 1
            if any(value not in (None, "") for value in row):
                unnumbered_unmeasured_rows.append(row_index + 1)
            continue
        key = _integer_key(serial, f"Workbook row {row_index + 1}")
        if key in labels:
            raise ValueError("Workbook serial keys are not unique.")
        patient = row[columns["病案号"]]
        if patient is None or str(patient).strip() == "":
            raise ValueError(f"Workbook row {row_index + 1}: missing patient group key.")
        if isinstance(patient, (int, float)) and not isinstance(patient, bool):
            patient = str(_integer_key(patient, "Patient grouping key"))
        else:
            patient = str(patient).strip()
        labels[key] = {
            "_patient_key": patient,
            "gpr": _gpr(row[columns["2%/2mm"]], f"Workbook row {row_index + 1}, 2%/2mm"),
            "gpr_3pct_2mm": _gpr(row[columns["3%/2mm"]], f"Workbook row {row_index + 1}, 3%/2mm"),
            "gpr_3pct_3mm": _gpr(row[columns["3%/3mm"]], f"Workbook row {row_index + 1}, 3%/3mm"),
            "label_row": row_index + 1,
            "label_sheet": sheet_name,
        }
    if not labels:
        raise ValueError("Workbook has no numbered measurements.")
    return labels, {"label_records": len(labels), "header_row": header_index + 1,
                    "ignored_blank_serial_rows": ignored_blank_serial_rows,
                    "unnumbered_unmeasured_rows": unnumbered_unmeasured_rows,
                    "gamma_columns": ["2%/2mm", "3%/2mm", "3%/3mm"]}


def _read_labels(data: bytes) -> tuple[dict, dict]:
    try:
        import xlrd
    except ImportError as exc:
        raise ImportError("Reading the source XLS requires optional dependency xlrd>=2.0.") from exc
    book = xlrd.open_workbook(file_contents=data, on_demand=True)
    try:
        candidates = []
        for sheet in book.sheets():
            rows = [sheet.row_values(index) for index in range(sheet.nrows)]
            if any(all(header in [str(v).strip() for v in row]
                       for header in REQUIRED_HEADERS) for row in rows[:10]):
                candidates.append((rows, sheet.name))
        if len(candidates) != 1:
            raise ValueError("Source XLS must contain one matching measurement sheet.")
        return parse_label_rows(*candidates[0])
    finally:
        book.release_resources()


def validated_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject traversal/absolute/symlink members, then ignore macOS metadata."""
    members = []
    seen = set()
    total = 0
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts or any(":" in x for x in path.parts):
            raise ValueError("Unsafe path in source archive.")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Archive symlinks are unsupported.")
        if info.is_dir() or "__MACOSX" in path.parts or any(x.startswith("._") for x in path.parts):
            continue
        if path.suffix.lower() not in {".dcm", ".xls"}:
            continue
        if name.casefold() in seen:
            raise ValueError("Duplicate archive member path.")
        seen.add(name.casefold())
        total += info.file_size
        if info.file_size > 512 * 1024**2 or total > 2 * 1024**3:
            raise ValueError("Source archive exceeds the cohort extraction size limit.")
        members.append(info)
    return members


def _store_unchanged(destination: Path, content: bytes, output_root: Path) -> None:
    destination = destination.resolve()
    if not destination.is_relative_to(output_root.resolve()):
        raise ValueError("Extraction target escapes the intended output directory.")
    if destination.exists():
        if _sha(destination.read_bytes()) != _sha(content):
            raise FileExistsError("Existing derived file differs; select a new output directory.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)


def delivery_geometry_hash(dataset) -> str:
    """Hash source delivery geometry/MU without identifiers or label information.

    This identifies identical complexity inputs, not identity or clinical
    equivalence. Absolute isocentre coordinates and descriptive metadata are
    intentionally excluded. CP state is inherited as specified by DICOM.
    """
    def numeric(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite treatment geometry value.")
        return number

    fraction_groups = dataset.get("FractionGroupSequence", [])
    if len(fraction_groups) != 1:
        raise ValueError("Expected one fraction group for this cohort.")
    metersets = {}
    for reference in fraction_groups[0].get("ReferencedBeamSequence", []):
        if "BeamMeterset" in reference:
            number = int(reference.ReferencedBeamNumber)
            if number in metersets:
                raise ValueError("Duplicate beam meterset reference.")
            metersets[number] = numeric(reference.BeamMeterset)
    beams = []
    scalar_fields = ("GantryAngle", "BeamLimitingDeviceAngle", "PatientSupportAngle",
                     "NominalBeamEnergy", "DoseRateSet")
    direction_fields = ("GantryRotationDirection", "BeamLimitingDeviceRotationDirection",
                        "PatientSupportRotationDirection")
    for beam in dataset.get("BeamSequence", []):
        if str(beam.get("TreatmentDeliveryType", "")) != "TREATMENT":
            continue
        number = int(beam.BeamNumber)
        if number not in metersets or metersets[number] <= 0:
            raise ValueError("Treatment beam is missing a positive meterset.")
        devices = []
        for device in beam.get("BeamLimitingDeviceSequence", []):
            devices.append({"type": str(device.RTBeamLimitingDeviceType),
                            "pairs": int(device.NumberOfLeafJawPairs),
                            "boundaries": [numeric(v) for v in device.get("LeafPositionBoundaries", [])]})
        state, positions, control_points = {}, {}, []
        for cp in beam.get("ControlPointSequence", []):
            for field in scalar_fields:
                if field in cp:
                    state[field] = numeric(getattr(cp, field))
            for field in direction_fields:
                if field in cp:
                    state[field] = str(getattr(cp, field))
            for device in cp.get("BeamLimitingDevicePositionSequence", []):
                positions[str(device.RTBeamLimitingDeviceType)] = [numeric(v) for v in device.LeafJawPositions]
            control_points.append({"state": dict(state), "positions": dict(positions),
                                   "weight": numeric(cp.CumulativeMetersetWeight)})
        if not control_points:
            raise ValueError("Treatment beam has no control points.")
        beams.append({"mu": metersets[number], "devices": sorted(devices, key=lambda d: d["type"]),
                      "beam_type": str(beam.get("BeamType", "")),
                      "radiation_type": str(beam.get("RadiationType", "")),
                      "final_weight": numeric(beam.FinalCumulativeMetersetWeight), "cp": control_points})
    if not beams:
        raise ValueError("No treatment beams in plan.")
    # Beam order does not change the set of delivered complexity inputs.
    beams.sort(key=lambda b: json.dumps(b, sort_keys=True))
    return _sha(_json_bytes({"schema": GEOMETRY_VERSION, "beams": beams}))


def assign_groups(records: list[dict]) -> None:
    """Union patient keys and geometry hashes without using outcome labels."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    patient_first = {}
    for field in ("_patient_key", "geometry_sha256"):
        first = {}
        for index, record in enumerate(records):
            key = record[field]
            if key in first:
                parents[find(index)] = find(first[key])
            else:
                first[key] = index
            if field == "_patient_key":
                patient_first.setdefault(key, record["sample_id"])
    components = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record["sample_id"])
    for index, record in enumerate(records):
        record["group_id"] = "group-" + min(components[find(index)])
        record["patient_group_id"] = "patient-" + patient_first[record["_patient_key"]]


def _counts(records: list[dict]) -> dict:
    return {str(threshold): {"pass": sum(row["gpr"] >= threshold for row in records),
                             "fail": sum(row["gpr"] < threshold for row in records)}
            for threshold in (90, 95)}


def prepare_taichi_202609(archive_path=None, output_dir=None,
                         mismatch_policy: str = "exclude") -> Cohort:
    """Prepare an explicit-key cohort and byte-preserved local source files.

    ``exclude`` is the strict baseline: conflicting internal PatientIDs are
    excluded and audited. ``error`` stops instead. ``filename`` is an explicit
    sensitivity analysis retaining such records with an unverified annotation.
    A different policy/content requires a distinct output directory.
    """
    if mismatch_policy not in {"exclude", "error", "filename"}:
        raise ValueError("mismatch_policy must be exclude, error or filename.")
    import pydicom

    archive_path = Path(archive_path or paths.DATA_DIR / "raw" / "archives" / ARCHIVE_NAME).resolve()
    output_dir = Path(output_dir or paths.DATA_DIR / "processed" / "taichi_202609").resolve()
    archive_sha = _sha(archive_path.read_bytes())
    records, excluded = [], []
    with zipfile.ZipFile(archive_path) as archive:
        members = validated_members(archive)
        label_members = [m for m in members if PurePosixPath(m.filename).suffix.lower() == ".xls"]
        dicom_members = [m for m in members if PurePosixPath(m.filename).suffix.lower() == ".dcm"]
        if len(label_members) != 1 or not dicom_members:
            raise ValueError("Expected exactly one XLS and at least one DICOM in source archive.")
        label_member = label_members[0]
        label_bytes = archive.read(label_member)
        labels, label_audit = _read_labels(label_bytes)
        keyed_members = {}
        for member in dicom_members:
            match = re.fullmatch(r"RP_pID(\d+)\.dcm", PurePosixPath(member.filename).name, re.IGNORECASE)
            if not match:
                raise ValueError("DICOM filename does not contain the documented RP_pID serial key.")
            key = _integer_key(match.group(1), "DICOM filename")
            if key in keyed_members:
                raise ValueError("DICOM filename serial keys are not unique.")
            keyed_members[key] = member
        if set(keyed_members) != set(labels):
            raise ValueError("Workbook serial keys and DICOM filename keys do not match exactly.")
        _store_unchanged(output_dir / "source" / "labels.xls", label_bytes, output_dir)
        for key, member in sorted(keyed_members.items()):
            content = archive.read(member)
            dataset = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
            if str(dataset.get("Modality", "")) != "RTPLAN":
                raise ValueError("Source contains a non-RTPLAN DICOM.")
            try:
                internal_key = _integer_key(str(dataset.get("PatientID", "")), "DICOM PatientID")
            except ValueError:
                internal_key = None
            matched = internal_key == key
            if not matched and mismatch_policy == "error":
                raise ValueError("DICOM internal ID disagrees with explicit filename/workbook serial key.")
            destination = output_dir / "source" / "plans" / f"RP_pID{key:03d}.dcm"
            _store_unchanged(destination, content, output_dir)
            record = {
                "sample_id": f"tc202609-{key:04d}", "cohort": "taichi_202609",
                "dicom_path": str(destination), "dicom_relative_path": destination.relative_to(output_dir).as_posix(),
                "archive_member": member.filename, "dicom_sha256": _sha(content),
                "label_archive_member": label_member.filename, "label_sha256": _sha(label_bytes),
                "gamma_criteria": "2%/2mm", "gpr_unit": "percent",
                "mapping_status": "explicit_serial_and_internal_id_agree" if matched else "internal_id_conflict",
                **labels[key],
            }
            record["geometry_sha256"] = delivery_geometry_hash(dataset)
            if not matched and mismatch_policy == "exclude":
                record["exclusion_reason"] = "dicom_internal_id_conflicts_with_filename_and_workbook_serial"
                excluded.append(record)
            else:
                records.append(record)
    assign_groups(records)
    source_records = records + excluded
    geometry_counts = Counter(row["geometry_sha256"] for row in records)
    audit = {
        "schema_version": SCHEMA_VERSION, "geometry_hash_version": GEOMETRY_VERSION,
        "archive_sha256": archive_sha, "label_sha256": _sha(label_bytes),
        "mapping_rule": "XLS explicit serial = RP_pID filename serial; independently check DICOM PatientID",
        "mismatch_policy": mismatch_policy, "source_dicom_count": len(source_records),
        "included_count": len(records), "excluded_count": len(excluded),
        "internal_id_conflict_count": sum(row["mapping_status"] == "internal_id_conflict" for row in source_records),
        "source_class_counts": _counts(source_records), "included_class_counts": _counts(records),
        "patient_group_count": len({row["patient_group_id"] for row in records}),
        "split_group_count": len({row["group_id"] for row in records}),
        "unique_geometry_count": len(geometry_counts),
        "duplicate_geometry_group_count": sum(count > 1 for count in geometry_counts.values()),
        "grouping_rule": "Connected components of workbook patient key OR identical delivery geometry/MU hash; labels never used",
        "limitations": ["Explicit archive keys establish internal source consistency, not independent clinical adjudication.",
                        "Grouping hashes identify identical complexity inputs, not complete clinical plan equivalence."],
        **label_audit,
    }
    for row in source_records:
        row.pop("_patient_key", None)
    manifest = {"schema_version": SCHEMA_VERSION, "archive_sha256": archive_sha,
                "mismatch_policy": mismatch_policy, "samples": records, "excluded": excluded}
    manifest_path = output_dir / "manifest.json"
    audit_path = output_dir / "audit.json"
    _store_unchanged(manifest_path, _json_bytes(manifest), output_dir)
    _store_unchanged(audit_path, _json_bytes(audit), output_dir)
    return Cohort(records, audit, manifest_path, audit_path)


def audit_legacy_644(dicom_dir=None, output_dir=None) -> dict:
    """Record evidence about the legacy cohort without inventing a sample join.

    A matching ordinal or date is not treated as evidence that the anonymization
    mapping is known. This function deliberately never emits a training manifest.
    """
    import pydicom
    from openpyxl import load_workbook

    dicom_dir = Path(dicom_dir or paths.DICOM_DIR).resolve()
    output_dir = Path(output_dir or paths.DATA_DIR / "processed" / "taichi_legacy644").resolve()
    workbook_path = dicom_dir / "匿名化—太极ArcCheck验证患者.xlsx"
    labels_path = dicom_dir / "arccheck_labels.json"
    book = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheet1 = list(book["Sheet1"].values)
        sheet2 = list(book["Sheet2"].values)
    finally:
        book.close()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    def text(value):
        return "" if value is None else str(value).strip()

    excel_ids = {text(row[2]) for row in sheet1[1:] if row[2]}
    excel_names = {text(row[1]) for row in sheet1[1:] if row[1]}
    json_ids = {text(row.get("patient_id")) for row in labels}
    json_names = {text(row.get("patient_name")) for row in labels}
    json_ids.discard("")
    json_names.discard("")
    fields = ("PatientID", "PatientName", "RTPlanLabel", "RTPlanName", "StudyID",
              "ClinicalTrialSubjectID", "StudyDate", "RTPlanDate", "InstanceCreationDate")
    records, source_digests = [], []
    for file in sorted(dicom_dir.glob("RP_pID*.dcm")):
        content = file.read_bytes()
        source_digests.append((file.name, _sha(content)))
        ds = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True, specific_tags=list(fields))
        records.append({field: text(ds.get(field, "")) for field in fields})
    if not records:
        raise ValueError("No legacy RP_pID DICOM files available for audit.")
    field_summary = {}
    for field in fields[:6]:
        values = [row[field] for row in records]
        field_summary[field] = {
            "nonempty": sum(bool(value) for value in values),
            "unique_nonempty": len({value for value in values if value}),
            "matches_excel_patient_id": sum(value in excel_ids for value in values if value),
            "matches_json_patient_id": sum(value in json_ids for value in values if value),
            "matches_excel_name": sum(value in excel_names for value in values if value),
            "matches_json_name": sum(value in json_names for value in values if value),
            "numeric_or_pid_surrogate": sum(bool(re.fullmatch(r"(?:pID)?\d+", value, re.I)) for value in values),
        }
    sheet1_dates = {row[9].strftime("%Y%m%d") for row in sheet1[1:] if hasattr(row[9], "strftime")}
    sheet2_dates = {row[4].strftime("%Y%m%d") for row in sheet2[1:] if hasattr(row[4], "strftime")}
    date_summary = {}
    for field in fields[6:]:
        values = [row[field] for row in records]
        counts = Counter(value for value in values if value)
        date_summary[field] = {
            "nonempty": sum(bool(value) for value in values), "unique_nonempty": len(counts),
            "dates_with_only_one_dicom": sum(count == 1 for count in counts.values()),
            "matches_any_sheet1_date": sum(value in sheet1_dates for value in values),
            "matches_any_sheet2_date": sum(value in sheet2_dates for value in values),
        }
    comparison_count = min(len(sheet2) - 1, len(labels))
    audit = {
        "schema_version": "legacy644-mapping-audit-v1", "alignment_status": "unverified",
        "training_manifest_created": False,
        "dicom_count": len(records), "json_label_count": len(labels),
        "sheet1_rows_including_header": len(sheet1), "sheet2_rows_including_header": len(sheet2),
        "source_sha256": {"workbook": _sha(workbook_path.read_bytes()),
                          "labels_json": _sha(labels_path.read_bytes()),
                          "dicom_collection": _sha(_json_bytes(source_digests))},
        "identifier_evidence": field_summary, "date_evidence": date_summary,
        "sheet2_first_column_is_sequential_ordinal": all(
            text(row[0]) == str(index) or row[0] == index for index, row in enumerate(sheet2[1:], 1)),
        "json_sheet1_patient_id_overlap_count": len(json_ids & excel_ids),
        "positional_gamma_comparison": {
            "compared_rows": comparison_count,
            "equal_values": sum(sheet2[index + 1][-2] == labels[index].get("gpr_2mm_2pct")
                                for index in range(comparison_count)),
            "interpretation": "Diagnostic comparison only; row positions are not a verified sample mapping.",
        },
        "required_next_input": "Auditable anonymized DICOM ID to original plan/QA measurement crosswalk, or the original anonymization script and its mapping log.",
        "policy": "Do not generate a training cohort or choose labels using ordering, date coincidence, class balance or model performance.",
    }
    _store_unchanged(output_dir / "audit.json", _json_bytes(audit), output_dir)
    return audit
