"""Fail-closed cohort checks and train-only preprocessing contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ai4ct.reproduction.run import load_manifest, training_inliers


class ReproductionRunTests(unittest.TestCase):
    def test_filter_sees_only_supplied_training_matrix(self):
        train = np.array([[1., 2.], [3., 4.], [5., 6.]])
        with patch("ai4ct.reproduction.run.IsolationForest") as detector:
            detector.return_value.fit_predict.return_value = np.array([1, -1, 1])
            mask = training_inliers(train, 42)
            detector.return_value.fit_predict.assert_called_once()
            self.assertIs(detector.return_value.fit_predict.call_args.args[0], train)
            np.testing.assert_array_equal(mask, [True, False, True])

    def test_unverified_mapping_is_never_evaluated(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps([{
                "sample_id": "s1", "group_id": "g1", "dicom_path": "unused",
                "gpr": 95, "dicom_sha256": "unused", "mapping_status": "internal_id_conflict",
            }]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unverified"):
                load_manifest(manifest)

    def test_changed_input_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            dicom = Path(folder) / "source.dcm"
            dicom.write_bytes(b"changed")
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps([{
                "sample_id": "s1", "group_id": "g1", "dicom_path": str(dicom), "gpr": 95,
                "dicom_sha256": hashlib.sha256(b"original").hexdigest(),
                "mapping_status": "explicit_serial_and_internal_id_agree",
            }]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content differs"):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
