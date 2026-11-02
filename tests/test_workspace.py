"""Regression checks for paths and the new data diagnostics, using synthetic records."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ai4ct.doctor import inspect_data
from ai4ct import paths

ROOT = Path(__file__).resolve().parents[1]


class DataDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.raw = self.data / "raw" / "taichi"
        self.raw.mkdir(parents=True)
        (self.raw / "RP_pID001.dcm").touch()  # Count-only check, no fabricated DICOM parsing.

    def labels(self, values):
        (self.raw / "arccheck_labels.json").write_text(
            json.dumps([{"gpr_2mm_2pct": value} for value in values]), encoding="utf-8")

    def test_count_mismatch_is_an_error(self):
        self.labels([95, 91])
        report = inspect_data(self.data)
        self.assertTrue(any("counts differ" in error for error in report["errors"]))

    def test_matching_counts_do_not_claim_alignment(self):
        self.labels([95])
        report = inspect_data(self.data)
        self.assertEqual(report["alignment_status"], "unverified")
        self.assertEqual(report["invalid_gpr_count"], 0)
        self.assertEqual(report["all_labels_descriptive_counts"]["95"]["pass"], 1)

    def test_invalid_values_are_detected(self):
        self.labels([None, True, "95", float("nan"), 101, -1])
        self.assertEqual(inspect_data(self.data)["invalid_gpr_count"], 6)

    def test_bad_json_schema_is_reported(self):
        (self.raw / "arccheck_labels.json").write_text('{"wrong": []}', encoding="utf-8")
        self.assertTrue(any("list of objects" in error for error in inspect_data(self.data)["errors"]))

    def test_changed_labels_with_same_features_are_detected(self):
        import openpyxl
        self.labels([95])
        processed = self.data / "processed"
        processed.mkdir()
        for name, label in (("radiotherapy_features.xlsx", 0), ("radiotherapy_features_v2.xlsx", 1)):
            workbook = openpyxl.Workbook()
            workbook.active.append(["AAV", "label"])
            workbook.active.append([0.4, label])
            workbook.save(processed / name)
            workbook.close()
        report = inspect_data(self.data)
        comparison = report["feature_version_comparisons"][0]
        self.assertEqual(comparison["identical_feature_rows"], 1)
        self.assertEqual(comparison["different_label_rows"], 1)
        self.assertTrue(any("historical labels differ" in error for error in report["errors"]))

    def test_different_table_lengths_are_not_silently_truncated(self):
        import openpyxl
        self.labels([95])
        processed = self.data / "processed"
        processed.mkdir()
        for name, size in (("radiotherapy_features.xlsx", 1), ("radiotherapy_features_v2.xlsx", 2)):
            workbook = openpyxl.Workbook()
            workbook.active.append(["AAV", "label"])
            for _ in range(size):
                workbook.active.append([0.4, 1])
            workbook.save(processed / name)
            workbook.close()
        self.assertTrue(any("row counts differ" in error for error in inspect_data(self.data)["errors"]))

    def test_blank_workbook_headers_are_reported(self):
        import openpyxl
        self.labels([95])
        processed = self.data / "processed"
        processed.mkdir()
        workbook = openpyxl.Workbook()
        workbook.active.append(["AAV", "label", None])
        workbook.active.append([0.4, 1, 99])
        workbook.save(processed / "radiotherapy_features.xlsx")
        workbook.close()
        report = inspect_data(self.data)
        self.assertTrue(any("headers must be" in error for error in report["errors"]))


class PathAndCliTests(unittest.TestCase):
    def subprocess_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        for name in ("AI4CT_PROJECT_ROOT", "AI4CT_DATA_DIR", "AI4CT_OUTPUT_DIR"):
            env.pop(name, None)
        return env

    def test_paths_work_outside_repository(self):
        result = subprocess.run([sys.executable, "-c", "from ai4ct.paths import PROJECT_ROOT; print(PROJECT_ROOT)"],
                                cwd=tempfile.gettempdir(), env=self.subprocess_env(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), ROOT)

    def test_relative_override_is_rejected(self):
        with patch.dict(os.environ, {"AI4CT_DATA_DIR": "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                paths._absolute_override("AI4CT_DATA_DIR", ROOT)

    def test_config_import_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            env = self.subprocess_env()
            env["AI4CT_OUTPUT_DIR"] = str(output)
            result = subprocess.run([sys.executable, "-c",
                                     "import ai4ct.benchmarks.config; import ai4ct.advanced.config"],
                                    cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())

    def test_help_does_not_train_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.subprocess_env()
            env["AI4CT_OUTPUT_DIR"] = str(Path(directory) / "not_created")
            for arguments in (("--help",), ("doctor", "--help"), ("export-metrics", "--help")):
                result = subprocess.run([sys.executable, "-m", "ai4ct", *arguments],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
            self.assertFalse(Path(env["AI4CT_OUTPUT_DIR"]).exists())


if __name__ == "__main__":
    unittest.main()
