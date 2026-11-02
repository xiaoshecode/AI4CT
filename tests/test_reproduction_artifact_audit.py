"""Fail-closed audit of synthetic persisted runs; no models are fitted here."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from ai4ct.reproduction import models
from ai4ct.reproduction.artifact_audit import audit_run
from ai4ct.reproduction.evaluation import classification_metrics, group_bootstrap_accuracy, make_folds


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rates = np.array([85., 96., 85., 96., 85., 96.])
        self.groups = np.array([f"group-{i}" for i in range(6)])
        self.y = (self.rates >= 95).astype(int)
        self.probabilities = .1 + .8 * self.y
        self.folds = make_folds(self.rates, self.groups, n_splits=2, seed=42)
        source = self.root / "source.dcm"
        source.write_bytes(b"synthetic source bytes")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        self.samples = [{"sample_id": f"sample-{i}", "group_id": self.groups[i], "gpr": self.rates[i],
                         "dicom_path": str(source), "dicom_sha256": source_hash,
                         "mapping_status": "explicit_serial_and_internal_id_agree"} for i in range(6)]
        manifest = self.root / "manifest.json"
        write_json(manifest, {"samples": self.samples})
        source_key = "src/ai4ct/reproduction/models.py"
        model_source = Path(models.__file__).read_bytes()
        with zipfile.ZipFile(self.root / "source_snapshot.zip", "w") as archive:
            archive.writestr(source_key, model_source)
        write_json(self.root / "protocol.json", {
            "manifest_path": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "n_samples": 6, "n_groups": 6, "folds": 2, "seed": 42, "thresholds": [95.], "models": ["gbdt"],
            "source_sha256": {source_key: hashlib.sha256(model_source).hexdigest()},
        })
        write_json(self.root / "status.json", {"status": "complete", "completed_fits": 2})
        write_json(self.root / "splits.json", [
            {"sample_id": row["sample_id"], "group_id": row["group_id"], "test_fold": int(fold)}
            for row, fold in zip(self.samples, self.folds)])
        write_json(self.root / "feature_names.json", ["feature"])
        write_json(self.root / "feature_schema.json", {"ordered_names": ["feature"]})
        self.feature_rows = [[row["sample_id"], float(p)] for row, p in zip(self.samples, self.probabilities)]
        write_csv(self.root / "features.csv", ["sample_id", "feature"], self.feature_rows)
        filters, majority = [], np.empty(6)
        for fold in range(2):
            train = self.folds != fold
            test = ~train
            filters.append({"fold": fold, "train_before": int(train.sum()), "train_after": int(train.sum()),
                            "test_count": int(test.sum()),
                            "retained_sample_ids": [row["sample_id"] for row, keep in zip(self.samples, train) if keep]})
            majority[test] = int(np.bincount(self.y[train], minlength=2).argmax())
            model_dir = self.root / "gpr_95" / f"fold_{fold}" / "gbdt"
            write_json(model_dir / "metadata.json", {"fixture": True})
            (model_dir / "model.joblib").write_bytes(b"mock checkpoint")
        write_json(self.root / "training_filter.json", filters)
        self.prediction_headers = ["sample_id", "group_id", "fold", "gpr", "label_pass", "gbdt_pass_probability", "majority_pass_probability"]
        self.prediction_rows = [[row["sample_id"], row["group_id"], int(self.folds[i]), float(self.rates[i]),
                                 int(self.y[i]), float(self.probabilities[i]), float(majority[i])]
                                for i, row in enumerate(self.samples)]
        write_csv(self.root / "gpr_95_predictions.csv", self.prediction_headers, self.prediction_rows)
        results = {}
        for name, p in {"gbdt": self.probabilities, "majority": majority}.items():
            oof = classification_metrics(self.y, p)
            oof["accuracy_ci95"] = group_bootstrap_accuracy(self.y, p, self.groups, seed=42, repetitions=20)
            folds = [classification_metrics(self.y[self.folds == f], p[self.folds == f]) for f in range(2)]
            results[name] = {"oof": oof, "primary_holdout_fold0": folds[0], "folds": folds}
        write_json(self.root / "results.json", {"gpr_95": results})

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def loader(directory, device="cpu"):
        fold = int(Path(directory).parent.name.split("_")[1])
        return SimpleNamespace(name="gbdt", n_features=1, seed=42 + fold,
                               predict_proba=lambda x: x[:, 0])

    def test_complete_artifacts_replay_without_training(self):
        with patch.object(models, "load_model", side_effect=self.loader), patch.object(models, "train_model") as train:
            report = audit_run(self.root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checkpoints_replayed"], 2)
        train.assert_not_called()

    def test_changed_prediction_gpr_rejected_even_when_label_stays_same(self):
        self.prediction_rows[0][3] = 84.0
        write_csv(self.root / "gpr_95_predictions.csv", self.prediction_headers, self.prediction_rows)
        with self.assertRaisesRegex(ValueError, "GPR differs"):
            audit_run(self.root)
        self.assertEqual(json.loads((self.root / "artifact_audit.json").read_text())["status"], "failed")

    def test_reordered_feature_rows_rejected(self):
        write_csv(self.root / "features.csv", ["sample_id", "feature"], list(reversed(self.feature_rows)))
        with self.assertRaisesRegex(ValueError, "Feature sample order"):
            audit_run(self.root)

    def test_missing_filter_records_rejected(self):
        write_json(self.root / "training_filter.json", [])
        with self.assertRaisesRegex(ValueError, "every fold"):
            audit_run(self.root)

    def test_changed_source_dicom_rejected(self):
        (self.root / "source.dcm").write_bytes(b"different content")
        with self.assertRaisesRegex(ValueError, "Source DICOM"):
            audit_run(self.root)

    def test_changed_checkpoint_predictions_rejected(self):
        def wrong_loader(directory, device="cpu"):
            result = self.loader(directory, device)
            result.predict_proba = lambda x: 1 - x[:, 0]
            return result
        with patch.object(models, "load_model", side_effect=wrong_loader), self.assertRaisesRegex(ValueError, "Checkpoint predictions"):
            audit_run(self.root)

    def test_changed_frozen_source_snapshot_rejected(self):
        with zipfile.ZipFile(self.root / "source_snapshot.zip", "w") as archive:
            archive.writestr("src/ai4ct/reproduction/models.py", b"wrong source")
        with self.assertRaisesRegex(ValueError, "snapshot content"):
            audit_run(self.root)


if __name__ == "__main__":
    unittest.main()
