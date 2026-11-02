import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from ai4ct.qbench.data import sha256
from ai4ct.reproduction.features import FEATURE_NAMES, FEATURE_UNITS, FEATURE_VERSION
from ai4ct.reproduction.cohort_shift import (
    check_artifact_hashes, check_fitted_scope, check_version_identity, domain_metrics, feature_statistics, feature_table,
    fit_domain_models, split_assignment, union_groups, validate_config, view_indices,
)


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = sorted(FEATURE_NAMES)
        self.schema = {"version": FEATURE_VERSION, "ordered_names": self.names,
                       "units": {n: FEATURE_UNITS[n] for n in self.names}}

    def table(self, ids, values=None, names=None):
        path = self.root / "features.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", *(names or self.names)])
            for i, key in enumerate(ids):
                writer.writerow([key, *((values[i] if values is not None else [i] * len(self.names)))])
        return path

    def test_explicit_key_join_and_column_order(self):
        path = self.table(["b", "a"])
        x = feature_table(path, self.schema, ["a", "b"])
        np.testing.assert_array_equal(x[:, 0], [1, 0])
        path = self.table(["a", "b"], names=list(reversed(self.names)))
        with self.assertRaisesRegex(ValueError, "columns"):
            feature_table(path, self.schema, ["a", "b"])

    def test_duplicate_missing_and_extra_keys_refused(self):
        for ids in (["a", "a"], ["a"], ["a", "b", "c"]):
            path = self.table(ids)
            with self.assertRaisesRegex(ValueError, "sample keys"):
                feature_table(path, self.schema, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "sample keys"):
            feature_table(self.table(["a", "b"]), self.schema, ["a", "a"])

    def test_nonfinite_and_changed_units_refused(self):
        values = [[0] * len(self.names), [1] * len(self.names)]
        values[0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "matrix"):
            feature_table(self.table(["a", "b"], values), self.schema, ["a", "b"])
        wrong = dict(self.schema, units={**self.schema["units"], "MU": "cGy"})
        with self.assertRaisesRegex(ValueError, "schema"):
            feature_table(self.table(["a", "b"]), wrong, ["a", "b"])

    def test_frozen_protocol_refuses_qa_or_changed_tasks(self):
        config = {"schema": "cohort-shift-v1", "old": {"expected_samples": 644},
                  "new": {"expected_samples": 309}, "models": ["logistic", "random_forest"],
                  "views": ["all_plans", "one_per_geometry"],
                  "evaluation": {"folds": 5, "seeds": [17, 42, 2026]}}
        validate_config(config)
        config["evaluation"]["seeds"] = [42]
        with self.assertRaises(ValueError):
            validate_config(config)


class GroupAndStatisticsTests(unittest.TestCase):
    def test_transitive_links_and_source_scoping(self):
        rows = [
            {"sample_id": "old:a", "cohort": "old", "geometry_sha256": "A"},
            {"sample_id": "new:a", "cohort": "new", "geometry_sha256": "A", "patient_group_id": "p"},
            {"sample_id": "new:b", "cohort": "new", "geometry_sha256": "B", "patient_group_id": "p"},
            {"sample_id": "old:b", "cohort": "old", "geometry_sha256": "C", "patient_group_id": "ignored"},
            {"sample_id": "old:c", "cohort": "old", "geometry_sha256": "D", "patient_group_id": "ignored"},
        ]
        x = np.array([[0], [0], [2], [2], [4]], dtype=float)
        groups = union_groups(rows, x)
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertNotEqual(groups[3], groups[4])
        self.assertEqual(view_indices(rows, "one_per_geometry").tolist(), list(range(5)))
        rows.append({"sample_id": "old:z", "cohort": "old", "geometry_sha256": "A"})
        self.assertEqual(view_indices(rows, "one_per_geometry").tolist(), list(range(5)))

    def test_statistics_have_known_direction_and_no_independent_p_values(self):
        x = np.array([[0., 5., 0.], [2., 5., 0.], [2., 5., 7.], [4., 5., 7.]])
        stats = feature_statistics(x, np.array([0, 0, 1, 1]), ["a", "b", "c"],
                                   {"a": "unit", "b": "unit", "c": "unit"})
        self.assertAlmostEqual(stats[0]["smd_new_minus_old"], np.sqrt(2))
        self.assertEqual(stats[0]["wasserstein"], 2)
        self.assertEqual(stats[0]["ks_statistic"], .5)
        self.assertEqual(stats[1]["ks_statistic"], 0)
        self.assertIsNone(stats[1]["smd_new_minus_old"])
        self.assertEqual(stats[2]["ks_statistic"], 1)
        self.assertIsNone(stats[2]["wasserstein_over_pooled_sd"])
        self.assertFalse(any("pvalue" in key for row in stats for key in row))

    def test_domain_metrics_and_ties(self):
        result = domain_metrics(np.array([0, 0, 1, 1]), np.array([.1, .5, .5, .9]), np.array(["a", "a", "b", "c"]))
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["old_recall"], 1)
        self.assertEqual(result["new_recall"], .5)
        self.assertEqual(result["balanced_accuracy"], .75)
        with self.assertRaises(ValueError):
            domain_metrics(np.array([0, 1]), np.array([np.nan, 2]), np.array(["a", "b"]))

    def test_group_isolation_and_oof_coverage(self):
        groups = np.repeat(np.arange(30).astype(str), 2)
        y = np.repeat(np.arange(30) % 2, 2)
        x = np.arange(120).reshape(60, 2)
        fold = split_assignment(x, y, groups, seed=17)
        self.assertEqual(set(fold), set(range(5)))
        for k in range(5):
            self.assertFalse(set(groups[fold == k]) & set(groups[fold != k]))
            self.assertEqual(set(y[fold == k]), {0, 1})


class ModelArtifactTests(unittest.TestCase):
    def test_frozen_source_environment_and_snapshot_enforced(self):
        source = b"example = 42\n"
        hashes = {"src/ai4ct/example.py": hashlib.sha256(source).hexdigest()}
        config_bytes = b'{"schema": "synthetic"}'
        env = {"python": "synthetic", "packages": {"numpy": "example"}}
        protocol = {"source_hashes": hashes, "environment": env,
                    "config": json.loads(config_bytes), "config_sha256": hashlib.sha256(config_bytes).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot.zip"
            with zipfile.ZipFile(snapshot, "w") as archive:
                archive.writestr("src/ai4ct/example.py", source)
                archive.writestr("run_config.json", config_bytes)
            with patch("ai4ct.reproduction.cohort_shift.source_identity", return_value=(None, hashes)), \
                 patch("ai4ct.reproduction.cohort_shift.environment_identity", return_value=env):
                check_version_identity(protocol, snapshot)
                with self.assertRaisesRegex(ValueError, "environment"):
                    check_version_identity({**protocol, "environment": {}}, snapshot)
                with self.assertRaisesRegex(ValueError, "source version"):
                    check_version_identity({**protocol, "source_hashes": {}}, snapshot)
                with zipfile.ZipFile(snapshot, "w") as archive:
                    archive.writestr("src/ai4ct/example.py", b"changed")
                    archive.writestr("run_config.json", config_bytes)
                with self.assertRaisesRegex(ValueError, "snapshot file"):
                    check_version_identity(protocol, snapshot)

    def test_train_only_scaling_saved_scope_and_exact_replay(self):
        rng = np.random.default_rng(4)
        y = np.arange(50) % 2
        x = rng.normal(size=(50, 3)) + y[:, None] * .3
        x[0] = [1000, -2000, 3000]
        groups = np.arange(50).astype(str)
        records = [{"sample_id": f"sample{i:03}", "cohort": "old" if y[i] == 0 else "new",
                    "geometry_sha256": str(i)} for i in range(50)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, metrics, views = fit_domain_models(root, records, x, y, groups,
                seeds=(17,), models=("logistic",), views=("all_plans",))
            self.assertEqual(len(tasks), 5)
            self.assertEqual(len(metrics), 1)
            self.assertEqual(views["all_plans"]["samples"], 50)
            for task in tasks:
                model = joblib.load(root / task["model_path"])
                np.testing.assert_allclose(model[0].mean_, x[task["train_indices"]].mean(axis=0))
                check_fitted_scope(model, "logistic", 17, x[task["train_indices"]])
                with self.assertRaisesRegex(ValueError, "Scaler"):
                    check_fitted_scope(model, "logistic", 17, x)
                with self.assertRaisesRegex(ValueError, "parameters"):
                    check_fitted_scope(model, "logistic", 42, x[task["train_indices"]])
                self.assertFalse(np.allclose(model[0].mean_, x.mean(axis=0)))
                self.assertEqual(task["replay_max_absolute_error"], 0)
            test_indices = [i for task in tasks for i in task["test_indices"]]
            self.assertEqual(sorted(test_indices), list(range(50)))

    def test_artifact_tampering_and_escape_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.joblib"
            path.write_bytes(b"original")
            expected = {"model.joblib": sha256(path)}
            check_artifact_hashes(root, expected)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                check_artifact_hashes(root, expected)
            with self.assertRaises(ValueError):
                check_artifact_hashes(root, {"../escaped": "anything"})


if __name__ == "__main__":
    unittest.main()
