"""Synthetic artifact tests; no patient data or actual model fitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ai4ct.qbench import audit, data, evaluation, runner


class FakeFitted:
    def __init__(self, bias=0.0):
        self.bias = bias

    def predict_score(self, x):
        return np.asarray(x)[:, 0] / 10 + self.bias

    def predict_proba(self, x):
        return 1 / (1 + np.exp(-self.predict_score(x)))


def fake_fit(model_id, x, y, *, seed, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir()
    bias = float(np.mean(x[:, 1]) / 100)
    data.write_json(output_dir / "model.json", {"bias": bias, "seed": seed, "n": len(y)})
    data.write_json(output_dir / "metadata.json", {"model_id": model_id, "seed": seed, "n_train": len(y),
                                                   "checkpoint_epoch_snapshots": "not_applicable", "config": config})
    return FakeFitted(bias)


def fake_load(path):
    return FakeFitted(data.read_json(Path(path) / "model.json")["bias"])


class QBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        names = [f"f{i}" for i in range(17)]
        samples, splits = [], []
        for i in range(20):
            raw = self.root / f"source_{i}.bin"
            raw.write_bytes(str(i).encode())
            samples.append({"sample_id": f"sample-{i}", "group_id": f"group-{i}",
                            "gpr": [85.0, 92.0, 97.0, 99.0][i % 4],
                            "mapping_status": "explicit_serial_and_internal_id_agree", "dicom_path": str(raw),
                            "dicom_sha256": data.sha256(raw)})
            splits.append({"sample_id": f"sample-{i}", "group_id": f"group-{i}", "test_fold": i // 4})
        data.write_json(self.root / "manifest.json", {"samples": samples})
        data.write_json(self.root / "splits.json", list(reversed(splits)))
        data.write_json(self.root / "schema.json", {"version": "synthetic17", "ordered_names": names})
        with (self.root / "features.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", *names])
            for i, sample in enumerate(samples):
                writer.writerow([sample["sample_id"], *[i + k - 10 for k in range(17)]])
        self.config = {"executable": True, "cohorts": {"new": {"manifest": str(self.root / "manifest.json"),
            "manifest_sha256": data.sha256(self.root / "manifest.json"), "samples": 20, "groups": 20,
            "features": 17, "feature_version": "synthetic17"}},
            "features": {"features_path": str(self.root / "features.csv"), "features_sha256": data.sha256(self.root / "features.csv"),
                         "feature_schema_path": str(self.root / "schema.json"), "feature_schema_sha256": data.sha256(self.root / "schema.json")},
            "evaluation": {"gpr_thresholds": [95.0, 90.0], "outer_folds": 5, "training_seeds": [17, 42],
                           "split_seed": 42, "outer_split_source": str(self.root / "splits.json"),
                           "outer_split_sha256": data.sha256(self.root / "splits.json"), "outer_split_regeneration_allowed": False},
            "active_model_ids": ["synthetic_model"],
            "model_configs": {"synthetic_model": {"family": "classical", "track": "standard_v1", "config": {}}},
            "execution": {"max_wall_seconds_per_trial": 10, "max_total_device_seconds": 100}}
        self.source = self.root / "source.py"
        self.source.write_text("# immutable synthetic source\n", encoding="utf-8")
        self.source_patch = patch.object(runner, "source_files", return_value={"src/ai4ct/stub.py": self.source})
        self.model_patch = patch.object(runner, "model_functions", return_value=(fake_fit, fake_load))
        self.source_patch.start()
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.source_patch.stop()
        self.temp.cleanup()

    def execute(self, path, timeout):
        runner.execute_task(path)
        return {**data.read_json(Path(data.read_json(path)["output_dir"]) / "result.json"), "wall_seconds": .01}

    def run_fixture(self, name="run", folds=None, seeds=None):
        output = self.root / name
        runner.run_shard(self.config, output, folds=folds, seeds=seeds, executor=self.execute)
        return output

    def test_metrics_margin_is_not_a_probability(self):
        result = evaluation.metrics([0, 1], [-3, 2])
        self.assertEqual(result["accuracy"], 1)
        self.assertIsNone(result["log_loss"])
        self.assertIsNone(result["brier"])
        self.assertEqual(result["average_precision_fail"], 1)
        self.assertEqual(evaluation.metrics([0], [0])["accuracy"], 1)
        with self.assertRaises(ValueError):
            evaluation.metrics([0, 1], [-3, 2], [-3, 2])

    def test_frozen_folds_join_by_id_not_order(self):
        dataset = data.load_dataset(self.config)
        np.testing.assert_array_equal(dataset.fold, np.repeat(np.arange(5), 4))
        self.assertEqual(dataset.x.shape, (20, 17))

    def test_changed_source_hash_rejected(self):
        (self.root / "source_0.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "DICOM"):
            data.load_dataset(self.config)

    def test_feature_reordering_rejected_even_with_updated_file_hash(self):
        path = self.root / "features.csv"
        lines = path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        path.write_text("\n".join(lines), encoding="utf-8")
        self.config["features"]["features_sha256"] = data.sha256(path)
        with self.assertRaisesRegex(ValueError, "sample order"):
            data.load_dataset(self.config)

    def test_group_crossing_fold_rejected(self):
        manifest = data.read_json(self.root / "manifest.json")
        manifest["samples"][4]["group_id"] = "group-0"
        data.write_json(self.root / "manifest.json", manifest)
        splits = data.read_json(self.root / "splits.json")
        next(s for s in splits if s["sample_id"] == "sample-4")["group_id"] = "group-0"
        data.write_json(self.root / "splits.json", splits)
        self.config["cohorts"]["new"].update(groups=19, manifest_sha256=data.sha256(self.root / "manifest.json"))
        self.config["evaluation"]["outer_split_sha256"] = data.sha256(self.root / "splits.json")
        with self.assertRaisesRegex(ValueError, "crosses"):
            data.load_dataset(self.config)

    def test_full_mock_run_replay_and_oof(self):
        output = self.run_fixture()
        report, tasks, _, _ = audit.audit_run(output)
        self.assertEqual(report["checkpoint_replays"], 20)
        result = audit.aggregate([output], self.root / "aggregate")
        self.assertTrue(result["full_catalog_coverage"])
        self.assertEqual(len(result["per_model_threshold_seed_oof"]), 4)
        self.assertEqual(len(tasks), 20)

    def test_partial_shards_do_not_produce_complete_ranking(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        result = audit.aggregate([output], self.root / "aggregate")
        self.assertEqual(result["ranking_status"], "withheld_incomplete_coverage")
        self.assertEqual(result["per_model_threshold_seed_oof"], {})

    def test_resume_completed_replays_without_refit(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        def forbidden(*args):
            raise AssertionError("resume must not train successful task")
        result = runner.run_shard(self.config, output, folds=[0], seeds=[17], resume_completed=True, executor=forbidden)
        self.assertEqual(result["succeeded"], 2)
        self.source.write_text("# changed source\n")
        with self.assertRaisesRegex(ValueError, "Resume"):
            runner.run_shard(self.config, output, folds=[0], seeds=[17], resume_completed=True, executor=forbidden)

    def test_timeout_is_retained_and_not_counted_success(self):
        output = self.root / "timeout"
        runner.run_shard(self.config, output, folds=[0], seeds=[17], executor=lambda *args: {"status": "timed_out", "wall_seconds": .01})
        report, tasks, _, _ = audit.audit_run(output)
        self.assertEqual(tasks, [])
        self.assertEqual([r["status"] for r in report["task_states"]], ["timed_out", "timed_out"])
        self.assertFalse(report["all_selected_tasks_succeeded"])

    def test_predictions_cannot_change_gpr_even_with_new_checksum(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        task = runner.expected_tasks(self.config, folds=[0], seeds=[17])[0]
        state_path = output / "tasks" / task["task_id"] / "status.json"
        state = data.read_json(state_path)
        attempt = output / state["attempt"]
        rows = data.read_json(attempt / "predictions.json")
        rows[0]["gpr"] = 86.0
        data.write_json(attempt / "predictions.json", rows)
        result = data.read_json(attempt / "result.json")
        result["predictions_sha256"] = data.sha256(attempt / "predictions.json")
        data.write_json(attempt / "result.json", result)
        data.write_json(state_path, {**state, **result})
        with self.assertRaisesRegex(ValueError, "GPR"):
            audit.audit_run(output)

    def test_loader_prediction_mismatch_rejected(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        with patch.object(runner, "model_functions", return_value=(fake_fit, lambda p: FakeFitted(50))):
            with self.assertRaisesRegex(ValueError, "scores differ"):
                audit.audit_run(output)

    def test_source_snapshot_change_rejected(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        with (output / "source_snapshot.zip").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "archive hash"):
            audit.audit_run(output)

    def test_supervisor_kills_timed_out_worker(self):
        directory = self.root / "timeout_worker"
        directory.mkdir()
        request = directory / "request.json"
        data.write_json(request, {"output_dir": str(directory)})
        with patch.object(runner.subprocess, "Popen") as spawn:
            process = spawn.return_value
            process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 1]
            state = runner.supervise(request, 1)
            self.assertEqual(state["status"], "timed_out")
            process.kill.assert_called_once()

    def test_worker_reports_original_timeout_type(self):
        directory = self.root / "error_worker"
        directory.mkdir()
        request = directory / "request.json"
        data.write_json(request, {"output_dir": str(directory)})
        with patch.object(runner, "execute_task", side_effect=TimeoutError("budget exhausted")), patch.object(runner.traceback, "print_exc"):
            self.assertEqual(runner.main(["--execute-task", str(request)]), 1)
        with patch.object(runner.subprocess, "Popen") as spawn:
            spawn.return_value.wait.return_value = 1
            state = runner.supervise(request, 1)
        self.assertEqual(state["status"], "timed_out")
        self.assertEqual(state["error_type"], "TimeoutError")

    def test_registered_epoch_snapshots_must_exist(self):
        output = self.run_fixture(folds=[0], seeds=[17])
        task = runner.expected_tasks(self.config, folds=[0], seeds=[17])[0]
        state_path = output / "tasks" / task["task_id"] / "status.json"
        state = data.read_json(state_path)
        attempt = output / state["attempt"]
        metadata = data.read_json(attempt / "model/metadata.json")
        metadata.update(checkpoint_epoch_snapshots="saved", config={"epochs": 1})
        data.write_json(attempt / "model/metadata.json", metadata)
        result = data.read_json(attempt / "result.json")
        result["model_file_sha256"] = runner.model_file_hashes(attempt / "model")
        data.write_json(attempt / "result.json", result)
        data.write_json(state_path, {**state, **result})
        with self.assertRaisesRegex(ValueError, "checkpoint missing"):
            audit.audit_run(output)


if __name__ == "__main__":
    unittest.main()
