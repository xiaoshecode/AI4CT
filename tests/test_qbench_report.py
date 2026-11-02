import json
from pathlib import Path
import tempfile
import unittest

from ai4ct.qbench.report import METRICS, category, load_batches, summary_rows


class ReportTests(unittest.TestCase):
    def test_incomplete_or_unreplayed_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "protocol.json").write_text("{}", encoding="utf-8")
            for status, complete, replay in [("incomplete", False, True), ("complete", True, False)]:
                result = dict(status=status, full_catalog_coverage=complete,
                              checkpoint_replay_requested=replay, expected_tasks=30,
                              successful_unique_tasks=30 if complete else 29)
                (root / "results.json").write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_batches([root])

    def test_tables_keep_each_seed_and_nullable_probability_metrics(self):
        model, threshold, seeds = "linear_svm", 90., [17, 42, 2026]
        stats = {m: dict(mean=.8, min=.7, max=.9) for m in METRICS}
        stats.update(brier=None, log_loss=None, training_seeds=seeds)
        per_seed = {}
        for seed, value in zip(seeds, [.7, .8, .9]):
            entry = {m: value for m in METRICS}
            entry.update(brier=None, log_loss=None, accuracy_ci95=dict(low=.6, high=.95), n_groups=251)
            per_seed[f"{model}__gpr90__seed{seed}"] = entry
        batches = [dict(config=dict(active_model_ids=[model], protocol_id="test",
                                    model_configs={model: dict(track="standard_v1", family="classical")},
                                    evaluation=dict(gpr_thresholds=[threshold])),
                        results=dict(seed_summary={f"{model}__gpr90": stats}, per_model_threshold_seed_oof=per_seed))]
        rows, individual = summary_rows(batches)
        self.assertEqual(len(rows), 1)
        self.assertEqual([r["seed"] for r in individual], seeds)
        self.assertEqual([r["accuracy"] for r in individual], [.7, .8, .9])
        self.assertEqual(rows[0]["accuracy_mean"], .8)
        self.assertIsNone(rows[0]["brier_mean"])
        self.assertTrue(all(r["brier"] is None for r in individual))
        self.assertEqual(category("qboost_qubo_12_classical"), "classical_qubo_control")
        self.assertEqual(category("dressed_classical_5d"), "classical")


if __name__ == "__main__":
    unittest.main()
