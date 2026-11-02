"""Independently recompute metrics from persisted held-out predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import classification_metrics
from .run import sha256, write_json


def verify_run(output_dir):
    output_dir = Path(output_dir)
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Run is not complete")
    if sha256(protocol["manifest_path"]) != protocol["manifest_sha256"]:
        raise ValueError("Manifest has changed after the run")
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    splits = json.loads((output_dir / "splits.json").read_text(encoding="utf-8"))
    index = {row["sample_id"]: row for row in splits}
    if len(index) != len(splits) or len(splits) != protocol["n_samples"]:
        raise ValueError("Split sample IDs or sample counts are invalid")
    group_folds = {}
    for row in splits:
        group_folds.setdefault(row["group_id"], set()).add(row["test_fold"])
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A patient/duplicate group crosses test folds")
    filters = json.loads((output_dir / "training_filter.json").read_text(encoding="utf-8"))
    for record in filters:
        ids = record["retained_sample_ids"]
        if len(set(ids)) != len(ids) or len(ids) != record["train_after"]:
            raise ValueError("Training filter count/uniqueness mismatch")
        if any(index[sample]["test_fold"] == record["fold"] for sample in ids):
            raise ValueError("Held-out sample found in training IDs")
    checked = 0
    for threshold in protocol["thresholds"]:
        key = f"gpr_{threshold:g}"
        with (output_dir / f"{key}_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(index) or {row["sample_id"] for row in rows} != set(index):
            raise ValueError("Prediction records do not cover the frozen cohort exactly once")
        y = np.array([int(row["label_pass"]) for row in rows])
        rates = np.array([float(row["gpr"]) for row in rows])
        if not np.array_equal(y, (rates >= threshold).astype(int)):
            raise ValueError("Saved labels disagree with prespecified GPR threshold")
        folds = np.array([int(row["fold"]) for row in rows])
        for row in rows:
            frozen = index[row["sample_id"]]
            if int(row["fold"]) != frozen["test_fold"] or row["group_id"] != frozen["group_id"]:
                raise ValueError("Predictions do not use the frozen split")
        for name in (*protocol["models"], "majority"):
            probability = np.array([float(row[f"{name}_pass_probability"]) for row in rows])
            pairs = [("oof", classification_metrics(y, probability)),
                     ("primary_holdout_fold0", classification_metrics(y[folds == 0], probability[folds == 0]))]
            for scope, recomputed in pairs:
                original = results[key][name][scope]
                for metric, value in recomputed.items():
                    recorded = original[metric]
                    if isinstance(value, float):
                        if not np.isclose(value, recorded, rtol=1e-10, atol=1e-12):
                            raise ValueError(f"Metric mismatch: {key}/{name}/{scope}/{metric}")
                    elif value != recorded:
                        raise ValueError(f"Metric mismatch: {key}/{name}/{scope}/{metric}")
                checked += 1
    report = {"status": "passed", "n_samples": len(index), "n_groups": len(group_folds),
              "metric_scopes_recomputed": checked,
              "checks": ["complete run", "unchanged manifest", "unique cohort coverage",
                         "no group leakage", "no test IDs in training filters", "label thresholds",
                         "saved OOF and holdout metrics independently recomputed"]}
    write_json(output_dir / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify_run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
