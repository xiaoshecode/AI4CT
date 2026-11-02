"""Refit a predeclared fold from saved inputs to check numerical repeatability."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .models import train_model
from .run import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=95.)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args(argv)
    folder = args.output_dir
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    key = f"gpr_{args.threshold:g}"
    names = json.loads((folder / "feature_names.json").read_text(encoding="utf-8"))
    with (folder / "features.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    positions = {row["sample_id"]: i for i, row in enumerate(rows)}
    x = np.array([[float(row[name]) for name in names] for row in rows])
    with (folder / f"{key}_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = {row["sample_id"]: row for row in csv.DictReader(stream)}
    y = np.array([int(predictions[row["sample_id"]]["label_pass"]) for row in rows])
    filters = json.loads((folder / "training_filter.json").read_text(encoding="utf-8"))
    selected = next(record for record in filters if record["fold"] == args.fold)
    train = np.array([positions[sample] for sample in selected["retained_sample_ids"]])
    test = np.array([i for i, row in enumerate(rows) if int(predictions[row["sample_id"]]["fold"]) == args.fold])
    checks = {}
    for name in protocol["models"]:
        metadata = json.loads((folder / key / f"fold_{args.fold}" / name / "metadata.json").read_text(encoding="utf-8"))
        model = train_model(name, x[train], y[train], seed=metadata["seed"], config=metadata["config"])
        current = model.predict_proba(x[test])
        previous = np.array([float(predictions[rows[i]["sample_id"]][f"{name}_pass_probability"]) for i in test])
        delta = float(np.abs(current - previous).max())
        passed = bool(np.allclose(current, previous, rtol=1e-6, atol=1e-7))
        checks[name] = {"passed": passed, "maximum_probability_difference": delta}
        if not passed:
            raise RuntimeError(f"Refit repeatability check failed: {name}")
    report = {"status": "passed", "threshold": args.threshold, "fold": args.fold,
              "scope": "Same saved inputs, seed, parameters and local software/hardware; does not test cross-platform identity", "models": checks}
    write_json(folder / "repeatability.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
