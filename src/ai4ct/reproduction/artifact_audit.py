"""Audit persisted cohort, features, source snapshot and checkpoints without fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .evaluation import classification_metrics, group_bootstrap_accuracy, make_folds


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_metrics(expected, actual):
    for key, value in actual.items():
        _require(key in expected, "A saved metric is missing.")
        if isinstance(value, float):
            _require(isinstance(expected[key], (int, float)) and
                     np.isclose(value, expected[key], rtol=1e-10, atol=1e-12),
                     "Saved metric differs from replayed predictions.")
        else:
            _require(value == expected[key], "Saved metric differs from replayed predictions.")


def _audit(output_dir: Path, device: str) -> dict:
    from . import models as model_module

    protocol = _read(output_dir / "protocol.json")
    status = _read(output_dir / "status.json")
    _require(status.get("status") == "complete", "Run is not complete.")
    n_folds = protocol["folds"]
    thresholds = protocol["thresholds"]
    model_names = protocol["models"]
    expected_fits = len(thresholds) * n_folds * len(model_names)
    _require(status.get("completed_fits") == expected_fits, "Completed fit count is inconsistent.")
    manifest_path = Path(protocol["manifest_path"])
    _require(_sha(manifest_path) == protocol["manifest_sha256"], "Frozen manifest hash changed.")
    manifest = _read(manifest_path)
    samples = manifest["samples"] if isinstance(manifest, dict) else manifest
    ids = [row["sample_id"] for row in samples]
    _require(len(ids) == len(set(ids)) == protocol["n_samples"], "Manifest IDs/count do not match protocol.")
    sample_index = {row["sample_id"]: row for row in samples}
    groups = np.asarray([row["group_id"] for row in samples], dtype=str)
    rates = np.asarray([row["gpr"] for row in samples], dtype=float)
    _require(len(set(groups)) == protocol["n_groups"], "Manifest group count differs from protocol.")
    for sample in samples:
        _require(sample["mapping_status"] == "explicit_serial_and_internal_id_agree", "Manifest contains unverified sample mapping.")
        _require(_sha(sample["dicom_path"]) == sample["dicom_sha256"], "Source DICOM content differs from manifest.")

    snapshot_path = output_dir / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot_path) as snapshot:
        names = snapshot.namelist()
        for source_name, expected in protocol["source_sha256"].items():
            _require(names.count(source_name) == 1, "Frozen source snapshot is incomplete or ambiguous.")
            _require(hashlib.sha256(snapshot.read(source_name)).hexdigest() == expected,
                     "Source snapshot content differs from recorded training source hash.")
    loader_key = "src/ai4ct/reproduction/models.py"
    _require(_sha(model_module.__file__) == protocol["source_sha256"].get(loader_key),
             "Current checkpoint loader differs from the source used for fitting.")

    splits = _read(output_dir / "splits.json")
    _require([row["sample_id"] for row in splits] == ids, "Split IDs/order differ from manifest.")
    for row in splits:
        _require(row["group_id"] == sample_index[row["sample_id"]]["group_id"], "Split groups differ from manifest.")
        _require(type(row["test_fold"]) is int and 0 <= row["test_fold"] < n_folds, "Invalid held-out fold index.")
    assignment = np.asarray([row["test_fold"] for row in splits])
    expected_assignment = make_folds(rates, groups, n_splits=n_folds, seed=protocol["seed"])
    _require(np.array_equal(assignment, expected_assignment), "Saved folds differ from declared deterministic split.")
    for group in np.unique(groups):
        _require(len(set(assignment[groups == group])) == 1, "A patient/duplicate group crosses folds.")

    feature_headers, feature_rows = _csv(output_dir / "features.csv")
    feature_names = _read(output_dir / "feature_names.json")
    schema = _read(output_dir / "feature_schema.json")
    _require(feature_headers == ["sample_id", *feature_names] and feature_names == schema["ordered_names"],
             "Feature column order differs from recorded schema.")
    _require([row["sample_id"] for row in feature_rows] == ids, "Feature sample order differs from manifest.")
    x = np.asarray([[float(row[name]) for name in feature_names] for row in feature_rows])
    _require(x.shape == (len(samples), len(feature_names)) and np.isfinite(x).all(), "Invalid feature matrix.")

    filters = _read(output_dir / "training_filter.json")
    _require(len(filters) == n_folds and {row["fold"] for row in filters} == set(range(n_folds)),
             "Training filter records do not cover every fold exactly once.")
    filter_index = {row["fold"]: row for row in filters}
    id_positions = {sample_id: position for position, sample_id in enumerate(ids)}
    for fold, record in filter_index.items():
        retained = record["retained_sample_ids"]
        _require(len(retained) == len(set(retained)) == record["train_after"], "Training IDs/count are invalid.")
        _require(set(retained).issubset(id_positions), "Training ID is outside the frozen cohort.")
        _require(record["train_before"] == int((assignment != fold).sum()) and
                 record["test_count"] == int((assignment == fold).sum()), "Training/test counts differ from split.")
        _require(all(assignment[id_positions[sample_id]] != fold for sample_id in retained),
                 "Held-out sample is present in training IDs.")

    results = _read(output_dir / "results.json")
    replayed, metric_scopes, prediction_digests = [], 0, {}
    aliases = {"pqc_hybrid": "pqc", "qfm_vqc": "qfm"}
    for threshold in thresholds:
        key = f"gpr_{threshold:g}"
        prediction_path = output_dir / f"{key}_predictions.csv"
        _, prediction_rows = _csv(prediction_path)
        _require([row["sample_id"] for row in prediction_rows] == ids, "Prediction sample order differs from manifest.")
        y = (rates >= threshold).astype(int)
        for position, row in enumerate(prediction_rows):
            _require(float(row["gpr"]) == rates[position], "Prediction GPR differs from frozen manifest.")
            _require(row["group_id"] == groups[position], "Prediction group differs from frozen manifest.")
            _require(int(row["fold"]) == assignment[position], "Prediction fold differs from frozen split.")
            _require(int(row["label_pass"]) == y[position], "Prediction label differs from manifest GPR/threshold.")
        prediction_digests[prediction_path.name] = _sha(prediction_path)
        all_probabilities = {}
        for name in [*model_names, "majority"]:
            probabilities = np.asarray([float(row[f"{name}_pass_probability"]) for row in prediction_rows])
            all_probabilities[name] = probabilities
            _same_metrics(results[key][name]["oof"], classification_metrics(y, probabilities))
            _same_metrics(results[key][name]["primary_holdout_fold0"],
                          classification_metrics(y[assignment == 0], probabilities[assignment == 0]))
            _require(len(results[key][name]["folds"]) == n_folds, "Per-fold metric count is inconsistent.")
            metric_scopes += 2
            for fold in range(n_folds):
                mask = assignment == fold
                _same_metrics(results[key][name]["folds"][fold], classification_metrics(y[mask], probabilities[mask]))
                metric_scopes += 1
            ci = results[key][name]["oof"]["accuracy_ci95"]
            recomputed_ci = group_bootstrap_accuracy(y, probabilities, groups, seed=protocol["seed"], repetitions=ci["repetitions"])
            _same_metrics(ci, recomputed_ci)
        for fold in range(n_folds):
            mask = assignment == fold
            train_positions = [id_positions[sample_id] for sample_id in filter_index[fold]["retained_sample_ids"]]
            majority = int(np.bincount(y[train_positions], minlength=2).argmax())
            _require(np.all(all_probabilities["majority"][mask] == majority), "Majority baseline is not derived from retained training labels.")
            for name in model_names:
                directory = output_dir / key / f"fold_{fold}" / name
                fitted = model_module.load_model(directory, device=device)
                _require(fitted.name == aliases.get(name, name) and fitted.n_features == x.shape[1],
                         "Checkpoint model/feature metadata mismatch.")
                _require(fitted.seed == protocol["seed"] + fold, "Checkpoint seed differs from declared fold seed.")
                replay = np.asarray(fitted.predict_proba(x[mask]), dtype=float)
                stored = all_probabilities[name][mask]
                _require(replay.shape == stored.shape and np.isfinite(replay).all(), "Checkpoint produced invalid replay predictions.")
                _require(np.allclose(replay, stored, rtol=1e-6, atol=1e-7), "Checkpoint predictions differ from saved held-out probabilities.")
                _require(np.array_equal(replay > 0.5, stored > 0.5), "Checkpoint replay changes classification at the decision boundary.")
                checkpoint = directory / ("model.joblib" if name == "gbdt" else "weights.pt")
                replayed.append({"threshold": threshold, "fold": fold, "model": name,
                                 "heldout_count": int(mask.sum()), "max_probability_error": float(np.max(np.abs(replay - stored))),
                                 "checkpoint_sha256": _sha(checkpoint), "metadata_sha256": _sha(directory / "metadata.json")})
    _require(len(replayed) == expected_fits, "Not every checkpoint was independently replayed.")
    return {
        "status": "passed", "audit_schema": "paper-artifact-audit-v1", "audit_source_sha256": _sha(__file__),
        "n_samples": len(samples), "n_groups": len(set(groups)), "source_dicoms_verified": len(samples),
        "source_snapshot_files_verified": len(protocol["source_sha256"]), "source_snapshot_sha256": _sha(snapshot_path),
        "features_sha256": _sha(output_dir / "features.csv"), "prediction_files_sha256": prediction_digests,
        "manifest_sha256": protocol["manifest_sha256"], "metric_scopes_recomputed": metric_scopes,
        "checkpoints_replayed": len(replayed), "checkpoint_replays": replayed,
        "checks": ["source DICOM hashes", "frozen source snapshot hashes", "unchanged manifest",
                   "manifest/split/prediction IDs, GPR and groups", "declared deterministic group folds",
                   "feature row and column order", "complete train-only filter records", "majority from training labels",
                   "OOF, primary holdout, all fold metrics and group bootstrap intervals",
                   "every saved checkpoint replayed on its own held-out features, including class-boundary equality"],
        "scope": "Artifact integrity and saved-model inference only; no refitting and no independent clinical mapping adjudication.",
    }


def audit_run(output_dir, device="cpu") -> dict:
    """Write a fresh audit result; a failed re-audit cannot leave a stale pass."""
    directory = Path(output_dir)
    try:
        report = _audit(directory, device)
    except Exception as exc:
        report = {"status": "failed", "audit_schema": "paper-artifact-audit-v1", "error_type": type(exc).__name__}
        (directory / "artifact_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    (directory / "artifact_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    report = audit_run(args.output_dir, args.device)
    print(json.dumps({key: value for key, value in report.items() if key != "checkpoint_replays"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
