"""Run fixed paper algorithms on one explicitly matched cohort.

All models share predeclared group folds and train-only outlier masks. No test
labels enter fitting, preprocessing, model selection or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.ensemble import IsolationForest

from ai4ct.paths import PROJECT_ROOT, OUTPUT_ROOT
from .evaluation import make_folds, classification_metrics, group_bootstrap_accuracy

MODEL_NAMES = ("gbdt", "mlp", "pqc_hybrid", "qfm_vqc")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_inliers(x_train, seed, contamination=0.1):
    """Accept training features only; never fit a filter on evaluation samples."""
    if contamination == 0:
        return np.ones(len(x_train), dtype=bool)
    detector = IsolationForest(n_estimators=100, contamination=contamination,
                               random_state=seed, n_jobs=1)
    return detector.fit_predict(x_train) == 1


def load_manifest(path):
    """Read only manifests whose mapping status has been verified by cohort preparation."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = value["samples"] if isinstance(value, dict) else value
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a nonempty samples list")
    required = {"sample_id", "group_id", "dicom_path", "gpr", "dicom_sha256", "mapping_status"}
    seen = set()
    for sample in samples:
        if not required.issubset(sample):
            raise ValueError("Manifest lacks required provenance fields")
        if sample["sample_id"] in seen:
            raise ValueError("Duplicate sample_id in manifest")
        seen.add(sample["sample_id"])
        if sample["mapping_status"] not in {"explicit_serial_and_internal_id_agree"}:
            raise ValueError("Unverified label mapping cannot be used for reported baseline accuracy")
        if not str(sample["group_id"]).strip():
            raise ValueError("A verified patient/duplicate group is required for every sample")
        if sha256(sample["dicom_path"]) != sample["dicom_sha256"]:
            raise ValueError("DICOM file content differs from the verified manifest")
    return samples


def build_features(samples, output_dir):
    from .features import extract_features, FEATURE_VERSION, FEATURE_UNITS
    names = None
    rows = []
    for index, sample in enumerate(samples, 1):
        feature = extract_features(sample["dicom_path"])
        if names is None:
            names = sorted(feature)
        if sorted(feature) != names:
            raise ValueError("Inconsistent feature schema; no sample may be silently dropped")
        values = [feature[name] for name in names]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("Nonfinite feature value encountered; inspect source rather than imputing silently")
        rows.append(values)
        if index % 50 == 0 or index == len(samples):
            print(f"Features: {index}/{len(samples)}", flush=True)
    with (output_dir / "features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", *names])
        for sample, values in zip(samples, rows):
            writer.writerow([sample["sample_id"], *values])
    write_json(output_dir / "feature_schema.json", {
        "version": FEATURE_VERSION, "ordered_names": names,
        "units": {name: FEATURE_UNITS[name] for name in names},
        "aggregation": "plan-level MU-weighted single-layer geometry adaptation; see protocol document",
    })
    return np.asarray(rows, dtype=np.float64), names


def run_experiment(samples, manifest_path, output_dir, *, models=MODEL_NAMES,
                   thresholds=(95., 90.), seed=42, n_splits=5, device="cpu",
                   contamination=0.1, threads=1):
    from .models import train_model, load_model
    import torch

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; select a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    gpr = np.array([sample["gpr"] for sample in samples], dtype=float)
    groups = np.array([sample["group_id"] for sample in samples], dtype=str)
    assignment = make_folds(gpr, groups, n_splits=n_splits, seed=seed)
    for threshold in thresholds:
        y = (gpr >= threshold).astype(int)
        for fold in range(n_splits):
            if len(np.unique(y[assignment == fold])) != 2 or len(np.unique(y[assignment != fold])) != 2:
                raise ValueError("Every training and test fold must contain both classes; review split feasibility")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_doi": "10.1016/j.radphyschem.2026.113981",
        "experiment_type": "paper model reconstruction with single-layer plan-level cohort adaptation",
        "manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": sha256(manifest_path),
        "n_samples": len(samples), "n_groups": len(set(groups)), "models": list(models),
        "thresholds": list(thresholds), "positive_class": "pass=1 if GPR>=threshold; fail=0 otherwise",
        "gamma_criteria": "2%/2mm as recorded by source workbook; detector protocol not assumed identical to paper",
        "seed": seed, "folds": n_splits, "split": "StratifiedGroupKFold; joint GPR bins <90,90..95,>=95",
        "primary_holdout": "fold 0, fixed before fitting; other folds provide out-of-fold validation, not tuning",
        "normalization": "none (paper section 2.2)",
        "outliers": {"method": "IsolationForest", "contamination": contamination, "n_estimators": 100,
                     "fit_scope": "training fold only; all test samples retained", "seed_rule": "seed+fold"},
        "device": device, "threads": threads, "python": sys.version, "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ("numpy", "scikit-learn", "torch", "pydicom")},
        "source_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p)
                          for p in sorted((PROJECT_ROOT / "src/ai4ct/reproduction").glob("*.py"))},
        "adaptations": ["single-layer 17 features instead of unavailable dual-layer 33 features",
                        "plan-level labels instead of per-field Portal Dosimetry",
                        "patient and geometrically duplicate plans grouped in splits",
                        "outlier filter fit on training fold, not before the global split",
                        "published hyperparameters fixed; no new hyperparameter search",
                        "full-batch training and other unspecified details are explicit reconstruction choices"],
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "splits.json", [{"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                                             "test_fold": int(fold)} for sample, fold in zip(samples, assignment)])
    write_json(output_dir / "status.json", {"status": "running", "completed_fits": 0})
    x, feature_names = build_features(samples, output_dir)
    write_json(output_dir / "feature_names.json", feature_names)
    retained = {}
    for fold in range(n_splits):
        train = np.flatnonzero(assignment != fold)
        retained[fold] = train[training_inliers(x[train], seed+fold, contamination)]
    write_json(output_dir / "training_filter.json", [{
        "fold": fold,
        "train_before": int((assignment != fold).sum()), "train_after": len(retained[fold]),
        "retained_sample_ids": [samples[i]["sample_id"] for i in retained[fold]],
        "test_count": int((assignment == fold).sum()),
    } for fold in range(n_splits)])
    all_results = {}
    completed = 0
    for threshold in thresholds:
        label = f"gpr_{threshold:g}"
        y = (gpr >= threshold).astype(int)
        oof = {model: np.full(len(y), np.nan) for model in (*models, "majority")}
        fold_results = {model: [] for model in (*models, "majority")}
        for fold in range(n_splits):
            train = retained[fold]
            test = np.flatnonzero(assignment == fold)
            if len(np.unique(y[train])) != 2:
                raise ValueError("Training-only outlier filter removed an entire class; no result reported")
            majority = int(np.bincount(y[train], minlength=2).argmax())
            oof["majority"][test] = majority
            fold_results["majority"].append(classification_metrics(y[test], oof["majority"][test]))
            for name in models:
                print(f"Train {label} fold {fold+1}/{n_splits} {name}: train={len(train)}, test={len(test)}", flush=True)
                started = time.perf_counter()
                fitted = train_model(name, x[train], y[train], seed=seed+fold,
                                     config={"device": device, "batch_size": None, "num_threads": threads})
                probabilities = np.asarray(fitted.predict_proba(x[test]), dtype=float)
                metrics = classification_metrics(y[test], probabilities)
                metrics["fit_and_predict_seconds"] = time.perf_counter() - started
                model_dir = output_dir / label / f"fold_{fold}" / name
                fitted.save(model_dir)
                reloaded = load_model(model_dir, device=device)
                reloaded_p = np.asarray(reloaded.predict_proba(x[test]), dtype=float)
                if not np.allclose(probabilities, reloaded_p, rtol=1e-6, atol=1e-7):
                    raise RuntimeError("Saved model failed prediction roundtrip verification")
                metrics["model_roundtrip_verified"] = True
                oof[name][test] = probabilities
                fold_results[name].append(metrics)
                write_json(model_dir / "heldout_metrics.json", metrics)
                completed += 1
                write_json(output_dir / "status.json", {"status": "running", "completed_fits": completed})
                print(f"  accuracy={metrics['accuracy']:.4f}, balanced={metrics['balanced_accuracy']:.4f}, fail_recall={metrics['specificity_fail']:.4f}", flush=True)
        result = {}
        for name, probabilities in oof.items():
            metrics = classification_metrics(y, probabilities)
            metrics["accuracy_ci95"] = group_bootstrap_accuracy(y, probabilities, groups, seed=seed)
            result[name] = {"oof": metrics, "primary_holdout_fold0": fold_results[name][0],
                            "folds": fold_results[name]}
        with (output_dir / f"{label}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "group_id", "fold", "gpr", "label_pass", *[f"{m}_pass_probability" for m in oof]])
            for i, sample in enumerate(samples):
                writer.writerow([sample["sample_id"], groups[i], int(assignment[i]), float(gpr[i]), int(y[i]),
                                 *[float(probabilities[i]) for probabilities in oof.values()]])
        all_results[label] = result
        write_json(output_dir / "results.json", all_results)
    write_json(output_dir / "status.json", {"status": "complete", "completed_fits": completed,
                                            "n_samples": len(samples), "n_groups": len(set(groups))})
    return all_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Verified cohort manifest from ai4ct.reproduction.cohort")
    parser.add_argument("--output", type=Path, required=True, help="New, empty run directory")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--thresholds", default="95,90")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args(argv)
    models = tuple(args.models.split(","))
    if not models or len(set(models)) != len(models) or set(models) - set(MODEL_NAMES):
        parser.error("Choose distinct supported model names: " + ",".join(MODEL_NAMES))
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 < t <= 100 for t in thresholds):
        parser.error("Thresholds must be distinct percentages in (0, 100]")
    if not 0 <= args.contamination <= 0.5:
        parser.error("Contamination must be between 0 and 0.5")
    samples = load_manifest(args.manifest)
    try:
        run_experiment(samples, args.manifest, args.output, models=models, thresholds=thresholds,
                       seed=args.seed, n_splits=args.folds, device=args.device,
                       contamination=args.contamination, threads=args.threads)
    except Exception as exc:
        status = args.output / "status.json"
        if status.exists():
            previous = json.loads(status.read_text(encoding="utf-8"))
            if previous.get("status") == "running":
                write_json(status, {**previous, "status": "failed", "error_type": type(exc).__name__})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
