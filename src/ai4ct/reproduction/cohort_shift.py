"""Audited covariate-shift diagnostics; target is cohort, never an inferred QA label.

Old=0 and new=1 are known source membership. This task does not estimate PSQA
accuracy, recover anonymous identities, or establish independent old patients.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import time
import zipfile

import joblib
import numpy as np
import pydicom
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai4ct.qbench.data import checked_path, digest_json, read_json, resolve_path, sha256, write_json
from .cohort import delivery_geometry_hash
from .features import FEATURE_NAMES, FEATURE_UNITS, FEATURE_VERSION, extract_features

VIEWS = ("all_plans", "one_per_geometry")
MODELS = ("logistic", "random_forest")
MODEL_PARAMETERS = {
    "logistic": {"C": 1.0, "max_iter": 5000, "class_weight": "balanced", "solver": "lbfgs"},
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
                      "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
}
LIMITATION = ("Source discrimination only, not QA accuracy. Old patient identities remain unknown; "
              "geometry/input grouping is a lower bound on dependence, not verified patient grouping. "
              "Repeated seeds reuse the same cohorts and are not independent studies.")


def validate_config(config):
    if config.get("schema") != "cohort-shift-v1":
        raise ValueError("Unknown cohort shift schema")
    if config.get("models") != list(MODELS) or config.get("views") != list(VIEWS):
        raise ValueError("This protocol fixes both domain models and both views")
    if config.get("evaluation") != {"folds": 5, "seeds": [17, 42, 2026]}:
        raise ValueError("This protocol fixes five folds and three split/model seeds")
    for cohort in ("old", "new"):
        if not isinstance(config[cohort]["expected_samples"], int) or config[cohort]["expected_samples"] < 2:
            raise ValueError("Invalid sample count")


def feature_table(path, schema, source_ids):
    """Join exact keys, refusing changed column definitions, missing or extra rows."""
    names = schema["ordered_names"]
    if (schema.get("version") != FEATURE_VERSION or names != sorted(FEATURE_NAMES)
            or schema.get("units") != {n: FEATURE_UNITS[n] for n in names}):
        raise ValueError("Unsupported feature schema, units or order")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", *names]:
            raise ValueError("Feature columns differ from frozen schema")
        rows = list(reader)
    by_id = {r["sample_id"]: r for r in rows}
    if (len(by_id) != len(rows) or len(set(source_ids)) != len(source_ids)
            or set(by_id) != set(source_ids)):
        raise ValueError("Duplicate, missing or extra sample keys")
    matrix = np.array([[float(by_id[key][name]) for name in names] for key in source_ids], dtype=float)
    if matrix.shape != (len(source_ids), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid feature matrix")
    return matrix


def union_groups(records, x):
    """Transitive patient, geometry and exact finite-feature links, without QA values."""
    if len(records) != len(x) or not np.isfinite(x).all():
        raise ValueError("Records/features differ or are nonfinite")
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        keys = [("geometry", record["geometry_sha256"]), ("features", tuple(x[i].tolist()))]
        if record["cohort"] == "new":
            for field in ("patient_group_id", "original_group_id"):
                if record.get(field):
                    keys.append((field, record[field]))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = {}
    for i, record in enumerate(records):
        members.setdefault(find(i), []).append(record["sample_id"])
    names = {group: "shift_" + digest_json(sorted(ids))[:20] for group, ids in members.items()}
    return np.array([names[find(i)] for i in range(len(records))])


def view_indices(records, view):
    if view == "all_plans":
        return np.arange(len(records))
    if view != "one_per_geometry":
        raise ValueError("Unknown view")
    # One representative per geometry WITHIN each source; shared cross-source
    # geometries remain represented on both sides and connected into one group.
    chosen = {}
    for i in sorted(range(len(records)), key=lambda j: records[j]["sample_id"]):
        chosen.setdefault((records[i]["cohort"], records[i]["geometry_sha256"]), i)
    return np.array(sorted(chosen.values()), dtype=int)


def prepare_inputs(config, *, recompute_features=True):
    validate_config(config)
    records, matrices, schemas, provenance = [], [], [], {}
    max_error = 0.0
    for cohort in ("old", "new"):
        spec = config[cohort]
        record_key = "sources" if cohort == "old" else "manifest"
        source_path = checked_path(spec[record_key + "_path"], spec[record_key + "_sha256"], record_key)
        table_path = checked_path(spec["features_path"], spec["features_sha256"], "features")
        schema_path = checked_path(spec["feature_schema_path"], spec["feature_schema_sha256"], "schema")
        schema = read_json(schema_path)
        raw = read_json(source_path)
        rows = raw["samples"] if isinstance(raw, dict) else raw
        if len(rows) != spec["expected_samples"]:
            raise ValueError("Source sample count changed")
        x = feature_table(table_path, schema, [r["sample_id"] for r in rows])
        for i, row in enumerate(rows):
            if cohort == "new" and (not row.get("patient_group_id") or not row.get("group_id")):
                raise ValueError("New cohort requires verified patient/group keys")
            if cohort == "new" and row.get("dicom_relative_path"):
                dicom_path = source_path.parent / row["dicom_relative_path"]
            else:
                dicom_path = resolve_path(row["dicom_path"])
            if sha256(dicom_path) != row["dicom_sha256"]:
                raise ValueError("Source DICOM hash changed")
            geometry = delivery_geometry_hash(pydicom.dcmread(dicom_path))
            if cohort == "new" and geometry != row["geometry_sha256"]:
                raise ValueError("New source geometry differs from manifest")
            if recompute_features:
                actual = extract_features(dicom_path)
                values = np.array([actual[name] for name in schema["ordered_names"]])
                max_error = max(max_error, float(np.max(np.abs(values - x[i]))))
                if not np.allclose(values, x[i], rtol=1e-12, atol=1e-12):
                    raise ValueError("Recomputed source features differ from saved table")
            records.append({"sample_id": cohort + ":" + row["sample_id"], "cohort": cohort,
                            "dicom_sha256": row["dicom_sha256"], "geometry_sha256": geometry,
                            "patient_group_id": row.get("patient_group_id") if cohort == "new" else None,
                            "original_group_id": row.get("group_id") if cohort == "new" else None})
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"Verified {cohort}: {i + 1}/{len(rows)}", flush=True)
        matrices.append(x)
        schemas.append(schema)
        provenance[cohort] = {"source_records_sha256": sha256(source_path),
                              "features_sha256": sha256(table_path), "schema_sha256": sha256(schema_path),
                              "samples": len(rows), "source_dicoms_verified": len(rows)}
    if schemas[0] != schemas[1]:
        raise ValueError("Old/new feature definitions differ")
    x = np.vstack(matrices)
    y = np.array([int(r["cohort"] == "new") for r in records])
    groups = union_groups(records, x)
    for r, group in zip(records, groups):
        r["group_id"] = str(group)
    provenance["feature_recomputation"] = {
        "performed": recompute_features, "samples": len(records) if recompute_features else 0,
        "max_absolute_error": max_error if recompute_features else None, "rtol": 1e-12, "atol": 1e-12}
    return records, x, y, groups, schemas[0], provenance


def feature_statistics(x, y, names, units):
    if x.ndim != 2 or x.shape[1] != len(names) or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid statistics input")
    if set(np.unique(y)) != {0, 1} or min(np.sum(y == k) for k in (0, 1)) < 2:
        raise ValueError("Both domains need at least two samples")
    result = []
    for j, name in enumerate(names):
        old, new = x[y == 0, j], x[y == 1, j]
        scale = float(np.sqrt((np.var(old, ddof=1) + np.var(new, ddof=1)) / 2))
        row = {"feature": name, "unit": units[name]}
        for label, values in (("old", old), ("new", new)):
            q25, median, q75 = np.quantile(values, [.25, .5, .75])
            row.update({f"{label}_n": len(values), f"{label}_mean": float(np.mean(values)),
                        f"{label}_sd": float(np.std(values, ddof=1)), f"{label}_median": float(median),
                        f"{label}_q25": float(q25), f"{label}_q75": float(q75)})
        distance = float(wasserstein_distance(old, new))
        row.update({"mean_difference_new_minus_old": float(np.mean(new) - np.mean(old)),
                    "smd_new_minus_old": float((np.mean(new) - np.mean(old)) / scale) if scale else None,
                    "ks_statistic": float(ks_2samp(old, new).statistic),
                    "wasserstein": distance, "wasserstein_over_pooled_sd": distance / scale if scale else None,
                    "normalization_status": "ok" if scale else "zero_within_domain_variance"})
        result.append(row)
    return result


def split_assignment(x, y, groups, seed, folds=5):
    assignment = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Group leakage")
        if set(y[train]) != {0, 1} or set(y[test]) != {0, 1} or np.any(assignment[test] != -1):
            raise ValueError("Invalid domain split")
        assignment[test] = fold
    if np.any(assignment < 0):
        raise ValueError("OOF coverage incomplete")
    return assignment


def model_for(name, seed):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(**MODEL_PARAMETERS[name], random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(**MODEL_PARAMETERS[name], random_state=seed)
    raise ValueError("Unknown domain model")


def domain_metrics(y, probability, groups):
    if len(y) != len(probability) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid source probabilities")
    predicted = (probability > .5).astype(int)
    counts = Counter(groups)
    weight = np.array([1 / counts[g] for g in groups])
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "auc_new": float(roc_auc_score(y, probability)),
            "old_recall": float(recall_score(y, predicted, pos_label=0)),
            "new_recall": float(recall_score(y, predicted, pos_label=1)),
            "group_weighted_balanced_accuracy": float(balanced_accuracy_score(y, predicted, sample_weight=weight)),
            "group_weighted_auc_new": float(roc_auc_score(y, probability, sample_weight=weight))}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_domain_models(output, records, x, y, groups, *, seeds=(17, 42, 2026), folds=5, models=MODELS, views=VIEWS):
    """Internal array API also used for synthetic training-scope tests."""
    output = Path(output)
    metrics, tasks, view_info = [], [], {}
    for view in views:
        indices = view_indices(records, view)
        xv, yv, gv = x[indices], y[indices], groups[indices]
        view_info[view] = {"indices": indices.tolist(), "samples": len(indices), "groups": len(set(gv)),
                           "old_samples": int(np.sum(yv == 0)), "new_samples": int(np.sum(yv == 1)),
                           "always_old_accuracy": float(np.mean(yv == 0)), "always_old_balanced_accuracy": .5,
                           "always_old_auc": .5}
        for seed in seeds:
            assignment = split_assignment(xv, yv, gv, seed, folds)
            for model_name in models:
                probability = np.full(len(indices), np.nan)
                for fold in range(folds):
                    train_local, test_local = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                    train, test = indices[train_local], indices[test_local]
                    directory = output / "models" / view / model_name / f"seed_{seed}" / f"fold_{fold}"
                    directory.mkdir(parents=True, exist_ok=False)
                    fitted = model_for(model_name, seed)
                    started = time.monotonic()
                    fitted.fit(x[train], y[train])
                    seconds = time.monotonic() - started
                    pred = fitted.predict_proba(x[test])[:, 1]
                    model_path = directory / "estimator.joblib"
                    joblib.dump(fitted, model_path)
                    restored = joblib.load(model_path).predict_proba(x[test])[:, 1]
                    delta = float(np.max(np.abs(pred - restored)))
                    if delta > 1e-12:
                        raise ValueError("Checkpoint replay changed domain predictions")
                    if model_name == "logistic" and np.max(fitted[-1].n_iter_) >= MODEL_PARAMETERS[model_name]["max_iter"]:
                        raise ValueError("Domain logistic model reached its iteration limit")
                    probability[test_local] = pred
                    task = {"view": view, "model": model_name, "seed": seed, "fold": fold,
                            "train_indices": train.tolist(), "test_indices": test.tolist(),
                            "model_path": model_path.relative_to(output).as_posix(), "model_sha256": sha256(model_path),
                            "prediction_new": pred.tolist(), "fit_seconds": seconds,
                            "replay_max_absolute_error": delta, "final_estimator_saved": True,
                            "epoch_checkpoints": "not_applicable_sklearn_estimator"}
                    write_json(directory / "fit_scope.json", task)
                    tasks.append(task)
                    print(f"Domain fit {len(tasks)}: {view}/{model_name}/seed{seed}/fold{fold}", flush=True)
                metrics.append({"view": view, "model": model_name, "seed": seed,
                                **domain_metrics(yv, probability, gv)})
    return tasks, metrics, view_info


def check_artifact_hashes(output, hashes):
    root = Path(output).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError("Run artifact hash changed")


def source_identity():
    source_root = Path(__file__).resolve().parents[2]
    hashes = {p.relative_to(source_root.parent).as_posix(): sha256(p)
              for p in sorted(source_root.rglob("*.py"))}
    return source_root, hashes


def environment_identity():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn", "joblib", "pydicom")}}


def check_version_identity(protocol, snapshot):
    if source_identity()[1] != protocol["source_hashes"]:
        raise ValueError("Current source version differs from this run; use its frozen source")
    if environment_identity() != protocol["environment"]:
        raise ValueError("Current Python/platform/packages differ from the recorded environment")
    with zipfile.ZipFile(snapshot) as archive:
        for relative, expected in protocol["source_hashes"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError("Source snapshot file differs from protocol")
        config_bytes = archive.read("run_config.json")
        if (hashlib.sha256(config_bytes).hexdigest() != protocol["config_sha256"]
                or json.loads(config_bytes) != protocol["config"]):
            raise ValueError("Frozen config snapshot differs from protocol")


def check_fitted_scope(fitted, model_name, seed, x_train):
    if not np.array_equal(fitted.classes_, [0, 1]):
        raise ValueError("Saved estimator changed class order")
    estimator = fitted[-1] if model_name == "logistic" else fitted
    params = estimator.get_params()
    for key, expected in {**MODEL_PARAMETERS[model_name], "random_state": seed}.items():
        if params[key] != expected:
            raise ValueError("Saved estimator parameters differ from protocol")
    if model_name == "logistic":
        expected = StandardScaler().fit(x_train)
        for field in ("mean_", "var_", "scale_", "n_samples_seen_"):
            if not np.allclose(getattr(fitted[0], field), getattr(expected, field), rtol=1e-12, atol=1e-12):
                raise ValueError("Scaler was not fitted on the training partition")


def verify_run(output):
    output = Path(output)
    hashes = read_json(output / "files_sha256.json")
    check_artifact_hashes(output, hashes)
    protocol = read_json(output / "protocol.json")
    check_version_identity(protocol, output / "source_snapshot.zip")
    results = read_json(output / "results.json")
    if results.get("status") != "complete" or results.get("psqa_supervised_training") is not False:
        raise ValueError("Run not complete or target changed")
    # Input bytes and geometry are verified again; the expensive independent
    # feature reconstruction already occurred before training and is recorded.
    records, x, y, groups, schema, _ = prepare_inputs(protocol["config"], recompute_features=False)
    prepared = np.load(output / "prepared.npz", allow_pickle=False)
    if not (np.array_equal(x, prepared["x"]) and np.array_equal(y, prepared["y_domain"])
            and np.array_equal(groups, prepared["groups"]) and records == read_json(output / "cohort.json")):
        raise ValueError("Saved data differs from frozen inputs")
    statistics = []
    for view in protocol["config"]["views"]:
        selected = view_indices(records, view)
        statistics.extend({"view": view, **row} for row in feature_statistics(
            x[selected], y[selected], schema["ordered_names"], schema["units"]))
    if statistics != read_json(output / "feature_statistics.json"):
        raise ValueError("Descriptive statistics differ from source features")
    tasks = read_json(output / "tasks.json")
    if len(tasks) != results["expected_fits"]:
        raise ValueError("Incomplete task set")
    task_index = {(t["view"], t["model"], t["seed"], t["fold"]): t for t in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("Duplicate task identity")
    maximum = 0.0
    replayed = 0
    recomputed_metrics = []
    for view in protocol["config"]["views"]:
        indices = view_indices(records, view)
        for seed in protocol["config"]["evaluation"]["seeds"]:
            assignment = split_assignment(x[indices], y[indices], groups[indices], seed,
                                          protocol["config"]["evaluation"]["folds"])
            for name in protocol["config"]["models"]:
                probability = np.full(len(indices), np.nan)
                for fold in range(protocol["config"]["evaluation"]["folds"]):
                    task = task_index[(view, name, seed, fold)]
                    train, test = indices[assignment != fold], indices[assignment == fold]
                    if task["train_indices"] != train.tolist() or task["test_indices"] != test.tolist():
                        raise ValueError("Training scope or OOF assignment changed")
                    path = output / task["model_path"]
                    if sha256(path) != task["model_sha256"]:
                        raise ValueError("Saved estimator changed")
                    fitted = joblib.load(path)
                    check_fitted_scope(fitted, name, seed, x[train])
                    pred = fitted.predict_proba(x[test])[:, 1]
                    delta = float(np.max(np.abs(pred - np.asarray(task["prediction_new"]))))
                    maximum = max(maximum, delta)
                    if delta > 1e-12:
                        raise ValueError("Independent estimator replay failed")
                    probability[assignment == fold] = pred
                    replayed += 1
                recomputed_metrics.append({"view": view, "model": name, "seed": seed,
                                           **domain_metrics(y[indices], probability, groups[indices])})
    if recomputed_metrics != results["per_seed"]:
        raise ValueError("OOF metrics changed")
    audit = {"status": "passed", "source_dicoms_verified": len(records), "checkpoint_replays": replayed,
             "max_prediction_absolute_error": maximum, "group_isolation": "passed",
             "target": "source_membership_old0_new1", "psqa_accuracy_estimated": False,
             "files_manifest_sha256": sha256(output / "files_sha256.json")}
    write_json(output / "artifact_audit.json", audit)
    return audit


def run(config_path, output):
    config_path, output = resolve_path(config_path), resolve_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    validate_config(config)
    write_json(output / "status.json", {"status": "verifying_inputs"})
    source_root, source_hashes = source_identity()
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in source_hashes:
            archive.write(source_root.parent / relative, relative)
        archive.write(config_path, "run_config.json")
    protocol = {"schema": "cohort-shift-run-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "config_sha256": sha256(config_path), "source_hashes": source_hashes,
                "target": "source_membership_old0_new1", "psqa_supervised_training": False,
                "limitation": LIMITATION, "resolved_model_parameters": MODEL_PARAMETERS,
                "probability_rule": "new if P(new)>0.5, otherwise old",
                "split_and_model_seed_shared": True,
                "one_per_geometry_rule": "lexicographically first prefixed sample ID per geometry within cohort",
                "environment": environment_identity()}
    write_json(output / "protocol.json", protocol)
    try:
        records, x, y, groups, schema, provenance = prepare_inputs(config)
        write_json(output / "cohort.json", records)
        write_json(output / "feature_schema.json", schema)
        write_json(output / "input_verification.json", provenance)
        np.savez_compressed(output / "prepared.npz", x=x, y_domain=y, groups=groups)
        stats = []
        for view in VIEWS:
            indices = view_indices(records, view)
            stats.extend({"view": view, **row} for row in feature_statistics(
                x[indices], y[indices], schema["ordered_names"], schema["units"]))
        write_csv(output / "feature_statistics.csv", stats)
        write_json(output / "feature_statistics.json", stats)
        write_json(output / "status.json", {"status": "fitting_domain_models", "expected_fits": 60})
        tasks, per_seed, views = fit_domain_models(output, records, x, y, groups)
        expected = len(VIEWS) * len(MODELS) * len(config["evaluation"]["seeds"]) * config["evaluation"]["folds"]
        if len(tasks) != expected:
            raise ValueError("Incomplete domain training")
        summary = []
        for view in VIEWS:
            for model in MODELS:
                subset = [m for m in per_seed if m["view"] == view and m["model"] == model]
                row = {"view": view, "model": model}
                for metric in subset[0]:
                    if metric in {"view", "model", "seed"}:
                        continue
                    values = [m[metric] for m in subset]
                    row.update({metric + "_mean": float(np.mean(values)), metric + "_min": min(values), metric + "_max": max(values)})
                summary.append(row)
        write_json(output / "tasks.json", tasks)
        write_csv(output / "domain_metrics_per_seed.csv", per_seed)
        write_csv(output / "domain_metrics_summary.csv", summary)
        old_geometry = {r["geometry_sha256"] for r in records if r["cohort"] == "old"}
        new_geometry = {r["geometry_sha256"] for r in records if r["cohort"] == "new"}
        results = {"status": "complete", "expected_fits": expected, "successful_fits": len(tasks),
                   "target": "source_membership_old0_new1", "psqa_supervised_training": False,
                   "limitation": LIMITATION, "views": views, "per_seed": per_seed, "summary": summary,
                   "source_counts": {"old": int(np.sum(y == 0)), "new": int(np.sum(y == 1))},
                   "unique_geometries": {"old": len(old_geometry), "new": len(new_geometry)},
                   "shared_geometries": len(old_geometry & new_geometry),
                   "feature_version": schema["version"], "feature_count": len(schema["ordered_names"]),
                   "descriptive_statistics_only": "KS statistic and standardized distances; no independent-sample p-values"}
        write_json(output / "results.json", results)
        write_json(output / "status.json", {"status": "complete_pending_independent_audit", "successful_fits": len(tasks)})
        write_json(output / "files_sha256.json", {p.relative_to(output).as_posix(): sha256(p)
                   for p in sorted(output.rglob("*")) if p.is_file() and p.name != "files_sha256.json"})
        return results
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        if args.config or args.output:
            parser.error("--verify cannot be combined with --config/--output")
        print(verify_run(resolve_path(args.verify)))
    elif args.config and args.output:
        result = run(args.config, args.output)
        print({"status": result["status"], "successful_fits": result["successful_fits"], "psqa_supervised_training": False})
    else:
        parser.error("Provide --config and --output, or --verify RUN")


if __name__ == "__main__":
    main()
