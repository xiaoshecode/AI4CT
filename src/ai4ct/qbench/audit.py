"""Independent ID-level artifact checks and coverage-aware shard aggregation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from .data import digest_json, load_dataset, read_json, sha256, write_json
from .evaluation import compare_predictions, metrics, prediction_arrays


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_metrics(left, right):
    require(set(left) == set(right), "Metric fields differ")
    for key, value in left.items():
        actual = right[key]
        if isinstance(value, float):
            require(isinstance(actual, (int, float)) and np.isclose(value, actual, rtol=1e-12, atol=1e-12), f"Metric differs: {key}")
        else:
            require(value == actual, f"Metric differs: {key}")


def attempt_directory(output, status):
    output = Path(output).resolve()
    directory = (output / status["attempt"]).resolve()
    require(directory.is_relative_to(output / "tasks"), "Attempt path escapes run tasks")
    return directory


def audit_task(output, task, dataset, *, replay=True):
    from .runner import fit_indices, model_file_hashes, model_functions, predict
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    task_dir = output / "tasks" / task["task_id"]
    require(read_json(task_dir / "task.json") == task, "Task configuration changed")
    state = read_json(task_dir / "status.json")
    require(state["status"] == "succeeded", "Task has not succeeded")
    expected_digest = digest_json({"task": task, "config_sha256": protocol["config_sha256"],
                                  "provenance": dataset.provenance, "source_hashes": protocol["source_hashes"]})
    require(state["task_digest"] == expected_digest and state["task_id"] == task["task_id"], "Task digest changed")
    directory = attempt_directory(output, state)
    result = read_json(directory / "result.json")
    require(all(state.get(key) == value for key, value in result.items()), "Status/result disagree")
    require(result["task_digest"] == expected_digest and result["status"] == "succeeded", "Invalid result identity")
    require(model_file_hashes(directory / "model") == result["model_file_sha256"], "Model artifact hashes differ")
    require(sha256(directory / "predictions.json") == result["predictions_sha256"], "Prediction artifact hash differs")
    train = fit_indices(dataset, task, protocol["config"])
    test = np.flatnonzero(dataset.fold == task["fold"])
    scope = read_json(directory / "fit_scope.json")
    require(scope["train_sample_ids"] == [dataset.samples[i]["sample_id"] for i in train], "Training IDs differ from frozen processing")
    require(scope["test_sample_ids"] == [dataset.samples[i]["sample_id"] for i in test], "Test IDs differ from frozen fold")
    require(scope["train_before"] == int((dataset.fold != task["fold"]).sum()) and scope["train_after"] == len(train), "Training scope counts differ")
    require(not set(dataset.groups[train]) & set(dataset.groups[test]), "Training/test group overlap")
    metadata = read_json(directory / "model" / "metadata.json")
    require(metadata["model_id"] == task["model_id"] and metadata["seed"] == task["seed"], "Model metadata identity differs")
    require(metadata.get("n_train", metadata.get("training_rows")) == len(train), "Model training count differs")
    saved_epochs = metadata.get("checkpoint_epoch_snapshots") == "saved" or task["family"] == "variational"
    if saved_epochs:
        epochs = int(metadata["config"].get("epochs", metadata["config"].get("alignment_epochs", 30)))
        checkpoint_dir = directory / "model" / "checkpoints"
        require((checkpoint_dir / "initial.pt").is_file() and (checkpoint_dir / "final.pt").is_file(), "Initial/final checkpoint missing")
        require(all((checkpoint_dir / f"epoch_{epoch:04d}.pt").is_file() for epoch in range(1, epochs+1)), "Completed epoch checkpoint missing")
        require((checkpoint_dir / "best.pt").is_file() or (checkpoint_dir / "aliases.json").is_file(), "Best checkpoint/reference missing")
    else:
        require(metadata.get("checkpoint_epoch_snapshots") == "not_applicable", "Epoch checkpoint applicability is unspecified")
    rows = read_json(directory / "predictions.json")
    require(len(rows) == len(test), "Prediction count differs")
    for row, position in zip(rows, test):
        sample = dataset.samples[position]
        require(all(row.get(k) == v for k, v in sample.items()), "Prediction ID/GPR/group/source differs from manifest")
        require(row["threshold"] == task["threshold"] and row["fold"] == task["fold"] and row["seed"] == task["seed"], "Prediction task differs")
        require(row["label_pass"] == int(sample["gpr"] >= task["threshold"]), "Prediction label differs")
    scores = np.asarray([r["pass_score"] for r in rows], dtype=float)
    available = [r["pass_probability"] is not None for r in rows]
    require(all(available) or not any(available), "Mixed probability availability")
    probability = np.asarray([r["pass_probability"] for r in rows], dtype=float) if all(available) else None
    labels = (dataset.rates[test] >= task["threshold"]).astype(int)
    same_metrics(metrics(labels, scores, probability), result["metrics"])
    replay_report = None
    if replay:
        _, loader = model_functions(task["family"])
        fitted = loader(directory / "model")
        score2, p2 = predict(fitted, dataset.x[test])
        replay_report = compare_predictions(scores, probability, score2, p2)
    return {"task_id": task["task_id"], "status": "passed", "n_test": len(test),
            "replay": replay_report, "task": task, "rows": rows, "result": result}


def audit_run(output, *, replay=True, save=True):
    from .runner import assert_sources, expected_tasks
    output = Path(output).resolve()
    try:
        protocol = read_json(output / "protocol.json")
        require(digest_json(protocol["config"]) == protocol["config_sha256"], "Run config hash differs")
        require(protocol["expected_full_task_ids"] == [t["task_id"] for t in expected_tasks(protocol["config"])], "Full task registry differs")
        require(len({t["task_id"] for t in protocol["selected_tasks"]}) == len(protocol["selected_tasks"]), "Repeated selected task")
        full = {t["task_id"]: t for t in expected_tasks(protocol["config"])}
        require(all(full.get(t["task_id"]) == t for t in protocol["selected_tasks"]), "Selected task differs from full registry")
        dataset = load_dataset(protocol["config"])
        require(dataset.provenance == protocol["provenance"], "Source provenance differs")
        require(sha256(output / "prepared.npz") == protocol["prepared_sha256"], "Prepared feature artifact changed")
        require(sha256(output / "cohort.json") == protocol["cohort_sha256"], "Prepared cohort artifact changed")
        require(read_json(output / "cohort.json") == dataset.samples, "Prepared sample IDs/GPR/groups differ")
        require(read_json(output / "feature_schema.json") == dataset.schema, "Prepared schema differs")
        with np.load(output / "prepared.npz", allow_pickle=False) as prepared:
            require(np.array_equal(prepared["x"], dataset.x) and np.array_equal(prepared["rates"], dataset.rates)
                    and np.array_equal(prepared["fold"], dataset.fold), "Prepared feature/label/fold order differs")
        snapshot = output / "source_snapshot.zip"
        require(sha256(snapshot) == protocol["source_snapshot_sha256"], "Source archive hash differs")
        import hashlib
        with zipfile.ZipFile(snapshot) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)) and set(names) == set(protocol["source_hashes"]) | {"config.json"}, "Source archive members differ")
            for name, expected in protocol["source_hashes"].items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == expected, "Frozen source member hash differs")
            require(json.loads(archive.read("config.json")) == protocol["config"], "Archived config differs")
        if replay:
            assert_sources(protocol["source_hashes"])
        tasks = []
        states = []
        for task in protocol["selected_tasks"]:
            path = output / "tasks" / task["task_id"] / "status.json"
            state = read_json(path) if path.exists() else {"task_id": task["task_id"], "status": "planned"}
            states.append({"task_id": task["task_id"], "status": state["status"], "error_type": state.get("error_type")})
            if state["status"] == "succeeded":
                tasks.append(audit_task(output, task, dataset, replay=replay))
        report = {"status": "passed", "schema": "qbench-audit-v1", "audit_source_sha256": sha256(__file__),
                  "n_samples": len(dataset.samples), "n_groups": len(set(dataset.groups)),
                  "n_selected_tasks": len(states), "n_successful_tasks_audited": len(tasks),
                  "checkpoint_replays": len(tasks) if replay else 0,
                  "all_selected_tasks_succeeded": len(tasks) == len(states), "task_states": states,
                  "source_dicoms_verified": dataset.provenance["source_dicoms_verified"],
                  "checks": ["source and snapshot hashes", "manifest-feature-split IDs and order", "training scopes and groups",
                             "all successful predictions and metrics", "checkpoint replay" if replay else "checkpoint hashes only"]}
        if save:
            write_json(output / "artifact_audit.json", report)
        return report, tasks, dataset, protocol
    except Exception as exc:
        if save:
            write_json(output / "artifact_audit.json", {"status": "failed", "error_type": type(exc).__name__,
                                                       "message": str(exc), "audit_source_sha256": sha256(__file__)})
        raise


def bootstrap_accuracy(y, score, probability, groups, seed=42, repetitions=1000):
    _, _, predicted = prediction_arrays(score, probability, len(y))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    sizes = np.asarray([(groups == g).sum() for g in unique])
    correct = np.asarray([((groups == g) & (np.asarray(y) == predicted)).sum() for g in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(repetitions, len(unique)))
    values = correct[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    low, high = np.quantile(values, [.025, .975])
    return {"low": float(low), "high": float(high), "repetitions": repetitions,
            "method": "fixed OOF predictions, whole-group percentile bootstrap; no refitting or selection uncertainty"}


def aggregate(shards, output, *, replay=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Aggregation output must be new/empty")
    verified = [audit_run(path, replay=replay) for path in shards]
    require(bool(verified), "No shards supplied")
    _, _, dataset, reference = verified[0]
    tasks = {}
    all_states = []
    duplicate_tasks = []
    for path, (report, records, _, protocol) in zip(shards, verified):
        for key in ("config_sha256", "provenance", "source_hashes", "environment", "expected_full_task_ids"):
            require(protocol[key] == reference[key], f"Shards use different {key}")
        all_states.extend({**s, "shard": str(Path(path).resolve())} for s in report["task_states"])
        for record in records:
            key = record["task_id"]
            if key in tasks:
                prior = tasks[key]
                require(prior["task"] == record["task"] and prior["rows"] == record["rows"]
                        and prior["result"]["model_file_sha256"] == record["result"]["model_file_sha256"], "Conflicting duplicate task across shards")
                duplicate_tasks.append(key)
            else:
                tasks[key] = record
    expected = set(reference["expected_full_task_ids"])
    require(set(tasks) <= expected, "Unregistered completed task")
    oof = {}
    per_seed = {}
    config = reference["config"]
    for model in config["active_model_ids"]:
        for threshold in config["evaluation"]["gpr_thresholds"]:
            for seed in config["evaluation"]["training_seeds"]:
                records = [r for r in tasks.values() if r["task"]["model_id"] == model and r["task"]["threshold"] == threshold and r["task"]["seed"] == seed]
                if {r["task"]["fold"] for r in records} != set(range(config["evaluation"]["outer_folds"])):
                    continue
                rows = [row for r in records for row in r["rows"]]
                row_index = {row["sample_id"]: row for row in rows}
                ids = [s["sample_id"] for s in dataset.samples]
                require(len(row_index) == len(rows) == len(ids) and set(row_index) == set(ids), "OOF does not uniquely cover cohort")
                rows = [row_index[s] for s in ids]
                score = np.asarray([row["pass_score"] for row in rows])
                availability = [row["pass_probability"] is not None for row in rows]
                require(all(availability) or not any(availability), "Probability capability differs across folds")
                p = np.asarray([row["pass_probability"] for row in rows]) if all(availability) else None
                y = (dataset.rates >= threshold).astype(int)
                key = f"{model}__gpr{threshold:g}__seed{seed}"
                summary = metrics(y, score, p)
                summary["accuracy_ci95"] = bootstrap_accuracy(y, score, p, dataset.groups)
                summary["n_groups"] = len(set(dataset.groups))
                per_seed[key] = summary
                oof[key] = rows
    coverage_complete = set(tasks) == expected
    seed_summary = {}
    for model in config["active_model_ids"]:
        for threshold in config["evaluation"]["gpr_thresholds"]:
            keys = [f"{model}__gpr{threshold:g}__seed{seed}" for seed in config["evaluation"]["training_seeds"]]
            if not all(key in per_seed for key in keys):
                continue
            summary = {"training_seeds": config["evaluation"]["training_seeds"], "aggregation": "mean and range of each seed's complete OOF metric; no best-seed selection"}
            for metric in ("accuracy", "balanced_accuracy", "fail_recall", "pass_recall", "auc_pass", "average_precision_fail", "brier", "log_loss"):
                values = [per_seed[key][metric] for key in keys]
                summary[metric] = None if any(value is None for value in values) else {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
            seed_summary[f"{model}__gpr{threshold:g}"] = summary
    report = {"schema": "qbench-aggregate-v1", "status": "complete" if coverage_complete else "incomplete",
              "interpretation": "exploratory existing-cohort fixed-configuration OOF; not external confirmation",
              "full_catalog_coverage": coverage_complete,
              "ranking_status": "eligible_for_descriptive_comparison_only" if coverage_complete else "withheld_incomplete_coverage",
              "expected_tasks": len(expected), "successful_unique_tasks": len(tasks),
              "missing_task_ids": sorted(expected - set(tasks)), "duplicate_identical_tasks": duplicate_tasks,
              "task_states": all_states, "per_model_threshold_seed_oof": per_seed, "seed_summary": seed_summary,
              "config_sha256": reference["config_sha256"], "provenance": reference["provenance"],
              "source_hashes": reference["source_hashes"], "environment": reference["environment"], "checkpoint_replay_requested": replay}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", report)
    write_json(output / "oof_predictions.json", oof)
    write_json(output / "protocol.json", {"config": config, "source_shards": [str(Path(s).resolve()) for s in shards],
                                           "source_protocol_sha256": [sha256(Path(s) / "protocol.json") for s in shards]})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--aggregate", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.aggregate:
        if not args.output:
            parser.error("--output required with --aggregate")
        report = aggregate(args.aggregate, args.output)
        print(json.dumps({k: report[k] for k in ("status", "expected_tasks", "successful_unique_tasks", "ranking_status")}))
    elif args.run:
        report, _, _, _ = audit_run(args.run)
        print(json.dumps({k: report[k] for k in ("status", "n_selected_tasks", "n_successful_tasks_audited", "checkpoint_replays")}))
    else:
        parser.error("Provide a run directory or --aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
