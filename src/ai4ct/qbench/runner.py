"""Fixed-configuration, frozen-fold benchmark shards; no outer-test selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
from sklearn.ensemble import IsolationForest

from .data import digest_json, load_dataset, read_json, sha256, write_json
from .evaluation import compare_predictions, metrics, prediction_arrays


FAMILIES = {"cv": ("cv", "fit_cv", "load_cv"),
            "reupload": ("reupload_reference", "fit_reupload", "load_reupload"),
            "inductive": ("inductive", "fit_inductive", "load_inductive"),
            "classical": ("classical", "fit_classical", "load_classical"),
            "kernel": ("kernels", "fit_kernel", "load_kernel"),
            "variational": ("variational", "fit_variational", "load_variational")}


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "pennylane", "pennylane-lightning", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "worker_threads": 1, "worker_environment": {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def model_functions(family):
    module, fit_name, load_name = FAMILIES[family]
    imported = importlib.import_module(f"ai4ct.qbench.{module}")
    return getattr(imported, fit_name), getattr(imported, load_name)


def source_files():
    package = Path(__file__).resolve().parent.parent
    return {"src/ai4ct/" + p.relative_to(package).as_posix(): p
            for p in sorted(package.rglob("*.py"))}


def source_hashes():
    return {name: sha256(path) for name, path in source_files().items()}


def assert_sources(expected):
    if source_hashes() != expected:
        raise ValueError("Current source differs from the frozen run snapshot")


def snapshot(output, config, hashes):
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in source_files().items():
            if sha256(path) != hashes[name]:
                raise ValueError("Source changed while snapshotting")
            archive.write(path, name)
        archive.writestr("config.json", json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2))


def task_key(model_id, threshold, fold, seed):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", model_id):
        raise ValueError("Unsafe model ID")
    return f"{model_id}__gpr{threshold:g}__fold{fold}__seed{seed}"


def expected_tasks(config, models=None, seeds=None, folds=None):
    evaluation = config["evaluation"]
    models = list(config["active_model_ids"] if models is None else models)
    seeds = list(evaluation["training_seeds"] if seeds is None else seeds)
    folds = list(range(evaluation["outer_folds"]) if folds is None else folds)
    if not models or not seeds or not folds or len(set(models)) != len(models) or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("Task filters must be nonempty and unique")
    if not set(models) <= set(config["active_model_ids"]) or not set(seeds) <= set(evaluation["training_seeds"]) or not set(folds) <= set(range(evaluation["outer_folds"])):
        raise ValueError("Requested task is not registered in the frozen configuration")
    tasks = []
    for model in models:
        spec = config["model_configs"][model]
        if spec["family"] not in FAMILIES or spec["track"] not in {"paper_compat", "standard_v1"}:
            raise ValueError("Invalid model family/track")
        for threshold in evaluation["gpr_thresholds"]:
            for fold in folds:
                for seed in seeds:
                    tasks.append({"task_id": task_key(model, float(threshold), fold, seed),
                                  "model_id": model, "threshold": float(threshold), "fold": int(fold),
                                  "seed": int(seed), "family": spec["family"], "track": spec["track"],
                                  "config": spec["config"]})
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task keys")
    return tasks


def fit_indices(dataset, task, config):
    indices = np.flatnonzero(dataset.fold != task["fold"])
    if task["track"] == "paper_compat":
        splitter_seed = int(config["evaluation"]["split_seed"]) + task["fold"]
        keep = IsolationForest(n_estimators=100, contamination=0.1, random_state=splitter_seed,
                               n_jobs=1).fit_predict(dataset.x[indices]) == 1
        indices = indices[keep]
    if len(set((dataset.rates[indices] >= task["threshold"]).astype(int))) != 2:
        raise ValueError("Training processing removed a class")
    return indices


def predict(fitted, x):
    score = fitted.predict_score(x)
    method = getattr(fitted, "predict_proba", None)
    probability = None if method is None else method(x)
    return prediction_arrays(score, probability, len(x))[:2]


def model_file_hashes(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("No persisted model artifacts")
    return files


def execute_task(request_path):
    """Run inside one isolated process; outer labels are used only after fitting."""
    request = read_json(request_path)
    output = Path(request["output_dir"])
    task = request["task"]
    assert_sources(request["source_hashes"])
    prepared = Path(request["prepared_path"])
    if sha256(prepared) != request["prepared_sha256"]:
        raise ValueError("Prepared feature matrix changed")
    with np.load(prepared, allow_pickle=False) as data:
        x = data["x"]
        rates = data["rates"]
    train = np.asarray(request["fit_indices"], dtype=int)
    test = np.asarray(request["test_indices"], dtype=int)
    y_train = (rates[train] >= task["threshold"]).astype(int)
    fit, loader = model_functions(task["family"])
    start = time.perf_counter()
    fitted = fit(task["model_id"], x[train], y_train, seed=task["seed"],
                 output_dir=output / "model", config=task["config"])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    score, probability = predict(fitted, x[test])
    predict_seconds = time.perf_counter() - start
    reloaded = loader(output / "model")
    replay_score, replay_probability = predict(reloaded, x[test])
    replay = compare_predictions(score, probability, replay_score, replay_probability)
    metadata = read_json(output / "model" / "metadata.json")
    rows = []
    for i, source in enumerate(request["test_samples"]):
        rows.append({**source, "fold": task["fold"], "seed": task["seed"], "threshold": task["threshold"],
                     "label_pass": int(source["gpr"] >= task["threshold"]), "pass_score": float(score[i]),
                     "score_direction": "higher_is_pass", "score_kind": "raw_model_score",
                     "probability_kind": None if probability is None else metadata.get("probability_kind", "model_probability_not_explicitly_calibrated"),
                     "pass_probability": None if probability is None else float(probability[i])})
    write_json(output / "predictions.json", rows)
    write_json(output / "result.json", {"status": "succeeded", "task_id": task["task_id"],
               "task_digest": request["task_digest"], "metrics": metrics((rates[test] >= task["threshold"]).astype(int), score, probability),
               "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "checkpoint_replay": replay,
               "prediction_numerics": getattr(fitted, "last_prediction_diagnostics", None),
               "replay_prediction_numerics": getattr(reloaded, "last_prediction_diagnostics", None),
               "model_file_sha256": model_file_hashes(output / "model"),
               "predictions_sha256": sha256(output / "predictions.json"), "completed_at": now()})


def supervise(request_path, timeout_seconds):
    """Kill a timed-out fit without losing its already written epoch artifacts."""
    output = Path(read_json(request_path)["output_dir"])
    started = time.perf_counter()
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "ai4ct.qbench.runner", "--execute-task", str(request_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=timeout_seconds)
            if code != 0:
                failure_path = output / "worker_error.json"
                failure = read_json(failure_path) if failure_path.exists() else {"status": "failed", "error_type": "WorkerProcessError"}
                return {**failure, "return_code": code, "wall_seconds": time.perf_counter() - started}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {"status": "timed_out", "error_type": "TrialWallClockLimit", "wall_seconds": time.perf_counter() - started}
        except BaseException:
            process.kill()
            process.wait()
            raise
    result = read_json(output / "result.json")
    return {**result, "wall_seconds": time.perf_counter() - started}


def run_shard(config, output, *, models=None, seeds=None, folds=None, resume_completed=False, executor=supervise):
    if config.get("executable") is not True:
        raise ValueError("Only an explicitly executable configuration may run")
    dataset = load_dataset(config)
    tasks = expected_tasks(config, models, seeds, folds)
    output = Path(output).resolve()
    sources = source_hashes()
    identity = {"schema_version": "qbench-run-v1", "config": config, "config_sha256": digest_json(config),
                "provenance": dataset.provenance, "source_hashes": sources,
                "environment": environment(),
                "selected_tasks": tasks, "expected_full_task_ids": [t["task_id"] for t in expected_tasks(config)],
                "analysis": "fixed_configuration_exploratory_oof; no outer-test selection"}
    if output.exists() and any(output.iterdir()):
        if not resume_completed:
            raise FileExistsError("Output is nonempty; choose another run or --resume-completed")
        prior = read_json(output / "protocol.json")
        if {key: prior[key] for key in identity} != identity:
            raise ValueError("Resume configuration/data/source/task hashes differ")
        if sha256(output / "prepared.npz") != prior["prepared_sha256"] or sha256(output / "cohort.json") != prior["cohort_sha256"]:
            raise ValueError("Prepared run artifacts changed")
        if sha256(output / "source_snapshot.zip") != prior["source_snapshot_sha256"]:
            raise ValueError("Source snapshot changed")
        with np.load(output / "prepared.npz", allow_pickle=False) as prepared:
            if not (np.array_equal(prepared["x"], dataset.x) and np.array_equal(prepared["rates"], dataset.rates)
                    and np.array_equal(prepared["fold"], dataset.fold)):
                raise ValueError("Prepared values differ from current verified sources")
        protocol = prior
    else:
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / "prepared.npz", x=dataset.x, rates=dataset.rates, fold=dataset.fold)
        write_json(output / "cohort.json", dataset.samples)
        write_json(output / "feature_schema.json", dataset.schema)
        snapshot(output, config, sources)
        protocol = {**identity, "created_at": now(), "prepared_sha256": sha256(output / "prepared.npz"),
                    "cohort_sha256": sha256(output / "cohort.json"),
                    "source_snapshot_sha256": sha256(output / "source_snapshot.zip")}
        write_json(output / "protocol.json", protocol)
    budget = config.get("execution", config.get("resources", {}))
    trial_limit = float(budget.get("max_wall_seconds_per_trial", config.get("max_wall_seconds_per_trial", 1800)))
    total_limit = float(budget.get("max_total_device_seconds", config.get("max_total_device_seconds", 43200)))
    if not np.isfinite([trial_limit, total_limit]).all() or min(trial_limit, total_limit) <= 0:
        raise ValueError("Positive finite execution budgets required")
    ledger = []
    prior_consumed = sum(float(read_json(p).get("wall_seconds", 0.0))
                         for p in output.glob("tasks/*/attempts/*/status.json"))
    consumed = prior_consumed
    for task in tasks:
        directory = output / "tasks" / task["task_id"]
        directory.mkdir(parents=True, exist_ok=True)
        core_digest = digest_json({"task": task, "config_sha256": protocol["config_sha256"],
                                   "provenance": dataset.provenance, "source_hashes": sources})
        previous = read_json(directory / "status.json") if (directory / "status.json").exists() else None
        if previous and previous["status"] == "succeeded":
            from .audit import audit_task
            audit_task(output, task, dataset, replay=True)
            ledger.append(previous)
            write_json(output / "ledger.json", ledger)
            continue
        attempts = directory / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04d}"
        attempt.mkdir()
        state = {"task_id": task["task_id"], "task_digest": core_digest, "attempt": attempt.relative_to(output).as_posix(),
                 "started_at": now(), "status": "running"}
        write_json(directory / "task.json", task)
        write_json(directory / "status.json", state)
        write_json(attempt / "status.json", state)
        try:
            if consumed >= total_limit:
                result = {"status": "cancelled", "error_type": "ShardResourceBudgetExhausted", "wall_seconds": 0.0}
            else:
                train = fit_indices(dataset, task, config)
                test = np.flatnonzero(dataset.fold == task["fold"])
                fit_scope = {"train_before": int((dataset.fold != task["fold"]).sum()), "train_after": len(train),
                             "train_sample_ids": [dataset.samples[i]["sample_id"] for i in train],
                             "test_sample_ids": [dataset.samples[i]["sample_id"] for i in test],
                             "track": task["track"], "outlier_filter": "IF(train,.1),seed=split_seed+fold" if task["track"] == "paper_compat" else "none"}
                write_json(attempt / "fit_scope.json", fit_scope)
                request = {"task": task, "task_digest": core_digest, "source_hashes": sources,
                           "prepared_path": str(output / "prepared.npz"), "prepared_sha256": protocol["prepared_sha256"],
                           "fit_indices": train.tolist(), "test_indices": test.tolist(),
                           "test_samples": [dataset.samples[i] for i in test], "output_dir": str(attempt)}
                write_json(attempt / "request.json", request)
                result = executor(attempt / "request.json", min(trial_limit, total_limit-consumed))
        except KeyboardInterrupt:
            result = {"status": "interrupted", "error_type": "KeyboardInterrupt", "wall_seconds": 0.0}
            write_json(attempt / "status.json", {**state, **result, "ended_at": now()})
            write_json(directory / "status.json", {**state, **result, "ended_at": now()})
            raise
        except Exception as exc:
            (attempt / "exception.log").write_text(traceback.format_exc(), encoding="utf-8")
            result = {"status": "failed", "error_type": type(exc).__name__, "wall_seconds": 0.0}
        consumed += result.get("wall_seconds", 0.0)
        state = {**state, **result, "ended_at": now()}
        write_json(attempt / "status.json", state)
        write_json(directory / "status.json", state)
        ledger.append(state)
        write_json(output / "ledger.json", ledger)
        write_json(output / "status.json", {"status": "running", "tasks_processed": len(ledger), "selected_tasks": len(tasks)})
    result = {"status": "complete" if all(x["status"] == "succeeded" for x in ledger) else "incomplete",
              "selected_tasks": len(tasks), "succeeded": sum(x["status"] == "succeeded" for x in ledger),
              "wall_seconds_this_invocation": consumed-prior_consumed, "wall_seconds_all_attempts": consumed,
              "full_catalog_coverage": {t["task_id"] for t in tasks} == set(protocol["expected_full_task_ids"])}
    write_json(output / "status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models")
    parser.add_argument("--seeds")
    parser.add_argument("--folds")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--execute-task", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute_task:
        try:
            execute_task(args.execute_task)
            return 0
        except Exception as exc:
            request = read_json(args.execute_task)
            write_json(Path(request["output_dir"]) / "worker_error.json",
                       {"status": "timed_out" if isinstance(exc, TimeoutError) else "failed",
                        "error_type": type(exc).__name__, "error_message": str(exc)})
            traceback.print_exc()
            return 1
    if args.config is None or args.output is None:
        parser.error("--config and --output are required")
    result = run_shard(read_json(args.config), args.output,
                       models=None if args.models is None else args.models.split(","),
                       seeds=None if args.seeds is None else list(map(int, args.seeds.split(","))),
                       folds=None if args.folds is None else list(map(int, args.folds.split(","))),
                       resume_completed=args.resume_completed)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
