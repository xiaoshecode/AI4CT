"""Build public aggregate tables and figures from independently audited batches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import read_json, sha256, write_json


METRICS = ("accuracy", "balanced_accuracy", "fail_recall", "pass_recall",
           "auc_pass", "average_precision_fail", "brier", "log_loss")
CLASSICAL = {"always_pass", "logistic", "linear_svm", "rbf_svm", "gbdt",
             "random_forest", "small_mlp", "rff_linear", "dressed_classical_5d"}


def category(model):
    if model == "qboost_qubo_12_classical":
        return "classical_qubo_control"
    return "classical" if model in CLASSICAL else "quantum_related_simulation"


def load_batches(paths):
    """Require complete separate audits; never merge heterogeneous fit manifests."""
    batches, seen, shared = [], set(), None
    for path in map(Path, paths):
        results, protocol = read_json(path / "results.json"), read_json(path / "protocol.json")
        if not (results["status"] == "complete" and results["full_catalog_coverage"]
                and results["checkpoint_replay_requested"]
                and results["expected_tasks"] == results["successful_unique_tasks"]):
            raise ValueError(f"Batch is incomplete or lacks checkpoint replay: {path}")
        config = protocol["config"]
        evaluation = config["evaluation"]
        identity = {"provenance": results["provenance"], "environment": results["environment"],
                    "thresholds": evaluation["gpr_thresholds"], "seeds": evaluation["training_seeds"],
                    "folds": evaluation["outer_folds"], "split": evaluation["outer_split_sha256"],
                    "positive_class": evaluation["positive_class"]}
        if shared is None:
            shared = identity
        elif shared != identity:
            raise ValueError("Batches differ in data, split, label, seed, or environment contract")
        models = set(config["active_model_ids"])
        if seen & models:
            raise ValueError(f"Repeated model IDs across batches: {sorted(seen & models)}")
        seen |= models
        shards = protocol["source_shards"]
        if len(shards) != len(protocol["source_protocol_sha256"]):
            raise ValueError("Shard provenance count differs")
        audits = []
        for shard, expected in zip(shards, protocol["source_protocol_sha256"]):
            directory = Path(shard)
            if sha256(directory / "protocol.json") != expected:
                raise ValueError("Audited shard protocol changed")
            audit = read_json(directory / "artifact_audit.json")
            if not (audit["status"] == "passed" and audit["all_selected_tasks_succeeded"]
                    and audit["checkpoint_replays"] == audit["n_selected_tasks"]):
                raise ValueError("Shard does not have a complete independent model replay")
            audits.append({"path": str(directory), "audit_sha256": sha256(directory / "artifact_audit.json"),
                           "checkpoint_replays": audit["checkpoint_replays"]})
        batches.append({"path": path, "results": results, "config": config, "audits": audits})
    if not batches:
        raise ValueError("At least one audited batch is required")
    return batches, shared


def summary_rows(batches):
    rows, per_seed = [], []
    for batch in batches:
        cfg, results = batch["config"], batch["results"]
        for model in cfg["active_model_ids"]:
            spec = cfg["model_configs"][model]
            for threshold in cfg["evaluation"]["gpr_thresholds"]:
                key = f"{model}__gpr{threshold:g}"
                stats = results["seed_summary"][key]
                row = {"model_id": model, "category": category(model), "track": spec["track"],
                       "family": spec["family"], "gpr_threshold": threshold,
                       "protocol_id": cfg["protocol_id"], "training_seeds": ",".join(map(str, stats["training_seeds"]))}
                for metric in METRICS:
                    for field in ("mean", "min", "max"):
                        row[f"{metric}_{field}"] = None if stats[metric] is None else stats[metric][field]
                rows.append(row)
                for seed in stats["training_seeds"]:
                    value = results["per_model_threshold_seed_oof"][f"{key}__seed{seed}"]
                    item = {"model_id": model, "gpr_threshold": threshold, "seed": seed,
                            "protocol_id": cfg["protocol_id"], **{m: value[m] for m in METRICS}}
                    item.update(accuracy_ci95_low=value["accuracy_ci95"]["low"],
                                accuracy_ci95_high=value["accuracy_ci95"]["high"], n_groups=value["n_groups"])
                    per_seed.append(item)
    return rows, per_seed


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    thresholds = sorted({r["gpr_threshold"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(19, max(9, len(rows) / len(thresholds) * .30)), squeeze=False)
    for ax, threshold in zip(axes[0], thresholds):
        subset = sorted((r for r in rows if r["gpr_threshold"] == threshold), key=lambda r: (-r["balanced_accuracy_mean"], r["model_id"]))
        y = np.arange(len(subset))
        mean = np.array([r["balanced_accuracy_mean"] for r in subset]) * 100
        low = np.array([r["balanced_accuracy_min"] for r in subset]) * 100
        high = np.array([r["balanced_accuracy_max"] for r in subset]) * 100
        colors = ["#1767a6" if r["category"] == "quantum_related_simulation" else "#247a48" for r in subset]
        ax.barh(y, [r["accuracy_mean"] * 100 for r in subset], color="#e1e5e8", height=.65, label="Accuracy mean")
        ax.errorbar(mean, y, xerr=[mean-low, high-mean], fmt="none", ecolor="#303a45", linewidth=1, capsize=2)
        ax.scatter(mean, y, c=colors, s=22, zorder=3, label="Balanced accuracy (seed mean/range)")
        ax.scatter([r["fail_recall_mean"] * 100 for r in subset], y, marker="x", color="#c34436", s=22, label="Failure recall mean")
        ax.axvline(50, linestyle="--", color="#8b939c", linewidth=.7)
        ax.set(yticks=y, yticklabels=[r["model_id"] for r in subset], xlim=(-1, 102), xlabel="Percent", title=f"GPR threshold {threshold:g}%")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=.18)

    fig.suptitle("Fixed grouped OOF benchmark | 3 training seeds | exploratory, no external validation\nBlue: quantum-related simulation; green: classical / exact QUBO control", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(output / "all_models.png", dpi=160)
    fig.savefig(output / "all_models.pdf")
    plt.close(fig)


def report(paths, output, *, plot=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Report output must be new/empty")
    batches, shared = load_batches(paths)
    rows, per_seed = summary_rows(batches)
    composition = {}
    for batch in batches:
        for model in batch["config"]["active_model_ids"]:
            for threshold in shared["thresholds"]:
                for seed in shared["seeds"]:
                    entry = batch["results"]["per_model_threshold_seed_oof"][f"{model}__gpr{threshold:g}__seed{seed}"]
                    counts = {k: entry[k] for k in ("n", "fail", "pass", "n_groups")}
                    if threshold in composition and composition[threshold] != counts:
                        raise ValueError("OOF cohort composition differs across models/seeds")
                    composition[threshold] = counts
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_models.csv", rows)
    write_csv(output / "per_seed.csv", per_seed)
    coverage = {"schema": "qbench-report-v1", "configs": len({r["model_id"] for r in rows}),
                "fits": sum(b["results"]["successful_unique_tasks"] for b in batches),
                "oof_scopes": len(per_seed), "shared_contract": shared,
                "cohort_composition": composition,
                "comparison": "separate source versions; common data/split/labels/seeds/environment; descriptive only",
                "batches": [{"protocol_id": b["config"]["protocol_id"], "path": str(b["path"].resolve()),
                             "results_sha256": sha256(b["path"] / "results.json"),
                             "predictions_sha256": sha256(b["path"] / "oof_predictions.json"),
                             "source_hashes": b["results"]["source_hashes"], "audits": b["audits"]} for b in batches]}
    write_json(output / "coverage.json", coverage)
    cohort_n = next(iter(composition.values()))["n"]
    composition_text = "；".join(f"{t:g}%阈值有{c['fail']}个失败计划，全通过基线准确率为{100*c['pass']/c['n']:.2f}%" for t, c in sorted(composition.items(), reverse=True))
    lines = ["# 量子经典混合模型固定配置结果", "", f"共 **{coverage['configs']} 个配置、{coverage['fits']} 次拟合、{coverage['oof_scopes']} 组完整OOF预测**；每组覆盖同一批{cohort_n}个计划。各批均完成独立模型重载验收。", "",
             "本表是同一队列、同一患者/重复几何分组五折和三个训练种子的探索性比较。每个数字先合并五折OOF，再对三个种子求均值；没有挑选最高种子。不同配置的预处理、损失和预算见执行协议，分数差异不能直接归因于量子电路。", "",
             composition_text + "。全通过模型的失败检出率均为0%。应结合balanced accuracy、失败检出率和失败类AP判断。旧644份没有已验证标签映射，因此没有训练分数。", "",
             "QBoost行使用经典精确QUBO求解；其余量子相关结果为经典模拟。没有量子硬件、CUDA-Q或GPU性能结论。组bootstrap区间仅针对固定OOF预测，不含重训及广泛模型搜索的不确定性。", "",
             "完整种子均值/范围及概率指标见 `all_models.csv`，每个种子和准确率区间见 `per_seed.csv`。模型参数、训练历史和逐样本预测保留在各运行目录。", ""]
    for threshold in sorted({r["gpr_threshold"] for r in rows}, reverse=True):
        lines += [f"## GPR通过阈值 {threshold:g}%", "", "排序仅按观察到的balanced accuracy均值，不能据此宣称统计显著或外部验证。", "",
                  "| 配置 | 准确率%（种子范围） | 平衡准确率% | 失败检出率% | ROC-AUC | 失败类AP |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in sorted((r for r in rows if r["gpr_threshold"] == threshold), key=lambda r: (-r["balanced_accuracy_mean"], r["model_id"])):
            a = f"{100*r['accuracy_mean']:.2f} ({100*r['accuracy_min']:.2f}–{100*r['accuracy_max']:.2f})"
            lines.append(f"| `{r['model_id']}` | {a} | {100*r['balanced_accuracy_mean']:.2f} | {100*r['fail_recall_mean']:.2f} | {r['auc_pass_mean']:.3f} | {r['average_precision_fail_mean']:.3f} |")
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    if plot:
        figures(rows, output)
    write_json(output / "files_sha256.json", {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    result = report(args.aggregates, args.output, plot=not args.no_plot)
    print(json.dumps({key: result[key] for key in ("configs", "fits", "oof_scopes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
