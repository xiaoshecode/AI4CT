"""Render aggregate-only cohort-shift reports after independent model replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ai4ct.qbench.data import read_json, resolve_path, sha256, write_json
from .cohort_shift import check_artifact_hashes


def render(run_dir, output):
    run_dir, output = resolve_path(run_dir), resolve_path(output)
    hashes = read_json(run_dir / "files_sha256.json")
    check_artifact_hashes(run_dir, hashes)
    audit = read_json(run_dir / "artifact_audit.json")
    results = read_json(run_dir / "results.json")
    if (audit["status"] != "passed" or results["status"] != "complete"
            or audit["checkpoint_replays"] != results["expected_fits"]
            or audit["files_manifest_sha256"] != sha256(run_dir / "files_sha256.json")
            or results["psqa_supervised_training"] is not False):
        raise ValueError("A complete, independently replayed source-classification run is required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty report directory")
    output.mkdir(parents=True, exist_ok=True)
    stats = read_json(run_dir / "feature_statistics.json")
    primary = [r for r in stats if r["view"] == "all_plans"]
    primary.sort(key=lambda r: abs(r["smd_new_minus_old"] or 0), reverse=True)
    summary = results["summary"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    positions = np.arange(len(primary))
    values = [r["smd_new_minus_old"] or 0 for r in primary]
    ax.barh(positions, values, color=["#007f82" if v >= 0 else "#df8154" for v in values])
    ax.set_yticks(positions, [r["feature"] for r in primary])
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_xlabel("Standardized mean difference (new minus old)")
    ax.set_title("Same 17 feature definitions; all plans")
    ax.grid(axis="x", alpha=.18)
    ax = axes[1]
    labels = [f"{'All plans' if r['view'] == 'all_plans' else 'One per geometry'}\n"
              f"{'Logistic regression' if r['model'] == 'logistic' else 'Random forest'}" for r in summary]
    means = np.array([r["auc_new_mean"] for r in summary])
    low = means - [r["auc_new_min"] for r in summary]
    high = [r["auc_new_max"] for r in summary] - means
    ax.barh(np.arange(len(summary)), means, xerr=[low, high], capsize=4, color="#4775a9")
    ax.set_yticks(np.arange(len(summary)), labels)
    ax.invert_yaxis()
    ax.axvline(.5, color="#a43139", linestyle="--", label="No ranking discrimination: 0.5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("OOF source ROC-AUC; mean and range of 3 split seeds")
    ax.set_title("Cohort membership, NOT QA outcome")
    ax.legend(loc="lower right", fontsize=8)
    for i, value in enumerate(means):
        ax.text(min(value + .04, .94), i, f"{value:.3f}", va="center", fontsize=10)
    fig.suptitle("Old 644 versus new 309 plans: feature distribution diagnostics", fontsize=14)
    fig.text(.5, .012, "Known duplicates and new patient groups stay within folds; old patient identities remain unknown.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output / "cohort_shift.png", dpi=180)
    fig.savefig(output / "cohort_shift.svg")
    plt.close(fig)

    prepared = np.load(run_dir / "prepared.npz", allow_pickle=False)
    names = read_json(run_dir / "feature_schema.json")["ordered_names"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, ax in zip(primary[:4], axes.flat):
        column = names.index(row["feature"])
        for source, color, label in [(0, "#df8154", "Old (644)"), (1, "#007f82", "New (309)")]:
            x = np.sort(prepared["x"][prepared["y_domain"] == source, column])
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=label)
        ax.set_title(f"{row['feature']}: KS D={row['ks_statistic']:.3f}")
        ax.set_xlabel(f"{row['feature']} ({row['unit']})")
        ax.set_ylabel("Empirical cumulative fraction")
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle("Four largest absolute mean shifts (descriptive, no significance selection)")
    fig.tight_layout()
    fig.savefig(output / "feature_ecdf.png", dpi=180)
    fig.savefig(output / "feature_ecdf.svg")
    plt.close(fig)

    lines = ["# 旧644与新309的特征差异及来源分类", "",
             "**这是数据来源识别，不是QA通过/失败预测。旧数据的QA标签映射仍未证实。**", "",
             f"完成{results['successful_fits']}次拟合和独立模型重载；"
             f"重载预测最大绝对误差为{audit['max_prediction_absolute_error']:.3g}。", "",
             "两批共953个计划；统一17特征，原始DICOM哈希和重新提取的特征均已核验。", "",
             "每个种子合并完整5折OOF后计算指标，再报告3种子的均值和范围。"
             "旧患者身份未知，几何/特征代理分组不能保证完整患者独立性。", "",
             "| 视图 | 模型 | 来源准确率 | 来源平衡准确率 | 来源AUC（种子范围） | 每组等权AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        view = "全部计划" if row["view"] == "all_plans" else "每种几何一个代表"
        model = "逻辑回归" if row["model"] == "logistic" else "随机森林"
        lines.append(f"| {view} | {model} | {row['accuracy_mean']:.2%} | "
                     f"{row['balanced_accuracy_mean']:.2%} | {row['auc_new_mean']:.4f} "
                     f"({row['auc_new_min']:.4f}–{row['auc_new_max']:.4f}) | {row['group_weighted_auc_new_mean']:.4f} |")
    lines += ["", "AUC较高说明指定模型能从复杂度特征辨别批次；不能确定差异成因，"
              "也不能据此推断QA准确率下降多少。几何敏感性视图不删除原数据。", "",
              "| 特征 | 单位 | 旧均值 | 新均值 | 标准化均值差（新−旧） | KS D |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in primary:
        smd = "不可定义" if row["smd_new_minus_old"] is None else f"{row['smd_new_minus_old']:+.4f}"
        lines.append(f"| {row['feature']} | {row['unit']} | {row['old_mean']:.5g} | "
                     f"{row['new_mean']:.5g} | {smd} | {row['ks_statistic']:.4f} |")
    lines += ["", "完整标准差、中位数/四分位数、Wasserstein距离、逐种子指标、模型与来源索引在运行目录。"
              "不提供忽略组相关性的KS显著性p值。", "",
              "![特征差异与来源分类](cohort_shift.png)", "",
              "![主要特征的经验累计分布](feature_ecdf.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "report_provenance.json", {"run_path": str(run_dir.resolve()),
               "results_sha256": sha256(run_dir / "results.json"),
               "independent_audit_sha256": sha256(run_dir / "artifact_audit.json"),
               "report_source_sha256": sha256(Path(__file__))})
    write_json(output / "files_sha256.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(render(args.run_dir, args.output))


if __name__ == "__main__":
    main()
