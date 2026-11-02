"""Render aggregate OOF comparisons from a completed, immutable experiment.

This module does not read patient-level predictions or train any models.  It
refuses to overwrite an existing image, including on repeated CLI invocations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MODEL_LABELS = {
    "gbdt": "GBDT",
    "mlp": "MLP",
    "pqc_hybrid": "PQC\nHybridNet",
    "qfm_vqc": "QFM\nVQCNet",
}
METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "specificity_fail": "Failure recall",
}
COLORS = ("#0072B2", "#009E73", "#CC79A7", "#E69F00")


def render_comparison(run_dir: Path, output: Path) -> Path:
    """Create a two-threshold, three-metric figure using saved OOF summaries."""
    run_dir, output = Path(run_dir), Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Output must be a .png file")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output path")
    result_bytes = (run_dir / "results.json").read_bytes()
    results = json.loads(result_bytes)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only completed experiments can be plotted")
    thresholds = (95, 90)
    for threshold in thresholds:
        table = results[f"gpr_{threshold}"]
        reference = table["majority"]["oof"]
        if reference["n"] != protocol["n_samples"]:
            raise ValueError("Results and protocol sample counts differ")
        for model in (*MODEL_LABELS, "majority"):
            row = table[model]["oof"]
            if any(row[key] != reference[key] for key in ("n", "fail", "pass")):
                raise ValueError("Model summaries do not describe the same cohort")
            for metric in METRICS:
                value = row[metric]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {model}/{metric} in saved results")

    # Plotting is optional and imported only when requested; import has no fileside effects.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), sharey=True)
        for row_index, threshold in enumerate(thresholds):
            table = results[f"gpr_{threshold}"]
            reference = table["majority"]["oof"]
            for col_index, (metric, title) in enumerate(METRICS.items()):
                ax = axes[row_index, col_index]
                values = [100 * table[model]["oof"][metric] for model in MODEL_LABELS]
                bars = ax.bar(range(4), values, color=COLORS, width=0.65, zorder=3)
                ax.axhline(100 * reference[metric], color="#555555", linestyle="--",
                           linewidth=1.25, zorder=4)
                # The stored CI only describes accuracy of fixed held-out predictions.
                if metric == "accuracy" and all("accuracy_ci95" in table[m]["oof"] for m in MODEL_LABELS):
                    lower = [100 * (table[m]["oof"][metric] - table[m]["oof"]["accuracy_ci95"]["low"]) for m in MODEL_LABELS]
                    upper = [100 * (table[m]["oof"]["accuracy_ci95"]["high"] - table[m]["oof"][metric]) for m in MODEL_LABELS]
                    ax.errorbar(range(4), values, yerr=[lower, upper], fmt="none",
                                ecolor="#222222", elinewidth=1, capsize=3, zorder=5)
                    label_heights = [v + u for v, u in zip(values, upper)]
                else:
                    label_heights = values
                for bar, value, height in zip(bars, values, label_heights):
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1.6,
                            f"{value:.1f}", ha="center", va="bottom", fontsize=9)
                ax.set_xticks(range(4), MODEL_LABELS.values())
                ax.set_ylim(0, 106)
                ax.set_yticks(range(0, 101, 20))
                ax.yaxis.set_major_formatter(PercentFormatter(100))
                ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
                ax.set_axisbelow(True)
                ax.set_title(title, fontsize=12, pad=10)
                if col_index == 0:
                    role = "Primary" if threshold == 95 else "Secondary"
                    ax.set_ylabel(f"{role}: GPR threshold {threshold}%\n"
                                  f"{reference['fail']} fail / {reference['pass']} pass", labelpad=12)

        fig.suptitle("TaiChi single-layer cohort: fixed-model, grouped five-fold OOF results",
                     fontsize=15, y=0.98)
        fig.text(0.5, 0.935,
                 f"n = {protocol['n_samples']} plans; {protocol['n_groups']} patient/duplicate groups; "
                 "17 features; no normalization; training-only 10% outlier filter",
                 ha="center", fontsize=10)
        fig.legend([Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.25)],
                   ["Always-pass baseline (failure recall = 0%)"], loc="lower center",
                   bbox_to_anchor=(0.5, 0.065), frameon=False)
        fig.text(0.5, 0.045,
                 "Accuracy error bars: 95% cluster-bootstrap intervals for saved OOF predictions; "
                 "no refitting. Other panels show point estimates.", ha="center", fontsize=9)
        fig.text(0.5, 0.02,
                 "Each plan is predicted by a model trained without its group. "
                 "These are cohort-adaptation results, not a direct replication of the paper's test set.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.87, bottom=0.18,
                            wspace=0.16, hspace=0.43)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as handle:
                fig.savefig(handle, format="png", dpi=200,
                            metadata={"Description": "Aggregate OOF comparison; no patient-level data",
                                      "SourceResultsSHA256": hashlib.sha256(result_bytes).hexdigest()})
        finally:
            plt.close(fig)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed experiment directory")
    parser.add_argument("--output", type=Path, required=True, help="New PNG path (existing files are rejected)")
    args = parser.parse_args(argv)
    print(render_comparison(args.run_dir, args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
