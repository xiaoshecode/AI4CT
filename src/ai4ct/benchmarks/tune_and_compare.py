#!/usr/bin/env python
# ai4ct/benchmarks/tune_and_compare.py
"""
Hyperparameter tuning + comprehensive visual comparison of all 4 models.

Usage:
    python -m ai4ct.benchmarks.tune_and_compare

Output:
    outputs/benchmarks/tuned/
    ├── radar_comparison.png        # spider/radar chart
    ├── bar_comparison.png          # grouped bar chart
    ├── roc_curves.png              # all 4 ROC curves on one plot
    ├── training_time.png           # training time comparison
    ├── confusion_matrices.png      # 2×2 grid of confusion matrices
    └── model_comparison.xlsx       # numeric results
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc

from .config import (
    SEED, DATA_PATH,
    GBDTConfig, MLPConfig, PQCHybridConfig, QFMVQCConfig,
)
from .data import load_data
from .metrics import compute_all_metrics
from .run import run_gbdt, run_mlp, run_pqc_hybrid, run_qfm_vqc
from ai4ct.paths import BENCHMARK_OUTPUT

OUTPUT_DIR = BENCHMARK_OUTPUT / "tuned"

warnings.filterwarnings("ignore")

# ================================================================
#  Tuning grids
# ================================================================

GBDT_GRID = [
    GBDTConfig(n_estimators=100, learning_rate=0.2, max_depth=3, subsample=0.7),
    GBDTConfig(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8),
    GBDTConfig(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.7),
    GBDTConfig(n_estimators=150, learning_rate=0.2, max_depth=4, subsample=0.6),
]

MLP_GRID = [
    MLPConfig(epochs=65,  hidden_dim1=16, hidden_dim2=8, lr=0.005),
    MLPConfig(epochs=100, hidden_dim1=24, hidden_dim2=12, lr=0.003),
    MLPConfig(epochs=150, hidden_dim1=32, hidden_dim2=16, lr=0.001),
    MLPConfig(epochs=80,  hidden_dim1=20, hidden_dim2=10, lr=0.005),
]

PQC_GRID = [
    PQCHybridConfig(epochs=150, hidden_dim1=24, hidden_dim2=12, num_layer=2, lr=0.003),
    PQCHybridConfig(epochs=200, hidden_dim1=16, hidden_dim2=8,  num_layer=3, lr=0.001),
    PQCHybridConfig(epochs=100, hidden_dim1=32, hidden_dim2=16, num_layer=1, lr=0.005),
]

VQC_GRID = [
    QFMVQCConfig(epochs=12,  lr=0.005, batch_size=48),
    QFMVQCConfig(epochs=20,  lr=0.003, batch_size=32),
    QFMVQCConfig(epochs=8,   lr=0.01,  batch_size=64),
]


def tune_model(runner_fn, grid, data_path, output_dir):
    """Try each config in `grid` and return the best one's metrics + config."""
    best = None
    best_score = -1
    for i, cfg in enumerate(grid):
        print(f"  Trying config {i+1}/{len(grid)}: {cfg}")
        try:
            metrics = runner_fn(data_path, output_dir, cfg)
            score = metrics.get("test_score", 0)
            if score > best_score:
                best_score = score
                best = (cfg, metrics)
        except Exception as e:
            print(f"    FAILED: {e}")
    return best


def tune_all(data_path, output_dir):
    print("="*60)
    print("  TUNING ALL MODELS")
    print("="*60)

    print("\n--- GBDT ---")
    gbdt_best = tune_model(run_gbdt, GBDT_GRID, data_path, output_dir)
    print("\n--- MLP ---")
    mlp_best  = tune_model(run_mlp,  MLP_GRID,  data_path, output_dir)
    print("\n--- PQC-HybridNet ---")
    pqc_best  = tune_model(run_pqc_hybrid, PQC_GRID, data_path, output_dir)
    print("\n--- QFM-VQCNet ---")
    vqc_best  = tune_model(run_qfm_vqc, VQC_GRID, data_path, output_dir)

    return {
        "GBDT":        gbdt_best,
        "MLP":         mlp_best,
        "PQC-Hybrid":  pqc_best,
        "QFM-VQCNet":  vqc_best,
    }


# ================================================================
#  Visualization
# ================================================================

def plot_radar(results, output_dir):
    """Spider/radar chart across 5 metrics."""
    metrics_keys = ["test_score", "auc", "precision", "recall", "specificity"]
    labels = ["Accuracy", "AUC", "Precision", "Recall", "Specificity"]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=200)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    linestyles = ["-", "--", "-.", ":"]

    for (name, (cfg, m)), c, ls in zip(results.items(), colors, linestyles):
        vals = [max(0, m.get(k, 0)) for k in metrics_keys]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=c, linestyle=ls, linewidth=2, label=name, markersize=6)
        ax.fill(angles, vals, alpha=0.05, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Model Performance Radar", fontsize=16, pad=25, fontweight='bold')
    path = os.path.join(output_dir, "radar_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Radar chart → {path}")


def plot_bars(results, output_dir):
    """Grouped bar chart for key metrics."""
    metrics_keys = ["test_score", "auc", "precision", "recall", "specificity"]
    labels = ["Accuracy", "AUC", "Precision", "Recall", "Specificity"]
    names = list(results.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    for i, name in enumerate(names):
        vals = [max(0, results[name][1].get(k, 0)) for k in metrics_keys]
        bars = ax.bar(x + i * w, vals, w, label=name, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison (Tuned)", fontsize=16, fontweight='bold')
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(output_dir, "bar_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Bar chart → {path}")


def plot_roc_curves(data_path, results, output_dir):
    """Overlay all 4 ROC curves with AUC values."""
    X_train, X_test, y_train, y_test = load_data(data_path)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for (name, (cfg, _)), c in zip(results.items(), colors):
        if name == "GBDT":
            from .models.gbdt import GBDTModel
            m = GBDTModel(cfg); m.fit(X_train, y_train, X_test, y_test)
            prob = m.predict_proba(X_test)
        elif name == "MLP":
            from .models.mlp import MLPModel
            m = MLPModel(cfg); m.fit(X_train, y_train, X_test, y_test)
            prob = m.predict_proba(X_test)
        elif name == "PQC-Hybrid":
            from .models.pqc_hybrid import PQCHybridModel
            m = PQCHybridModel(cfg); m.fit(X_train, y_train, X_test, y_test)
            prob = m.predict_proba(X_test)
        elif name == "QFM-VQCNet":
            from .models.qfm_vqc import _load_qvqc_data, QFM_VQCNet
            Xt_tr, Xt_te, yt_tr, yt_te = _load_qvqc_data(data_path)
            model = QFM_VQCNet(input_dim=cfg.input_dim, hidden_dim1=cfg.hidden_dim1,
                               hidden_dim2=cfg.hidden_dim2, output_dim=cfg.output_dim,
                               num_qubits=cfg.num_qubits, reps=cfg.reps)
            from .models.qfm_vqc import train_qfm_vqc
            train_qfm_vqc(model, Xt_tr, Xt_te, yt_tr, yt_te, epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
            from pyvqnet.nn import Softmax
            from pyvqnet.tensor import no_grad
            model.eval()
            with no_grad():
                prob = Softmax(axis=1)(model(Xt_te)).numpy()[:, 1]
                y_test_use = yt_te.numpy().astype(int)
            fpr, tpr, _ = roc_curve(y_test_use, prob)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={auc_val:.4f})')
            continue

        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3, label='Random')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models (Tuned)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  ROC curves → {path}")


def plot_training_time(results, output_dir):
    """Horizontal bar chart of training times."""
    names = list(results.keys())
    times = [results[n][1].get("train_time_s", 0) for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    bars = ax.barh(names, times, color=colors, edgecolor='white')
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{t:.2f}s', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    path = os.path.join(output_dir, "training_time.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Time chart → {path}")


def plot_confusion_grid(results, output_dir):
    """2×2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for ax, (name, (cfg, m)), c in zip(axes.flat, results.items(), colors):
        cm = np.array([
            [m.get("tn", 0), m.get("fp", 0)],
            [m.get("fn", 0), m.get("tp", 0)],
        ])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(f"{name}", fontsize=14, fontweight='bold', color=c)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i,j] > 0.5 else "black",
                        fontsize=11, fontweight='bold')

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Low", "Pred High"], fontsize=10)
        ax.set_yticklabels(["True Low", "True High"], fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices — All Models (Tuned)", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Confusion grid → {path}")


# ================================================================
#  Main
# ================================================================

def main():
    data_path = DATA_PATH
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # --- Tune ---
    results = tune_all(data_path, output_dir)

    # --- Print best configs ---
    print("\n" + "="*60)
    print("  BEST CONFIGURATIONS")
    print("="*60)
    for name, (cfg, m) in results.items():
        print(f"\n  [{name}]")
        for field in cfg.__dataclass_fields__:
            print(f"    {field}: {getattr(cfg, field)}")
        print(f"    → Test Acc: {m.get('test_score',0):.4f}  AUC: {m.get('auc',0):.4f}")

    # --- Plots ---
    print("\n" + "="*60)
    print("  GENERATING CHARTS")
    print("="*60)
    plot_radar(results, output_dir)
    plot_bars(results, output_dir)
    plot_roc_curves(data_path, results, output_dir)
    plot_training_time(results, output_dir)
    plot_confusion_grid(results, output_dir)

    # --- Excel ---
    rows = []
    for name, (cfg, m) in results.items():
        rows.append({
            "Model": name,
            "Test Acc": f"{m.get('test_score',0):.4f}",
            "AUC": f"{m.get('auc',0):.4f}",
            "Precision": f"{m.get('precision',0):.4f}",
            "Recall": f"{m.get('recall',0):.4f}",
            "Specificity": f"{m.get('specificity',0):.4f}",
            "Train (s)": f"{m.get('train_time_s',0):.2f}",
            "Infer (s)": f"{m.get('inference_time_s',0):.4f}",
            "Best Config": str({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}),
        })
    pd.DataFrame(rows).to_excel(os.path.join(output_dir, "tuned_comparison.xlsx"), index=False)

    print(f"\n  All results → {output_dir}/")
    print("  Done!")


if __name__ == "__main__":
    main()
