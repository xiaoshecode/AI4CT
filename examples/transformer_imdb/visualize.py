"""
Visualization Utilities for Transformer Training
================================================
Generates publication-quality plots for model evaluation:
1. Confusion matrix heatmap
2. Training & validation loss/accuracy curves
3. ROC curve with AUC score

All plots are saved to the output directory as PNG files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# =============================================================================
# Style Configuration
# =============================================================================

# Use a clean, modern style
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 6),
})


# =============================================================================
# 1. Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    class_names: tuple = ("Negative", "Positive"),
    save_path: Optional[str] = None,
):
    """Plot a confusion matrix as a heatmap.

    The matrix shows both absolute counts and row-normalized percentages,
    making it easy to see both the scale and the per-class accuracy.

    Layout:
        ┌─────────────────────────────────────┐
        │         Predicted Negative  Positive │
        │ Actual Negative    TN        FP      │
        │ Actual Positive    FN        TP      │
        └─────────────────────────────────────┘

    Args:
        tp, tn, fp, fn: Confusion matrix elements
        class_names: Names for the negative and positive classes
        save_path: Path to save the figure (PNG)
    """
    matrix = np.array([[tn, fp], [fn, tp]])
    total = matrix.sum()

    # Row-normalized percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_pct = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(7, 6))

    # Heatmap
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Annotate each cell with count AND percentage
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = row_pct[i, j]
            text = f"{count}\n({pct:.1f}%)"
            color = "white" if count > matrix.max() * 0.5 else "black"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color=color,
            )

    # Axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{class_names[0]}", f"Predicted\n{class_names[1]}"])
    ax.set_yticklabels([f"Actual\n{class_names[0]}", f"Actual\n{class_names[1]}"])

    # Derived metrics for title
    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    ax.set_title(
        f"Confusion Matrix\n"
        f"Accuracy={accuracy:.1f}% | Precision={precision:.1f}% | "
        f"Recall={recall:.1f}% | F1={f1:.1f}%",
        fontsize=13,
        pad=20,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Count", fontsize=11)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Confusion matrix saved to {save_path}")

    plt.close(fig)


# =============================================================================
# 2. Training Curves
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot training & validation loss and accuracy curves.

    Creates a two-panel figure:
    - Top panel: Loss curves (train + val)
    - Bottom panel: Accuracy curves (train + val)

    This lets you diagnose overfitting (val loss increasing while train loss
    decreases), underfitting (both still improving), and learning rate issues
    (loss oscillating or not decreasing).

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure (PNG)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Loss Panel ---
    ax1.plot(epochs, history["train_loss"], "o-", color="#2196F3", linewidth=2,
             markersize=5, label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "s-", color="#FF5722", linewidth=2,
             markersize=5, label="Validation Loss")

    # Mark best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val = history["val_loss"][best_epoch - 1]
    ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax1.annotate(
        f"Best: epoch {best_epoch}\n(val_loss={best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.5, best_val * 1.05),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="green", alpha=0.7),
    )

    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # --- Accuracy Panel ---
    ax2.plot(epochs, history["train_acc"], "o-", color="#2196F3", linewidth=2,
             markersize=5, label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], "s-", color="#FF5722", linewidth=2,
             markersize=5, label="Validation Accuracy")

    # Mark best epoch
    best_acc_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = history["val_acc"][best_acc_epoch - 1]
    ax2.axvline(x=best_acc_epoch, color="green", linestyle="--", alpha=0.5, linewidth=1)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # X-axis: integer epochs
    ax2.set_xticks(epochs)
    ax2.set_xticklabels([str(e) for e in epochs])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Training curves saved to {save_path}")

    plt.close(fig)


# =============================================================================
# 3. ROC Curve
# =============================================================================

def plot_roc_curve(
    labels: "np.ndarray",  # noqa: F821
    probs: "np.ndarray",   # noqa: F821
    save_path: Optional[str] = None,
):
    """Plot the Receiver Operating Characteristic (ROC) curve.

    ROC shows the trade-off between:
    - True Positive Rate (Sensitivity): how many actual positives we catch
    - False Positive Rate (1 - Specificity): how many negatives we misclassify

    The curve is traced by varying the classification threshold from 0 to 1.

    AUC (Area Under the Curve) summarizes performance in a single number:
    - AUC = 1.0: perfect classifier
    - AUC = 0.5: random guessing (diagonal line)
    - AUC < 0.5: worse than random (you could flip the predictions!)

    Args:
        labels: Ground truth binary labels, shape (N,)
        probs: Predicted probabilities, shape (N,)
        save_path: Path to save the figure (PNG)
    """
    # Sort by predicted probability (descending)
    # We compute ROC manually to avoid sklearn dependency
    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_probs = probs[sorted_indices]

    # Cumulative counts
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        print("  Warning: Only one class present, cannot compute ROC")
        return

    tpr_list = []  # True Positive Rate
    fpr_list = []  # False Positive Rate

    tp = 0
    fp = 0

    # Walk through sorted predictions
    for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot ROC curve
    ax.plot(fpr_list, tpr_list, linewidth=2.5, color="#2196F3",
            label=f"Transformer (AUC = {auc:.4f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5,
            label="Random (AUC = 0.500)")

    # Styling
    ax.fill_between(fpr_list, tpr_list, alpha=0.1, color="#2196F3")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — IMDB Sentiment Classification", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # AUC text box
    ax.text(
        0.68, 0.05,
        f"AUC = {auc:.4f}",
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2196F3", alpha=0.9),
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  ROC curve saved to {save_path}")

    plt.close(fig)

    return auc


# =============================================================================
# 4. All-in-One Plotting
# =============================================================================

def generate_all_plots(
    history: Dict[str, List[float]],
    test_metrics: Dict,
    plot_dir: str = str(Path(__file__).resolve().parent / "plots"),
):
    """Generate all evaluation plots from training history and test metrics.

    Args:
        history: Training history dictionary from Trainer
        test_metrics: Evaluation metrics from Trainer.evaluate()
        plot_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print(f"Generating Plots")
    print(f"{'='*60}")

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    plot_training_curves(
        history,
        save_path=str(plot_path / "training_curves.png"),
    )

    # 2. Confusion matrix
    plot_confusion_matrix(
        tp=int(test_metrics["tp"]),
        tn=int(test_metrics["tn"]),
        fp=int(test_metrics["fp"]),
        fn=int(test_metrics["fn"]),
        save_path=str(plot_path / "confusion_matrix.png"),
    )

    # 3. ROC curve
    plot_roc_curve(
        labels=test_metrics["labels"].numpy(),
        probs=test_metrics["probs"].numpy(),
        save_path=str(plot_path / "roc_curve.png"),
    )

    print(f"  All plots saved to {plot_dir}/")
    print(f"{'='*60}\n")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing visualization module...\n")

    # Mock data
    np.random.seed(42)
    history = {
        "train_loss": [0.65, 0.45, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13],
        "val_loss": [0.55, 0.48, 0.42, 0.40, 0.39, 0.41, 0.43, 0.45],
        "train_acc": [0.62, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90],
        "val_acc": [0.65, 0.73, 0.78, 0.80, 0.81, 0.80, 0.79, 0.78],
    }

    test_labels = np.concatenate([np.ones(200), np.zeros(200)])
    test_probs = np.concatenate([
        np.random.beta(8, 2, 200),   # Positive class probs (skewed high)
        np.random.beta(2, 8, 200),   # Negative class probs (skewed low)
    ])

    import torch as _torch
    generate_all_plots(
        history=history,
        test_metrics={
            "tp": 180, "tn": 185, "fp": 15, "fn": 20,
            "labels": _torch.tensor(test_labels),
            "probs": _torch.tensor(test_probs),
        },
        plot_dir=str(Path(__file__).resolve().parent / "plots"),
    )
    print("Visualization test passed")
