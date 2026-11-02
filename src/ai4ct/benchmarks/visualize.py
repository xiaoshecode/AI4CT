# ai4ct/benchmarks/visualize.py
"""
Visualization utilities — confusion matrix, accuracy/loss curves.

Consistent style across all models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_dir: str,
    class_names=("Class 0", "Class 1"),
    dpi: int = 500,
):
    """Plot and save a normalized confusion matrix."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero for all-negative predictions
    cm_normalized = cm.astype('float') / row_sums

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=14)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='both', which='both', length=5)

    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i,
                    f"{cm[i, j]}\n({cm_normalized[i, j] * 100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout(pad=3.0)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close(fig)
    return path


def plot_curves(
    train_values: List[float],
    test_values: List[float],
    model_name: str,
    output_dir: str,
    ylabel: str,
    xlabel: str = "Epoch",
):
    """Plot train/test curves (accuracy or loss) over epochs/iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, len(train_values) + 1)
    ax.plot(x, train_values, label=f'Train {ylabel}')
    ax.plot(x, test_values,  label=f'Test {ylabel}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_name} Train/Test {ylabel}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    suffix = ylabel.lower().replace(' ', '_')
    path = os.path.join(output_dir, f"{model_name}_{suffix}_curves.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_history(history: Dict, model_name: str, output_dir: str):
    """Save training history dict as a pickle file."""
    import pickle
    path = os.path.join(output_dir, f"{model_name}_history.pkl")
    with open(path, 'wb') as f:
        pickle.dump(history, f)
    return path
