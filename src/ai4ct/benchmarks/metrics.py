# ai4ct/benchmarks/metrics.py
"""
Evaluation metrics shared across all models.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss,
)
from typing import Dict, Tuple


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_train_true: np.ndarray = None,
    y_train_pred: np.ndarray = None,
    train_time: float = 0.0,
    inference_time: float = 0.0,
) -> Dict[str, float]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true, y_pred : shape (n,)
        Ground truth and predicted labels.
    y_prob : shape (n,)
        Predicted probability for the positive class.
    y_train_true, y_train_pred : optional
        Training set labels/predictions for computing train accuracy.
    train_time : float
        Training wall-clock time in seconds.
    inference_time : float
        Inference wall-clock time in seconds.

    Returns
    -------
    dict with keys: train_score, test_score, precision, recall,
    specificity, auc, ap, tn, fp, fn, tp, train_time_s, inference_time_s
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "test_score":   accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "specificity":  float(specificity),
        "auc":          roc_auc_score(y_true, y_prob),
        "ap":           average_precision_score(y_true, y_prob),
        "tn": int(tn),  "fp": int(fp),  "fn": int(fn),  "tp": int(tp),
        "train_time_s": train_time,
        "inference_time_s": inference_time,
    }

    if y_train_true is not None and y_train_pred is not None:
        metrics["train_score"] = accuracy_score(y_train_true, y_train_pred)

    return metrics


def metrics_to_dataframe(metrics: Dict[str, float]) -> 'pd.DataFrame':
    """Convert metrics dict to a pandas DataFrame for Excel export."""
    import pandas as pd
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value":  list(metrics.values()),
    })
