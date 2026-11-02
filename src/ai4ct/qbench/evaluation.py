"""Metrics distinguish ranking scores from calibrated probabilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score


def prediction_arrays(score, probability, n):
    score = np.asarray(score, dtype=float)
    if score.shape != (n,) or not np.isfinite(score).all():
        raise ValueError("Pass scores must be finite aligned vectors")
    p = None if probability is None else np.asarray(probability, dtype=float)
    if p is not None and (p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Pass probabilities must be finite in [0,1]")
    predicted = (score > 0).astype(int) if p is None else (p > 0.5).astype(int)
    return score, p, predicted


def metrics(y, score, probability=None):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonempty binary labels required")
    score, p, pred = prediction_arrays(score, probability, len(y))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    fail_recall = float(tn / (tn + fp)) if tn + fp else None
    pass_recall = float(tp / (tp + fn)) if tp + fn else None
    return {"n": len(y), "fail": int((y == 0).sum()), "pass": int((y == 1).sum()),
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": (fail_recall + pass_recall) / 2 if both else None,
            "fail_recall": fail_recall, "pass_recall": pass_recall,
            "fail_precision": float(tn / (tn + fn)) if tn + fn else 0.0,
            "pass_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "auc_pass": float(roc_auc_score(y, score)) if both else None,
            "average_precision_fail": float(average_precision_score(1-y, -score)) if both else None,
            "brier": float(brier_score_loss(y, p)) if p is not None else None,
            "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])) if p is not None else None,
            "probability_available": p is not None,
            "decision_boundary": 0.5 if p is not None else 0.0,
            "decision_kind": "pass_probability" if p is not None else "pass_score",
            "confusion_matrix_fail_pass": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def compare_predictions(a_score, a_p, b_score, b_p):
    a_score, a_p, a_class = prediction_arrays(a_score, a_p, len(a_score))
    b_score, b_p, b_class = prediction_arrays(b_score, b_p, len(a_score))
    if (a_p is None) != (b_p is None):
        raise ValueError("Probability capability changed after loading")
    if not np.allclose(a_score, b_score, rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint scores differ")
    if a_p is not None and not np.allclose(a_p, b_p, rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint probabilities differ")
    if not np.array_equal(a_class, b_class):
        raise ValueError("Checkpoint classes differ")
    return {"max_score_error": float(np.max(np.abs(a_score-b_score))),
            "max_probability_error": None if a_p is None else float(np.max(np.abs(a_p-b_p))),
            "classes_identical": True}
