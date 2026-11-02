"""Frozen group splits and independent evaluation of held-out predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score, log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(gpr, groups, n_splits=5, seed=42):
    """Use fixed <90 / 90..95 / >=95 strata, shared across label thresholds."""
    rates = np.asarray(gpr, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if rates.ndim != 1 or groups.shape != rates.shape:
        raise ValueError("GPR and group arrays must be aligned one-dimensional arrays")
    if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 100)):
        raise ValueError("GPR must be finite percentages in [0, 100]")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Too few independent groups for requested folds")
    strata = np.digitize(rates, [90.0, 95.0])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = np.full(len(rates), -1, dtype=int)
    for fold, (train, test) in enumerate(splitter.split(np.zeros((len(rates), 1)), strata, groups)):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Patient/duplicate group leakage detected")
        if np.any(assignment[test] != -1):
            raise RuntimeError("Sample assigned to more than one held-out fold")
        assignment[test] = fold
    if np.any(assignment < 0):
        raise RuntimeError("Some samples have no held-out fold")
    return assignment


def classification_metrics(y_true, pass_probability):
    """Pass is class 1; specificity is the failure-detection recall."""
    y = np.asarray(y_true)
    p = np.asarray(pass_probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary (fail=0, pass=1)")
    y = y.astype(int)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Predicted pass probabilities must be finite and in [0, 1]")
    # Equal logits select class 0 under the paper's argmax rule.
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "fail": int((y == 0).sum()), "pass": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both_classes else None,
        "sensitivity_pass": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity_fail": float(tn / (tn + fp)) if tn + fp else None,
        "precision_pass": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "auc_pass": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision_fail": float(average_precision_score(1-y, 1-p)) if both_classes else None,
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "confusion_matrix_fail_pass": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def group_bootstrap_accuracy(y_true, pass_probability, groups, seed=42, repetitions=1000):
    """Descriptive percentile CI for fixed OOF predictions, resampling whole groups.

This does not retrain models and does not estimate model-selection uncertainty.
"""
    y = np.asarray(y_true)
    pred = (np.asarray(pass_probability) > 0.5).astype(int)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    group_n = np.array([(groups == group).sum() for group in unique])
    group_correct = np.array([((y == pred) & (groups == group)).sum() for group in unique])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    values = group_correct[samples].sum(axis=1) / group_n[samples].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "repetitions": repetitions,
            "method": "percentile cluster bootstrap of fixed held-out predictions; no refitting"}
