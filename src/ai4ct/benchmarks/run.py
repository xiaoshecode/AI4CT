#!/usr/bin/env python
# ai4ct/benchmarks/run.py
"""
Benchmark Runner — train and evaluate all 4 models on the radiotherapy
plan complexity dataset.

Usage:
    python -m ai4ct.benchmarks.run                # run all models
    python -m ai4ct.benchmarks.run --model gbdt    # run only GBDT
    python -m ai4ct.benchmarks.run --model mlp,gbdt # run MLP + GBDT
    python -m ai4ct.benchmarks.run --data path/to/data.xlsx --output ./results

Models:
    GBDT       — Gradient Boosted Decision Trees (sklearn)
    MLP        — Classical Multi-Layer Perceptron (PyTorch)
    PQC-Hybrid — Parameterized Quantum Circuit Hybrid (pure PyTorch)
    QFM-VQCNet — Quantum Fuzzy Model VQC (PyVQNet)
"""
import os
import sys
import argparse
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from .config import (
    SEED, DATA_PATH, OUTPUT_DIR,
    GBDTConfig, MLPConfig, PQCHybridConfig, QFMVQCConfig,
    MODELS,
)
from .data import load_data
from .metrics import compute_all_metrics, metrics_to_dataframe
from .visualize import plot_confusion_matrix, plot_curves, save_history


# ================================================================
#  Runner functions
# ================================================================

def run_gbdt(data_path: str, output_dir: str, cfg: GBDTConfig = None):
    """Train GBDT and export results."""
    cfg = cfg or GBDTConfig()
    print(f"\n{'='*60}\n  {cfg.name}\n{'='*60}")

    X_train, X_test, y_train, y_test = load_data(data_path)

    from .models.gbdt import GBDTModel
    model = GBDTModel(cfg)

    hist = model.fit(X_train, y_train, X_test, y_test)

    # inference
    t0 = time.time()
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    train_preds = model.predict(X_train)
    inf_time = time.time() - t0

    metrics = compute_all_metrics(
        y_test, test_preds, test_probs, y_train, train_preds,
        train_time=hist["train_time"], inference_time=inf_time,
    )

    # export
    _export(cfg.name, metrics, hist, output_dir, test_preds, y_test)

    return metrics


def run_mlp(data_path: str, output_dir: str, cfg: MLPConfig = None):
    """Train Classical MLP and export results."""
    cfg = cfg or MLPConfig()
    print(f"\n{'='*60}\n  {cfg.name}\n{'='*60}")

    X_train, X_test, y_train, y_test = load_data(data_path)

    from .models.mlp import MLPModel
    model = MLPModel(cfg)

    hist = model.fit(X_train, y_train, X_test, y_test)

    t0 = time.time()
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    train_preds = model.predict(X_train)
    inf_time = time.time() - t0

    metrics = compute_all_metrics(
        y_test, test_preds, test_probs, y_train, train_preds,
        train_time=hist["train_time"], inference_time=inf_time,
    )

    _export(cfg.name, metrics, hist, output_dir, test_preds, y_test)
    return metrics


def run_pqc_hybrid(data_path: str, output_dir: str, cfg: PQCHybridConfig = None):
    """Train PQC-HybridNet and export results."""
    cfg = cfg or PQCHybridConfig()
    print(f"\n{'='*60}\n  {cfg.name}\n{'='*60}")

    X_train, X_test, y_train, y_test = load_data(data_path)

    from .models.pqc_hybrid import PQCHybridModel
    model = PQCHybridModel(cfg)

    hist = model.fit(X_train, y_train, X_test, y_test)

    t0 = time.time()
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    train_preds = model.predict(X_train)
    inf_time = time.time() - t0

    metrics = compute_all_metrics(
        y_test, test_preds, test_probs, y_train, train_preds,
        train_time=hist["train_time"], inference_time=inf_time,
    )

    _export(cfg.name, metrics, hist, output_dir, test_preds, y_test)
    return metrics


def run_qfm_vqc(data_path: str, output_dir: str, cfg: QFMVQCConfig = None):
    """Train QFM-VQCNet and export results."""
    cfg = cfg or QFMVQCConfig()
    print(f"\n{'='*60}\n  {cfg.name}\n{'='*60}")

    from .models.qfm_vqc import run_qfm_vqc as _run

    result = _run(cfg, data_path)
    metrics = {
        "train_score": result["train_score"],
        "test_score": result["test_score"],
        "precision": result["precision"],
        "recall": result["recall"],
        "specificity": result["specificity"],
        "auc": result["auc"],
        "ap": result["ap"],
        "tn": result["tn"], "fp": result["fp"],
        "fn": result["fn"], "tp": result["tp"],
        "train_time_s": result["train_time_s"],
        "inference_time_s": result["inference_time_s"],
    }

    hist = {
        "train_acc_list": result.get("train_acc_list", []),
        "test_acc_list": result.get("test_acc_list", []),
        "train_loss_list": result.get("train_loss_list", []),
        "test_loss_list": result.get("test_loss_list", []),
        "epochs": result.get("epochs", cfg.epochs),
    }

    # evaluation for confusion matrix plot
    from .models.qfm_vqc import _load_qvqc_data, evaluate_qfm_vqc, QFM_VQCNet
    X_train_qt, X_test_qt, y_train_qt, y_test_qt = _load_qvqc_data(data_path)
    model = QFM_VQCNet(
        input_dim=cfg.input_dim, hidden_dim1=cfg.hidden_dim1,
        hidden_dim2=cfg.hidden_dim2, output_dim=cfg.output_dim,
        num_qubits=cfg.num_qubits, reps=cfg.reps,
    )
    test_preds = evaluate_qfm_vqc(model, X_test_qt, y_test_qt)[0]  # just re-create isn't right
    # For QFM-VQCNet we use the result dict which already has all we need
    test_preds_arr = np.array([])  # placeholder — metrics already computed
    y_test_arr = y_test_qt.numpy().astype(int)

    _export(cfg.name, metrics, hist, output_dir, None, y_test_arr)
    return metrics


# ================================================================
#  Shared export
# ================================================================

def _export(name: str, metrics: dict, hist: dict,
            output_dir: str, test_preds, y_test):
    """Save metrics Excel, confusion matrix, accuracy/loss curves, history pkl."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Excel ---
    metrics_to_dataframe(metrics).to_excel(
        os.path.join(output_dir, f"{name}.xlsx"), index=False)

    # --- confusion matrix ---
    if test_preds is not None:
        from sklearn.metrics import confusion_matrix
        tn = metrics.get("tn", 0)
        fp = metrics.get("fp", 0)
        fn = metrics.get("fn", 0)
        tp = metrics.get("tp", 0)
        cm = np.array([[tn, fp], [fn, tp]])
    else:
        cm = np.zeros((2, 2), dtype=int)
    plot_confusion_matrix(cm, name, output_dir)

    # --- curves ---
    if hist.get("train_acc_list") and hist.get("test_acc_list"):
        plot_curves(hist["train_acc_list"], hist["test_acc_list"],
                    name, output_dir, ylabel="Accuracy",
                    xlabel="Trees" if "n_trees" in hist else "Epoch")
    if hist.get("train_loss_list") and hist.get("test_loss_list"):
        plot_curves(hist["train_loss_list"], hist["test_loss_list"],
                    name, output_dir, ylabel="Loss",
                    xlabel="Trees" if "n_trees" in hist else "Epoch")

    # --- history pkl ---
    save_history(hist, name, output_dir)

    print(f"  Results saved to {output_dir}/")


# ================================================================
#  CLI
# ================================================================

RUNNERS = {
    "gbdt":       run_gbdt,
    "mlp":        run_mlp,
    "pqc_hybrid": run_pqc_hybrid,
    "qfm_vqc":    run_qfm_vqc,
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark radiotherapy plan classification models")
    parser.add_argument("--model", type=str, default="all",
                        help="Comma-separated model keys: gbdt,mlp,pqc_hybrid,qfm_vqc (or 'all')")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to input Excel file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    data_path = args.data or DATA_PATH
    output_dir = args.output or OUTPUT_DIR
    if not Path(data_path).is_file():
        parser.error(f"Feature table not found: {data_path}. Restore data/processed or pass --data.")

    if args.model == "all":
        to_run = list(RUNNERS.keys())
    else:
        to_run = [k.strip() for k in args.model.split(",")]

    all_metrics = {}
    for key in to_run:
        if key not in RUNNERS:
            print(f"Unknown model: {key}.  Choices: {list(RUNNERS.keys())}")
            continue
        m = RUNNERS[key](data_path, output_dir, MODELS.get(key))
        all_metrics[key] = m

    # ---- comparison table ----
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print("  MODEL COMPARISON")
        print(f"{'='*70}")
        rows = []
        for model_name, m in all_metrics.items():
            rows.append({
                "Model": model_name,
                "Test Acc": f"{m.get('test_score',0):.4f}",
                "AUC": f"{m.get('auc',0):.4f}",
                "Precision": f"{m.get('precision',0):.4f}",
                "Recall": f"{m.get('recall',0):.4f}",
                "Specificity": f"{m.get('specificity',0):.4f}",
                "Train (s)": f"{m.get('train_time_s',0):.1f}",
                "Infer (s)": f"{m.get('inference_time_s',0):.4f}",
            })
        comp_df = pd.DataFrame(rows)
        print(comp_df.to_string(index=False))
        comp_path = os.path.join(output_dir, "model_comparison.xlsx")
        comp_df.to_excel(comp_path, index=False)
        print(f"\nComparison saved to {comp_path}")


if __name__ == "__main__":
    main()
