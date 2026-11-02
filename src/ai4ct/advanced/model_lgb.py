# ai4ct/advanced/model_lgb.py
"""LightGBM + SHAP — gradient boosting on 56 handcrafted features."""
import time, os, json, warnings, numpy as np, pandas as pd
import lightgbm as lgb
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)

from .config import SEED, OUTPUT_DIR

warnings.filterwarnings("ignore")

def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train LightGBM with scale_pos_weight and SHAP explainability."""
    model_dir = os.path.join(OUTPUT_DIR, "lgb"); os.makedirs(model_dir, exist_ok=True)
    scale_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # --- Train ---
    t0 = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt',
        'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'scale_pos_weight': scale_weight, 'random_state': SEED, 'verbose': -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    train_time = time.time() - t0

    # --- Evaluate ---
    t1 = time.time()
    prob = model.predict(X_test)
    pred = (prob >= 0.5).astype(int)
    infer_time = time.time() - t1

    acc  = accuracy_score(y_test, pred)
    sens = recall_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_test, pred, zero_division=0)
    bal  = (sens + spec) / 2.0
    auc_v = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)

    print(f"[LGB] Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} "
          f"Prec={prec:.4f} Bal={bal:.4f} AUC={auc_v:.4f} Time={train_time:.1f}s")

    # --- SHAP ---
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "shap_summary.png"), dpi=200); plt.close()

    # Importance ranking
    imp = np.abs(shap_vals).mean(axis=0)
    top10_idx = np.argsort(imp)[-10:][::-1]
    top10 = [(feature_names[i], float(imp[i])) for i in top10_idx]

    # ROC
    fig, ax = plt.subplots(figsize=(6,6), dpi=150)
    ax.plot(fpr, tpr, lw=2, label=f'LGB (AUC={auc_v:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("LightGBM ROC")
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "roc.png"), dpi=150); plt.close()

    # Save SHAP values
    np.savez_compressed(os.path.join(model_dir, "shap_values.npz"),
                        values=shap_vals, X_test=X_test, y_test=y_test)

    return {
        'model': 'LightGBM', 'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
        'precision': prec, 'balanced_accuracy': bal, 'auc': auc_v,
        'train_time_s': train_time, 'inference_time_s': infer_time,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'prob': prob, 'pred': pred, 'top10_shap': top10, 'shap_values': shap_vals,
    }
