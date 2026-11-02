#!/usr/bin/env python
# ai4ct/advanced/run_all.py
"""Train all 3 models on 644 samples and compare results."""
import os, json, time, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .config import OUTPUT_DIR
from .data_loader import load_full_dataset, split_data

warnings.filterwarnings("ignore")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*60)
    print("  NEWMODEL BENCHMARK — 3 Models on 644 Samples")
    print("="*60)

    # --- Load ---
    print("\n[1] Loading data...")
    X_feat, X_raw, y, feature_names, labels = load_full_dataset()
    (Xf_tr, Xr_tr, y_tr), (Xf_val, Xr_val, y_val), (Xf_te, Xr_te, y_te), \
        _, _, _ = split_data(X_feat, X_raw, y)
    print(f"    Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")
    print(f"    Pass: {(y_te==1).sum()}/{len(y_te)}  Fail: {(y_te==0).sum()}/{len(y_te)}")

    results = {}

    # --- Model 1: LightGBM ---
    print("\n[2] Training LightGBM + SHAP...")
    from .model_lgb import train_and_eval as run_lgb
    results['LightGBM'] = run_lgb(Xf_tr, y_tr, Xf_val, y_val, Xf_te, y_te, feature_names)

    # --- Model 2: Attention-GRU ---
    print("\n[3] Training Attention-GRU + Focal Loss...")
    from .model_attn_gru import train_and_eval as run_gru
    results['Attention-GRU'] = run_gru(Xr_tr, y_tr, Xr_val, y_val, Xr_te, y_te)

    # --- Model 3: InceptionTime ---
    print("\n[4] Training InceptionTime + Tversky Loss...")
    from .model_inception import train_and_eval as run_inc
    results['InceptionTime'] = run_inc(Xr_tr, y_tr, Xr_val, y_val, Xr_te, y_te)

    # --- Comparison ---
    print("\n" + "="*70)
    print("  FINAL COMPARISON (Test Set)")
    print("="*70)
    hdr = f'  {"Model":<18s} {"Acc":>6s} {"Sens":>6s} {"Spec":>6s} {"Prec":>6s} {"Bal":>6s} {"AUC":>6s} {"Time":>7s}'
    print(hdr); print("  " + "-"*66)
    rows = []
    for name, r in results.items():
        print(f'  {name:<18s} {r["accuracy"]:>6.4f} {r["sensitivity"]:>6.4f} '
              f'{r["specificity"]:>6.4f} {r["precision"]:>6.4f} '
              f'{r["balanced_accuracy"]:>6.4f} {r["auc"]:>6.4f} {r["train_time_s"]:>6.1f}s')
        rows.append({k: r[k] for k in ['model','accuracy','sensitivity','specificity',
                                        'precision','balanced_accuracy','auc','train_time_s']})

    # --- ROC overlay ---
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    colors = {'LightGBM':'#2196F3','Attention-GRU':'#4CAF50','InceptionTime':'#FF9800'}
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_te, r['prob'])
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC={r['auc']:.4f})")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(fontsize=11); ax.grid(alpha=.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves — 3 Models (644 samples)")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roc_comparison.png"),dpi=200); plt.close()

    # --- Bar comparison ---
    metrics = ['accuracy','sensitivity','specificity','precision','balanced_accuracy','auc']
    x = np.arange(len(metrics)); w = 0.25
    fig, ax = plt.subplots(figsize=(12,6), dpi=200)
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        ax.bar(x+i*w, vals, w, label=name, color=list(colors.values())[i], edgecolor='white')
    ax.set_xticks(x+w); ax.set_xticklabels(['Acc','Sens','Spec','Prec','BalAcc','AUC'], fontsize=11)
    ax.set_ylim(0,1.05); ax.legend(fontsize=11); ax.grid(axis='y',alpha=.3)
    ax.set_title("Model Comparison — 644 Samples"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"bar_comparison.png"),dpi=200); plt.close()

    # --- Save ---
    pd.DataFrame(rows).to_excel(os.path.join(OUTPUT_DIR,"comparison.xlsx"), index=False)
    with open(os.path.join(OUTPUT_DIR,"results.json"),"w") as f:
        json.dump({k:{kk:vv for kk,vv in v.items() if kk not in ('prob','pred','shap_values')}
                   for k,v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f"\n  Results → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
