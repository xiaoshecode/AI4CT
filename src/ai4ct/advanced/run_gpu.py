#!/usr/bin/env python
# ai4ct/advanced/run_gpu.py
"""GPU-optimized benchmark with hyperparameter tuning on 644 samples."""
import os, json, time, warnings, math, random, numpy as np, pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as skm

from .config import OUTPUT_DIR
from .data_loader import load_full_dataset, split_data

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    print("="*60)
    print("  GPU-OPTIMIZED BENCHMARK — 3 Models on 644 Samples")
    print("="*60)

    # --- Load ---
    print("\n[1] Loading data...")
    X_feat, X_raw, y, feature_names, labels = load_full_dataset()
    (Xf_tr, Xr_tr, y_tr), (Xf_val, Xr_val, y_val), (Xf_te, Xr_te, y_te), \
        _, _, _ = split_data(X_feat, X_raw, y)

    results = {}

    # ================================================================
    #  Model 1: LightGBM (tuned)
    # ================================================================
    print("\n[2] LightGBM — hyperparameter tuning...")
    from .model_lgb import train_and_eval as run_lgb
    best_auc = 0; best_params = None
    for lr in [0.02, 0.05, 0.1]:
        for leaves in [15, 31, 63]:
            for depth in [3, 5, 7]:
                scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                import lightgbm as lgb
                dtrain = lgb.Dataset(Xf_tr, label=y_tr)
                dval   = lgb.Dataset(Xf_val, label=y_val, reference=dtrain)
                params = {
                    'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt',
                    'num_leaves': leaves, 'max_depth': depth, 'learning_rate': lr,
                    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
                    'min_child_samples': 15, 'lambda_l1': 0.1, 'lambda_l2': 0.5,
                    'scale_pos_weight': scale, 'random_state': 42, 'verbose': -1,
                }
                model = lgb.train(params, dtrain, num_boost_round=500,
                                  valid_sets=[dval],
                                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
                prob = model.predict(Xf_val)
                auc_v = skm.roc_auc_score(y_val, prob)
                if auc_v > best_auc:
                    best_auc = auc_v; best_params = (lr, leaves, depth)
    print(f"  Best LGB params: lr={best_params[0]}, leaves={best_params[1]}, depth={best_params[2]} (val_auc={best_auc:.4f})")
    results['LightGBM'] = run_lgb(Xf_tr, y_tr, Xf_val, y_val, Xf_te, y_te, feature_names)

    # ================================================================
    #  Model 2: Attention-GRU (GPU + tuned)
    # ================================================================
    print("\n[3] Attention-GRU — GPU training + hyperparameter tuning...")
    from .model_attn_gru import AttnGRU, FocalLoss
    import torch.optim as optim

    # Tune on val set
    best_val_auc = 0; best_config = None
    configs = [
        # (hidden, layers, lr, epochs)
        (64, 2, 1e-3, 200),
        (128, 2, 5e-4, 200),
        (96, 3, 1e-3, 200),
    ]
    for hidden, nlayers, lr, epochs in configs:
        print(f"  Trying: hidden={hidden}, layers={nlayers}, lr={lr}")
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        model = AttnGRU(input_dim=124, hidden=hidden, num_layers=nlayers, dropout=0.3).to(DEVICE)
        criterion = FocalLoss(alpha=0.75, gamma=2.5)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # Quick validation (batched)
        Xt_tune = torch.tensor(np.stack([s[:362] for s in Xr_tr]), dtype=torch.float32)
        yt_tune = torch.tensor(y_tr, dtype=torch.long)
        Xv_tune = torch.tensor(np.stack([s[:362] for s in Xr_val]), dtype=torch.float32).to(DEVICE)
        for ep in range(30):  # 30-epoch quick eval
            model.train()
            for i in range(0, len(Xt_tune), 16):
                Xb = Xt_tune[i:i+16].to(DEVICE); yb = yt_tune[i:i+16].to(DEVICE)
                optimizer.zero_grad(); loss = criterion(model(Xb), yb); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(Xv_tune),1)[:,1].cpu().numpy()
            val_auc = skm.roc_auc_score(y_val, prob)
        print(f"    val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc; best_config = (hidden, nlayers, lr, epochs)

    hidden, nlayers, lr, epochs = best_config
    print(f"  Best GRU: hidden={hidden}, layers={nlayers}, lr={lr}, epochs={epochs}")

    # Full training with best config (BATCHED to avoid OOM)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model_gru = AttnGRU(input_dim=124, hidden=hidden, num_layers=nlayers, dropout=0.3).to(DEVICE)
    criterion = FocalLoss(alpha=0.75, gamma=2.5)
    optimizer = optim.AdamW(model_gru.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time(); best_v_auc = 0; wait = 0; patience = 50
    Xt_all = torch.tensor(np.stack([s[:362] for s in Xr_tr]), dtype=torch.float32)
    yt_all = torch.tensor(y_tr, dtype=torch.long)
    Xv_all = torch.tensor(np.stack([s[:362] for s in Xr_val]), dtype=torch.float32).to(DEVICE)
    yv_all = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    bs = 16  # small batch for 12GB GPU

    for ep in range(epochs):
        model_gru.train()
        perm = torch.randperm(len(Xt_all))
        total_loss = 0.0
        for i in range(0, len(Xt_all), bs):
            idx = perm[i:i+bs]
            Xb, yb = Xt_all[idx].to(DEVICE), yt_all[idx].to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model_gru(Xb), yb)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        scheduler.step()
        model_gru.eval()
        with torch.no_grad():
            pv = torch.softmax(model_gru(Xv_all),1)[:,1].cpu().numpy()
            v_auc = skm.roc_auc_score(y_val, pv)
        if v_auc > best_v_auc: best_v_auc = v_auc; wait = 0
        else: wait += 1
        if wait >= patience: break
        if (ep+1) % 50 == 0:
            print(f"  GRU E{ep+1:>3d} loss={total_loss:.2f} val_auc={v_auc:.4f}")
    train_time = time.time() - t0

    # Evaluate on test
    model_gru.eval()
    t1 = time.time()
    with torch.no_grad():
        Xte = torch.tensor(np.stack([s[:362] for s in Xr_te]), dtype=torch.float32).to(DEVICE)
        prob_te = torch.softmax(model_gru(Xte),1)[:,1].cpu().numpy()
        pred_te = (prob_te >= 0.5).astype(int)
        attn = model_gru._attn_weights.cpu().numpy() if model_gru._attn_weights is not None else np.zeros((1,362))
    inf_time = time.time() - t1

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 confusion_matrix, roc_auc_score)
    acc=accuracy_score(y_te,pred_te); sens=recall_score(y_te,pred_te)
    tn,fp,fn,tp=confusion_matrix(y_te,pred_te).ravel()
    spec=tn/(tn+fp) if (tn+fp)>0 else 0
    prec=precision_score(y_te,pred_te,zero_division=0)
    bal=(sens+spec)/2; au=roc_auc_score(y_te,prob_te)
    print(f"  GRU Test: Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} AUC={au:.4f} Time={train_time:.1f}s")
    results['Attention-GRU'] = {'model':'Attention-GRU','accuracy':acc,'sensitivity':sens,'specificity':spec,
                                'precision':prec,'balanced_accuracy':bal,'auc':au,
                                'train_time_s':train_time,'inference_time_s':inf_time,
                                'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
                                'prob':prob_te,'pred':pred_te}

    # Save attention heatmap
    fig, ax = plt.subplots(figsize=(12,3), dpi=150)
    a0 = attn[0] if attn.ndim > 1 else attn
    ax.plot(a0,'b-',lw=.5); ax.fill_between(range(len(a0)),0,a0,alpha=.3)
    ax.set_xlabel("Control Point"); ax.set_ylabel("Attention")
    ax.set_title(f"Attention-GRU (GPU) — AUC={au:.4f}")
    os.makedirs(os.path.join(OUTPUT_DIR,"attn_gru_gpu"), exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"attn_gru_gpu","attention.png"),dpi=150); plt.close()

    # ================================================================
    #  Model 3: InceptionTime (GPU + tuned)
    # ================================================================
    print("\n[4] InceptionTime — GPU training + hyperparameter tuning...")
    from .model_inception import InceptionTimeModel, TverskyLoss

    best_v = 0; best_inc = None
    inc_configs = [
        (64, 1e-3, 300), (128, 5e-4, 300), (96, 1e-3, 300),
    ]
    Xt_tune_inc = torch.tensor(np.stack([s[:362] for s in Xr_tr]), dtype=torch.float32)
    yt_tune_inc = torch.tensor(y_tr, dtype=torch.long)
    Xv_tune_inc = torch.tensor(np.stack([s[:362] for s in Xr_val]), dtype=torch.float32).to(DEVICE)

    for hidden, lr, epochs in inc_configs:
        print(f"  Trying: hidden={hidden}, lr={lr}")
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        m_inc = InceptionTimeModel(in_channels=124, hidden=hidden).to(DEVICE)
        crit_inc = TverskyLoss(alpha=0.7, beta=0.3)
        opt_inc = optim.AdamW(m_inc.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(30):
            m_inc.train()
            for i in range(0, len(Xt_tune_inc), 8):
                Xb = Xt_tune_inc[i:i+8].to(DEVICE); yb = yt_tune_inc[i:i+8].to(DEVICE)
                opt_inc.zero_grad(); loss_i = crit_inc(m_inc(Xb), yb); loss_i.backward(); opt_inc.step()
        m_inc.eval()
        with torch.no_grad():
            pv = torch.softmax(m_inc(Xv_tune_inc),1)[:,1].cpu().numpy()
            v_auc = skm.roc_auc_score(y_val, pv)
        print(f"    val_auc={v_auc:.4f}")
        if v_auc > best_v: best_v = v_auc; best_inc = (hidden, lr, epochs)

    hidden, lr, epochs = best_inc
    print(f"  Best IncTime: hidden={hidden}, lr={lr}, epochs={epochs}")

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    m_inc = InceptionTimeModel(in_channels=124, hidden=hidden).to(DEVICE)
    crit_inc = TverskyLoss(alpha=0.7, beta=0.3)
    opt_inc = optim.AdamW(m_inc.parameters(), lr=lr, weight_decay=1e-4)
    sched_inc = optim.lr_scheduler.ReduceLROnPlateau(opt_inc, mode='max', factor=0.5, patience=25)

    t0 = time.time(); best_v_auc=0; wait=0; bs_inc = 8
    for ep in range(epochs):
        m_inc.train(); perm = torch.randperm(len(Xt_tune_inc)); total_li = 0.0
        for i in range(0, len(Xt_tune_inc), bs_inc):
            idx = perm[i:i+bs_inc]
            Xb = Xt_tune_inc[idx].to(DEVICE); yb = yt_tune_inc[idx].to(DEVICE)
            opt_inc.zero_grad(); loss_i = crit_inc(m_inc(Xb), yb)
            loss_i.backward(); opt_inc.step(); total_li += loss_i.item()
        m_inc.eval()
        with torch.no_grad():
            pv = torch.softmax(m_inc(Xv_tune_inc),1)[:,1].cpu().numpy()
            v_auc = skm.roc_auc_score(y_val, pv)
        sched_inc.step(v_auc)
        if v_auc > best_v_auc: best_v_auc = v_auc; wait = 0
        else: wait += 1
        if wait >= 60: break
        if (ep+1) % 60 == 0:
            print(f"  INC E{ep+1:>3d} loss={total_li:.2f} val_auc={v_auc:.4f}")
    train_time = time.time() - t0

    m_inc.eval(); t1=time.time()
    with torch.no_grad():
        pte = torch.softmax(m_inc(Xte),1)[:,1].cpu().numpy()
        pde = (pte >= 0.5).astype(int)
        cam = m_inc.cam_heatmap(Xte[0:1]).numpy()[0]
    inf_time = time.time() - t1

    acc=accuracy_score(y_te,pde); sens=recall_score(y_te,pde)
    tn,fp,fn,tp=confusion_matrix(y_te,pde).ravel()
    spec=tn/(tn+fp) if (tn+fp)>0 else 0; prec=precision_score(y_te,pde,zero_division=0)
    bal=(sens+spec)/2; au_inc=roc_auc_score(y_te,pte)
    print(f"  INC Test: Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} AUC={au_inc:.4f} Time={train_time:.1f}s")
    results['InceptionTime'] = {'model':'InceptionTime','accuracy':acc,'sensitivity':sens,'specificity':spec,
                                'precision':prec,'balanced_accuracy':bal,'auc':au_inc,
                                'train_time_s':train_time,'inference_time_s':inf_time,
                                'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
                                'prob':pte,'pred':pde}

    # CAM heatmap
    fig, ax = plt.subplots(figsize=(12,3), dpi=150)
    ax.plot(cam,'g-',lw=.5); ax.fill_between(range(len(cam)),0,cam,alpha=.3,color='green')
    ax.set_xlabel("Control Point"); ax.set_ylabel("Score-CAM")
    ax.set_title(f"InceptionTime (GPU) — AUC={au_inc:.4f}")
    os.makedirs(os.path.join(OUTPUT_DIR,"inception_gpu"), exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"inception_gpu","scorecam.png"),dpi=150); plt.close()

    # ================================================================
    #  Comparison
    # ================================================================
    print("\n" + "="*70)
    print("  GPU-OPTIMIZED COMPARISON (Test Set)")
    print("="*70)
    print(f'  {"Model":<18s} {"Acc":>6s} {"Sens":>6s} {"Spec":>6s} {"Prec":>6s} {"Bal":>6s} {"AUC":>6s} {"Time":>7s}')
    print("  " + "-"*66)
    for name, r in results.items():
        print(f'  {name:<18s} {r["accuracy"]:>6.4f} {r["sensitivity"]:>6.4f} '
              f'{r["specificity"]:>6.4f} {r["precision"]:>6.4f} '
              f'{r["balanced_accuracy"]:>6.4f} {r["auc"]:>6.4f} {r["train_time_s"]:>6.1f}s')

    # ROC
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    colors = {'LightGBM':'#2196F3','Attention-GRU':'#4CAF50','InceptionTime':'#FF9800'}
    for name, r in results.items():
        if 'prob' in r:
            fpr, tpr, _ = roc_curve(y_te, r['prob'])
            ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC={r['auc']:.4f})")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(fontsize=11); ax.grid(alpha=.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC — GPU-Optimized (644 samples)")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roc_gpu.png"),dpi=200); plt.close()

    # Bar
    metrics = ['accuracy','sensitivity','specificity','precision','balanced_accuracy','auc']
    x = np.arange(len(metrics)); w = 0.25
    fig, ax = plt.subplots(figsize=(12,6), dpi=200)
    for i,(name,r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        ax.bar(x+i*w, vals, w, label=name, color=list(colors.values())[i], edgecolor='white')
    ax.set_xticks(x+w); ax.set_xticklabels(['Acc','Sens','Spec','Prec','BalAcc','AUC'], fontsize=11)
    ax.set_ylim(0,1.05); ax.legend(fontsize=11); ax.grid(axis='y',alpha=.3)
    ax.set_title("GPU-Optimized Comparison — 644 Samples"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"bar_gpu.png"),dpi=200); plt.close()

    pd.DataFrame([{k:r[k] for k in ['model','accuracy','sensitivity','specificity',
                                      'precision','balanced_accuracy','auc','train_time_s']}
                  for r in results.values()]).to_excel(os.path.join(OUTPUT_DIR,"comparison_gpu.xlsx"), index=False)
    print(f"\n  Results → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
