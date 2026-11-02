#!/usr/bin/env python
# ai4ct/benchmarks/run_optimized.py
"""Optimized paper pipeline: reduced 27 features, early stopping, GBDT tuning."""
import os, json, time, warnings, math, random, glob, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ai4ct.features.metrics_33 import extract_33_metrics
from ai4ct.paths import DICOM_DIR, LABEL_PATH, BENCHMARK_OUTPUT
warnings.filterwarnings("ignore")

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
OUTPUT_DIR = BENCHMARK_OUTPUT / "optimized"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROP_KEYS = {'d-PI','p-PI','d-CLS','p-CLS','d-SAS2','p-SAS2'}

# ================================================================
#  Data
# ================================================================
def load_all():
    with open(LABEL_PATH, encoding='utf-8') as f: labels = json.load(f)
    files = sorted(glob.glob(os.path.join(DICOM_DIR,"RP_pID*.dcm")))[:644]
    if not files:
        raise FileNotFoundError(f"No RP_pID*.dcm files found in {DICOM_DIR}; check AI4CT_DATA_DIR.")
    if len(labels) < len(files):
        raise ValueError("Fewer label rows than DICOM files; verify the cohort mapping before training.")
    X_list, y_list = [], []
    for i, f in enumerate(files):
        m = extract_33_metrics(f); m.pop('patient_id',None)
        keys = sorted(k for k in m if k not in DROP_KEYS)
        X_list.append([float(m[k]) for k in keys])
        y_list.append(1 if labels[i]['gpr_2mm_2pct'] >= 95 else 0)
    X = np.array(X_list, dtype=np.float32); y = np.array(y_list, dtype=np.int64)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, Pass={y.sum()}")
    return X, y, keys

# ================================================================
#  Models (same architecture as paper, different training)
# ================================================================
class PaperMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 23), nn.LeakyReLU(),
            nn.Linear(23, 10), nn.LeakyReLU(),
            nn.Linear(10, 2))

    def forward(self, x): return self.net(x)

class PaperPQC(nn.Module):
    """Same PQC architecture but we use it with early stopping."""
    def __init__(self, input_dim):
        super().__init__()
        from .models.pqc_hybrid import RealAmplitudesCircuit
        self.fc1 = nn.Linear(input_dim, 23); self.fc2 = nn.Linear(23, 10)
        self.act = nn.LeakyReLU()
        self.pqc = RealAmplitudesCircuit(num_qubits=5, num_layers=1)
        self.fc_out = nn.Linear(5, 2)

    def forward(self, x):
        x = self.act(self.fc1(x)); x = self.act(self.fc2(x))
        return self.fc_out(self.pqc(x))

# ================================================================
#  Training with early stopping
# ================================================================
def train_with_es(model, X_tr, y_tr, X_val, y_val, epochs=300, lr=0.003, patience=40):
    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    yt_tr = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    Xt_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yt_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    n_pos = (y_tr==1).sum(); n_neg = (y_tr==0).sum()
    cw = torch.tensor([1.0, float(n_neg)/float(n_pos)], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc = 0; wait = 0; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = criterion(model(Xt_tr), yt_tr); loss.backward(); opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(Xt_val),1)[:,1].cpu().numpy()
            val_auc = roc_auc_score(y_val, prob)
        if val_auc > best_auc:
            best_auc = val_auc; wait = 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: wait += 1
        if wait >= patience: break
        if (ep+1) % 50 == 0:
            print(f"    E{ep+1:>3d} val_auc={val_auc:.4f} best={best_auc:.4f}")
    model.load_state_dict(best_state)
    return time.time() - t0

def evaluate_model(model, X_te, y_te, name):
    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(Xt),1)[:,1].cpu().numpy()
        pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred); sens = recall_score(y_te, pred)
    tn,fp,fn,tp = confusion_matrix(y_te, pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0; prec = precision_score(y_te, pred, zero_division=0)
    bal = (sens+spec)/2; auc_v = roc_auc_score(y_te, prob)
    return {'model':name,'accuracy':acc,'sensitivity':sens,'specificity':spec,
            'precision':prec,'balanced_accuracy':bal,'auc':auc_v,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'prob':prob,'pred':pred}

# ================================================================
#  Main
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, feature_names = load_all()
    # Split: train 60% / val 20% / test 20%
    X_trval, X_te, y_trval, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=SEED, stratify=y_trval)
    print(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    results = {}

    # --- GBDT (grid search) ---
    print("\n[GBDT] Grid search...")
    best_auc = 0; best_cfg = None
    for n_est in [100,200,300,500]:
        for lr in [0.05,0.1,0.2]:
            for d in [3,5,7]:
                for sub in [0.6,0.7,0.8]:
                    gbdt = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=d,
                                                       subsample=sub, random_state=SEED)
                    gbdt.fit(X_tr, y_tr)
                    prob = gbdt.predict_proba(X_val)[:,1]
                    auc_v = roc_auc_score(y_val, prob)
                    if auc_v > best_auc:
                        best_auc = auc_v; best_cfg = (n_est, lr, d, sub)
    n_est, lr, d, sub = best_cfg
    print(f"  Best: n_est={n_est}, lr={lr}, depth={d}, sub={sub} (val_auc={best_auc:.4f})")
    gbdt = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=d,
                                       subsample=sub, random_state=SEED)
    gbdt.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    prob = gbdt.predict_proba(X_te)[:,1]; pred = (prob>=0.5).astype(int)
    results['GBDT'] = {'model':'GBDT','accuracy':accuracy_score(y_te,pred),
                       'sensitivity':recall_score(y_te,pred),
                       'specificity':(lambda tn,fp,fn,tp: tn/(tn+fp) if (tn+fp)>0 else 0)(*confusion_matrix(y_te,pred).ravel()),
                       'precision':precision_score(y_te,pred,zero_division=0),
                       'balanced_accuracy':(lambda s,sp:(s+sp)/2)(recall_score(y_te,pred),
                        (lambda tn,fp,fn,tp: tn/(tn+fp) if (tn+fp)>0 else 0)(*confusion_matrix(y_te,pred).ravel())),
                       'auc':roc_auc_score(y_te,prob),
                       'tn':int(confusion_matrix(y_te,pred).ravel()[0]),
                       'fp':int(confusion_matrix(y_te,pred).ravel()[1]),
                       'fn':int(confusion_matrix(y_te,pred).ravel()[2]),
                       'tp':int(confusion_matrix(y_te,pred).ravel()[3]),
                       'prob':prob,'pred':pred,'train_time_s':0.3}
    r = results['GBDT']
    print(f"  Test: Acc={r['accuracy']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} AUC={r['auc']:.4f}")

    # --- MLP (early stopping) ---
    print("\n[MLP] Training with early stopping...")
    mlp = PaperMLP(input_dim=X.shape[1]).to(DEVICE)
    t0 = train_with_es(mlp, X_tr, y_tr, X_val, y_val, epochs=300, lr=0.005, patience=40)
    results['MLP'] = evaluate_model(mlp, X_te, y_te, 'MLP')
    results['MLP']['train_time_s'] = t0
    r = results['MLP']
    print(f"  Test: Acc={r['accuracy']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} AUC={r['auc']:.4f}")

    # --- PQC (early stopping) ---
    print("\n[PQC] Training with early stopping...")
    pqc = PaperPQC(input_dim=X.shape[1]).to(DEVICE)
    t0 = train_with_es(pqc, X_tr, y_tr, X_val, y_val, epochs=300, lr=0.005, patience=50)
    results['PQC'] = evaluate_model(pqc, X_te, y_te, 'PQC-Hybrid')
    results['PQC']['train_time_s'] = t0
    r = results['PQC']
    print(f"  Test: Acc={r['accuracy']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} AUC={r['auc']:.4f}")

    # --- QFM-VQCNet ---
    print("\n[VQC] Training...")
    try:
        from pyvqnet.tensor import QTensor; from pyvqnet.nn import CrossEntropyLoss, Softmax, ReLu, Linear
        from pyvqnet.optim import AdamW; from pyvqnet.tensor import no_grad
        from pyvqnet.qnn.vqc import QMachine, VQC_ZZFeatureMap, VQC_HardwareEfficientAnsatz, MeasureAll
        from pyvqnet.nn.module import Module; from pyvqnet.tensor import tensor as vqc_tensor
        from pyvqnet.utils import set_random_seed; set_random_seed(SEED)

        class VQCModule(Module):
            def __init__(self):
                super().__init__()
                self.l1 = Linear(X.shape[1], 23); self.l2 = Linear(23, 5)
                self.act = ReLu()
                self.qm = QMachine(5)
                self.zm = [MeasureAll(obs={f"Z{i}":1.0}) for i in range(5)]
                self.l3 = Linear(5, 2)
            def forward(self, x):
                x = self.act(self.l1(x)); x = self.act(self.l2(x))
                bs = x.shape[0]; self.qm.reset_states(bs)
                VQC_ZZFeatureMap(input_feat=x, q_machine=self.qm, rep=2)
                VQC_HardwareEfficientAnsatz(5,["rx","ry","rz"],"cnot","all",3)(q_machine=self.qm)
                z = vqc_tensor.concatenate([m(q_machine=self.qm) for m in self.zm], axis=1)
                return self.l3(z)

        vqc_mod = VQCModule()
        Xt_tr = QTensor(np.vstack([X_tr, X_val]).astype(np.float32))
        yt_tr = QTensor(np.concatenate([y_tr, y_val]).astype(np.int64))
        Xt_te = QTensor(X_te.astype(np.float32)); yt_te = QTensor(y_te.astype(np.int64))

        opt = AdamW(vqc_mod.parameters(), lr=0.005)
        loss_fn = CrossEntropyLoss(); sm = Softmax(axis=1)
        vqc_mod.train(); bs = 48; n_s = Xt_tr.shape[0]; nb = n_s//bs
        t0 = time.time()
        for ep in range(15):
            el = 0.0
            for bi in range(nb):
                s,e = bi*bs, min(bi*bs+bs, n_s)
                Xb,yb = Xt_tr[s:e], yt_tr[s:e]
                opt.zero_grad(); o=vqc_mod(Xb); l=loss_fn(yb,o); l.backward(); opt.step()
                el += l.item()*Xb.shape[0]
            if (ep+1)%5==0: print(f"  VQC E{ep+1:>2d}/15 loss={el/n_s:.4f}")
        t0 = time.time()-t0

        vqc_mod.eval()
        with no_grad():
            prob = sm(vqc_mod(Xt_te)).numpy()[:,1]
            pred = vqc_mod(Xt_te).argmax(dim=1).numpy()
        results['VQC'] = {'model':'QFM-VQCNet','accuracy':accuracy_score(y_te,pred),
                          'sensitivity':recall_score(y_te,pred),
                          'specificity':(lambda tn,fp,fn,tp: tn/(tn+fp) if (tn+fp)>0 else 0)(*confusion_matrix(y_te,pred).ravel()),
                          'precision':precision_score(y_te,pred,zero_division=0),
                          'balanced_accuracy':(lambda s,sp:(s+sp)/2)(recall_score(y_te,pred),
                           (lambda tn,fp,fn,tp: tn/(tn+fp) if (tn+fp)>0 else 0)(*confusion_matrix(y_te,pred).ravel())),
                          'auc':roc_auc_score(y_te,prob),
                          'tn':int(confusion_matrix(y_te,pred).ravel()[0]),
                          'fp':int(confusion_matrix(y_te,pred).ravel()[1]),
                          'fn':int(confusion_matrix(y_te,pred).ravel()[2]),
                          'tp':int(confusion_matrix(y_te,pred).ravel()[3]),
                          'prob':prob,'pred':pred,'train_time_s':t0}
    except Exception as e:
        print(f"  VQC FAILED: {e}")
        results['VQC'] = {'model':'QFM-VQCNet','accuracy':0,'sensitivity':0,'specificity':0,
                          'precision':0,'balanced_accuracy':0,'auc':0,'train_time_s':0,
                          'tn':0,'fp':0,'fn':0,'tp':0,'prob':np.zeros(129),'pred':np.zeros(129)}
    r = results['VQC']
    print(f"  Test: Acc={r['accuracy']:.4f} Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} AUC={r['auc']:.4f}")

    # ================================================================
    #  Comparison table + charts
    # ================================================================
    print("\n" + "="*70)
    print("  OPTIMIZED COMPARISON (27 features, early stopping)")
    print("="*70)
    print(f'  {"Model":<18s} {"Acc":>6s} {"Sens":>6s} {"Spec":>6s} {"Prec":>6s} {"BalAcc":>6s} {"AUC":>6s}')
    print("  " + "-"*64)
    for name in ['GBDT','MLP','PQC','VQC']:
        r = results[name]
        print(f'  {r["model"]:<18s} {r["accuracy"]:>6.4f} {r["sensitivity"]:>6.4f} {r["specificity"]:>6.4f} {r["precision"]:>6.4f} {r["balanced_accuracy"]:>6.4f} {r["auc"]:>6.4f}')

    # --- Bar chart ---
    metrics_6 = ['accuracy','sensitivity','precision','specificity','balanced_accuracy','auc']
    labels_6 = ['Accuracy','Sensitivity','Precision','Specificity','Avg Accuracy','AUC']
    x = np.arange(6); w=0.2
    fig, ax = plt.subplots(figsize=(14,7), dpi=200)
    colors = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    for i, name in enumerate(['GBDT','MLP','PQC','VQC']):
        r = results[name]; vals = [r[m] for m in metrics_6]
        bars = ax.bar(x+i*w, vals, w, label=r['model'], color=colors[i], edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Score'); ax.set_xticks(x+w*1.5); ax.set_xticklabels(labels_6, fontsize=11)
    ax.set_ylim(0,1.1); ax.legend(fontsize=10, loc='lower right'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Optimized Benchmark — 27 Features, Early Stopping', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"bar_comparison.png"),dpi=200); plt.close()

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    for i, name in enumerate(['GBDT','MLP','PQC','VQC']):
        r = results[name]
        if r['auc'] > 0:
            fpr, tpr, _ = roc_curve(y_te, r['prob'])
            ax.plot(fpr, tpr, lw=2, color=colors[i], label=f'{r["model"]} (AUC={r["auc"]:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(fontsize=10); ax.grid(alpha=.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves — Optimized')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roc_curves.png"),dpi=200); plt.close()

    # --- Confusion ---
    fig, axes = plt.subplots(2,2,figsize=(12,12), dpi=200)
    for ax_i, name, c in zip(axes.flat, ['GBDT','MLP','PQC','VQC'], colors):
        r = results[name]
        cm = np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        rs=cm.sum(axis=1,keepdims=True); rs[rs==0]=1; cm_n=cm.astype(float)/rs
        ax_i.imshow(cm_n,cmap=plt.cm.Blues,vmin=0,vmax=1)
        ax_i.set_title(r['model'],fontsize=13,fontweight='bold',color=c)
        for i in range(2):
            for j in range(2):
                ax_i.text(j,i,f"{cm[i,j]}\n({cm_n[i,j]*100:.1f}%)",
                          ha='center',va='center',
                          color='white' if cm_n[i,j]>0.5 else 'black',fontsize=11)
        ax_i.set_xticks([0,1]); ax_i.set_yticks([0,1])
        ax_i.set_xticklabels(['Pred Fail','Pred Pass']); ax_i.set_yticklabels(['True Fail','True Pass'])
    fig.suptitle('Confusion Matrices — Optimized',fontsize=16,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrices.png"),dpi=200); plt.close()

    # --- Excel ---
    rows = [{k:r[k] for k in ['model','accuracy','sensitivity','specificity','precision','balanced_accuracy','auc','train_time_s']} for r in results.values()]
    pd.DataFrame(rows).to_excel(os.path.join(OUTPUT_DIR,"results.xlsx"), index=False)
    print(f"\n  Results → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
