#!/usr/bin/env python
# ai4ct/benchmarks/run_final.py
"""Final benchmark: all features, 4 models, SHAP analysis, comparison charts."""
import os, json, glob, math, time, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai4ct.features.feature_exploration import extract_all_features
from ai4ct.paths import DICOM_DIR, LABEL_PATH, BENCHMARK_OUTPUT

warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = BENCHMARK_OUTPUT / "final"

# ================================================================
#  Data
# ================================================================
def load():
    with open(LABEL_PATH, encoding='utf-8') as f: labels = json.load(f)
    files = sorted(glob.glob(os.path.join(DICOM_DIR,"RP_pID*.dcm")))[:644]
    if not files:
        raise FileNotFoundError(f"No RP_pID*.dcm files found in {DICOM_DIR}; check AI4CT_DATA_DIR.")
    if len(files) != 644 or len(labels) < 644:
        raise ValueError("This historical pipeline requires 644 DICOM files and at least 644 label rows.")
    feats = []
    for i, f in enumerate(files):
        m = extract_all_features(f)
        feats.append({k: float(v) if isinstance(v,(int,float)) and not np.isnan(v) else 0.0
                       for k, v in m.items()})
    keys = sorted(feats[0].keys())
    X = np.array([[d.get(k,0.0) for k in keys] for d in feats], dtype=np.float32)
    # Remove NaN columns
    valid_cols = ~np.isnan(X).any(axis=0)
    X = X[:, valid_cols]; keys = [k for i,k in enumerate(keys) if valid_cols[i]]
    # Remove zero-variance columns
    stds = X.std(axis=0); nonzero = stds > 1e-8
    X = X[:, nonzero]; keys = [k for i,k in enumerate(keys) if nonzero[i]]
    # Remove constant-energy columns (all 6.0)
    noconst = [i for i,k in enumerate(keys) if not (k.endswith('_Energy') and X[:,i].std()<1e-6)]
    X = X[:, noconst]; keys = [keys[i] for i in noconst]

    y_cls = np.array([1 if labels[i%644]['gpr_2mm_2pct'] >= 95 else 0 for i in range(644)])
    scaler = RobustScaler(); X = scaler.fit_transform(X)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, Pass={y_cls.sum()}/{len(y_cls)}")
    return X, y_cls, keys, scaler

# ================================================================
#  Models
# ================================================================
class MLPModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,23),nn.LeakyReLU(),nn.Linear(23,10),nn.LeakyReLU(),nn.Linear(10,2))
    def forward(self, x): return self.net(x)

class PQCModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        from .models.pqc_hybrid import RealAmplitudesCircuit
        self.fc1=nn.Linear(d,23); self.fc2=nn.Linear(23,10); self.act=nn.LeakyReLU()
        self.pqc=RealAmplitudesCircuit(5,1); self.fc_out=nn.Linear(5,2)
    def forward(self,x): x=self.act(self.fc1(x)); x=self.act(self.fc2(x)); return self.fc_out(self.pqc(x))

# ================================================================
#  Train helpers
# ================================================================
def train_torch(model, X_tr, y_tr, X_val, y_val, epochs=300, lr=0.003, patience=40):
    Xt_tr=torch.tensor(X_tr,dtype=torch.float32).to(DEVICE)
    yt_tr=torch.tensor(y_tr,dtype=torch.long).to(DEVICE)
    Xt_val=torch.tensor(X_val,dtype=torch.float32).to(DEVICE)
    yt_val=torch.tensor(y_val,dtype=torch.long).to(DEVICE)
    n_pos=(y_tr==1).sum(); n_neg=(y_tr==0).sum()
    cw=torch.tensor([1.0,float(n_neg)/float(n_pos)],dtype=torch.float32,device=DEVICE)
    crit=nn.CrossEntropyLoss(weight=cw)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=0; wait=0; best_st=None; t0=time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss=crit(model(Xt_tr),yt_tr); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            prob=torch.softmax(model(Xt_val),1)[:,1].cpu().numpy()
            va=roc_auc_score(y_val,prob)
        if va>best: best=va; wait=0; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait>=patience: break
    model.load_state_dict(best_st)
    return time.time()-t0

def evaluate_torch(model, X_te, y_te):
    model.eval()
    Xt=torch.tensor(X_te,dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob=torch.softmax(model(Xt),1)[:,1].cpu().numpy()
        pred=(prob>=0.5).astype(int)
    return pred, prob

def compute_metrics(y, p, prob, name, t):
    acc=accuracy_score(y,p); sens=recall_score(y,p)
    tn,fp,fn,tp=confusion_matrix(y,p).ravel()
    spec=tn/(tn+fp) if (tn+fp)>0 else 0; prec=precision_score(y,p,zero_division=0)
    bal=(sens+spec)/2; a=roc_auc_score(y,prob)
    return {'model':name,'accuracy':acc,'sensitivity':sens,'specificity':spec,
            'precision':prec,'balanced_accuracy':bal,'auc':a,
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'prob':prob,'train_time_s':t}

# ================================================================
#  Main
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, feature_names, scaler = load()
    # Split
    X_trval, X_te, y_trval, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=SEED, stratify=y_trval)
    print(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    results = {}

    # ===== GBDT (grid search + SHAP) =====
    print("\n[1] GBDT + SHAP...")
    best_a=0; best_cfg=None
    for n in [200,300,500]:
        for lr in [0.05,0.1,0.2]:
            for d in [3,5]:
                for s in [0.6,0.7,0.8]:
                    m=GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=d,subsample=s,random_state=SEED)
                    m.fit(X_tr,y_tr); p=m.predict_proba(X_val)[:,1]
                    a=roc_auc_score(y_val,p)
                    if a>best_a: best_a=a; best_cfg=(n,lr,d,s)
    n,lr,d,s=best_cfg
    print(f"  Best: n={n} lr={lr} d={d} sub={s} val_auc={best_a:.4f}")
    gbdt=GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=d,subsample=s,random_state=SEED)
    gbdt.fit(np.vstack([X_tr,X_val]), np.concatenate([y_tr,y_val]))
    t0=time.time(); prob=gbdt.predict_proba(X_te)[:,1]; pred=(prob>=0.5).astype(int); t0=time.time()-t0
    results['GBDT']=compute_metrics(y_te,pred,prob,'GBDT',t0)
    results['GBDT']['model_obj']=gbdt

    # ===== MLP =====
    print("\n[2] MLP (early stopping)...")
    mlp=MLPModel(X.shape[1]).to(DEVICE)
    t0=train_torch(mlp,X_tr,y_tr,X_val,y_val,epochs=300,lr=0.005,patience=40)
    pred,prob=evaluate_torch(mlp,X_te,y_te)
    results['MLP']=compute_metrics(y_te,pred,prob,'MLP',t0)

    # ===== PQC =====
    print("\n[3] PQC-HybridNet (early stopping)...")
    pqc=PQCModel(X.shape[1]).to(DEVICE)
    t0=train_torch(pqc,X_tr,y_tr,X_val,y_val,epochs=300,lr=0.005,patience=50)
    pred,prob=evaluate_torch(pqc,X_te,y_te)
    results['PQC']=compute_metrics(y_te,pred,prob,'PQC-Hybrid',t0)

    # ===== VQC =====
    print("\n[4] QFM-VQCNet...")
    try:
        from pyvqnet.tensor import QTensor; from pyvqnet.nn import CrossEntropyLoss,Softmax,ReLu,Linear
        from pyvqnet.optim import AdamW; from pyvqnet.tensor import no_grad
        from pyvqnet.qnn.vqc import QMachine,VQC_ZZFeatureMap,VQC_HardwareEfficientAnsatz,MeasureAll
        from pyvqnet.nn.module import Module; from pyvqnet.tensor import tensor as vt
        from pyvqnet.utils import set_random_seed; set_random_seed(SEED)
        class VQCModule(Module):
            def __init__(self):
                super().__init__(); self.l1=Linear(X.shape[1],23); self.l2=Linear(23,5)
                self.act=ReLu(); self.qm=QMachine(5)
                self.zm=[MeasureAll(obs={f"Z{i}":1.0}) for i in range(5)]; self.l3=Linear(5,2)
            def forward(self,x):
                x=self.act(self.l1(x)); x=self.act(self.l2(x)); bs=x.shape[0]; self.qm.reset_states(bs)
                VQC_ZZFeatureMap(input_feat=x,q_machine=self.qm,rep=2)
                VQC_HardwareEfficientAnsatz(5,["rx","ry","rz"],"cnot","all",3)(q_machine=self.qm)
                z=vt.concatenate([m(q_machine=self.qm) for m in self.zm],axis=1); return self.l3(z)
        vm=VQCModule()
        Xt_tr=QTensor(np.vstack([X_tr,X_val]).astype(np.float32))
        yt_tr=QTensor(np.concatenate([y_tr,y_val]).astype(np.int64))
        Xt_te=QTensor(X_te.astype(np.float32)); yt_te=QTensor(y_te.astype(np.int64))
        opt=AdamW(vm.parameters(),lr=0.005); lf=CrossEntropyLoss(); sm=Softmax(axis=1)
        vm.train(); bs=48; ns=Xt_tr.shape[0]; nb=ns//bs
        t0=time.time()
        for ep in range(15):
            el=0.0
            for bi in range(nb):
                s,e=bi*bs,min(bi*bs+bs, ns); Xb,yb=Xt_tr[s:e],yt_tr[s:e]
                opt.zero_grad(); o=vm(Xb); l=lf(yb,o); l.backward(); opt.step(); el+=l.item()*Xb.shape[0]
        t0=time.time()-t0
        vm.eval()
        with no_grad():
            prob_v=sm(vm(Xt_te)).numpy()[:,1]; pred_v=vm(Xt_te).argmax(dim=1).numpy()
        results['VQC']=compute_metrics(y_te,pred_v,prob_v,'QFM-VQCNet',t0)
    except Exception as e:
        print(f"  VQC FAILED: {e}")
        results['VQC']={'model':'QFM-VQCNet','accuracy':0,'sensitivity':0,'specificity':0,
                        'precision':0,'balanced_accuracy':0,'auc':0,'train_time_s':0,
                        'tn':0,'fp':0,'fn':0,'tp':0,'prob':np.zeros(len(y_te))}

    # ================================================================
    #  TABLE
    # ================================================================
    print("\n"+"="*80)
    print("  FINAL BENCHMARK (75+ features, all 644 samples)")
    print("="*80)
    hdr=f'  {"Model":<18s} {"Acc":>6s} {"Sens":>6s} {"Prec":>6s} {"Spec":>6s} {"AvgAcc":>6s} {"AUC":>6s}'
    print(hdr); print("  "+"-"*64)
    rows=[]
    for n in ['GBDT','MLP','PQC','VQC']:
        r=results[n]
        print(f'  {r["model"]:<18s} {r["accuracy"]:>6.4f} {r["sensitivity"]:>6.4f} '
              f'{r["precision"]:>6.4f} {r["specificity"]:>6.4f} {r["balanced_accuracy"]:>6.4f} {r["auc"]:>6.4f}')
        rows.append({k:r[k] for k in ['model','accuracy','sensitivity','precision','specificity','balanced_accuracy','auc','train_time_s']})

    # ================================================================
    #  SHAP analysis (GBDT)
    # ================================================================
    print("\n[SHAP] Computing feature importance...")
    explainer = shap.TreeExplainer(results['GBDT']['model_obj'])
    shap_vals = explainer.shap_values(X_te)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]

    # Top 15 features by mean |SHAP|
    mean_shap = np.abs(shap_vals).mean(axis=0)
    top15 = np.argsort(mean_shap)[-15:][::-1]

    print(f"\n  Top 15 Features by |SHAP|:")
    print(f"  {'Feature':<40s} {'|SHAP|':>8s}  {'Type':>8s}")
    print("  "+"-"*60)
    orig_set = {'TotalMU','d-AAV','d-ALG','d-BA','d-CA','d-LSV','d-MAD','d-MCS','d-PM','d-SAS10','d-SAS15','d-SAS5','d-SLG','d-UAA',
                'p-AAV','p-ALG','p-BA','p-CA','p-LSV','p-MAD','p-MCS','p-PM','p-SAS10','p-SAS15','p-SAS5','p-SLG','p-UAA'}
    for i in top15:
        ftype = 'MLC-metric' if feature_names[i] in orig_set else 'NEW-dynamic'
        print(f"  {feature_names[i]:<40s} {mean_shap[i]:>8.4f}  {ftype:>8s}")

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(12,10), dpi=200)
    shap.summary_plot(shap_vals, X_te, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"shap_summary.png"),dpi=200); plt.close()

    # ================================================================
    #  CHARTS
    # ================================================================
    # Bar
    metrics_6=['accuracy','sensitivity','precision','specificity','balanced_accuracy','auc']
    labels_6=['Accuracy','Sensitivity','Precision','Specificity','Avg Accuracy','AUC']
    x=np.arange(6); w=0.2
    fig,ax=plt.subplots(figsize=(14,7),dpi=200)
    colors=['#2196F3','#4CAF50','#FF9800','#E91E63']
    for i,n in enumerate(['GBDT','MLP','PQC','VQC']):
        r=results[n]; vals=[r[m] for m in metrics_6]
        bars=ax.bar(x+i*w,vals,w,label=r['model'],color=colors[i],edgecolor='white')
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+0.01,
                    f'{val:.3f}',ha='center',va='bottom',fontsize=7)
    ax.set_ylabel('Score'); ax.set_xticks(x+w*1.5); ax.set_xticklabels(labels_6,fontsize=11)
    ax.set_ylim(0,1.1); ax.legend(fontsize=10,loc='lower right'); ax.grid(axis='y',alpha=0.3)
    ax.set_title('Final Benchmark — Expanded Features',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"bar_comparison.png"),dpi=200); plt.close()

    # ROC
    fig,ax=plt.subplots(figsize=(8,8),dpi=200)
    for i,n in enumerate(['GBDT','MLP','PQC','VQC']):
        r=results[n]
        if r['auc']>0:
            fpr,tpr,_=roc_curve(y_te,r['prob'])
            ax.plot(fpr,tpr,lw=2,color=colors[i],label=f'{r["model"]} (AUC={r["auc"]:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=.3); ax.legend(fontsize=10); ax.grid(alpha=.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves — Final Benchmark')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roc_curves.png"),dpi=200); plt.close()

    # Confusion
    fig,axes=plt.subplots(2,2,figsize=(12,12),dpi=200)
    for ax_i,n,c in zip(axes.flat,['GBDT','MLP','PQC','VQC'],colors):
        r=results[n]
        cm=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]]); rs=cm.sum(axis=1,keepdims=True); rs[rs==0]=1
        cm_n=cm.astype(float)/rs
        ax_i.imshow(cm_n,cmap=plt.cm.Blues,vmin=0,vmax=1)
        ax_i.set_title(r['model'],fontsize=13,fontweight='bold',color=c)
        for i in range(2):
            for j in range(2):
                ax_i.text(j,i,f"{cm[i,j]}\n({cm_n[i,j]*100:.1f}%)",
                          ha='center',va='center',color='white' if cm_n[i,j]>0.5 else 'black',fontsize=11)
        ax_i.set_xticks([0,1]); ax_i.set_yticks([0,1])
        ax_i.set_xticklabels(['Pred Fail','Pred Pass']); ax_i.set_yticklabels(['True Fail','True Pass'])
    fig.suptitle('Confusion Matrices — Final',fontsize=16,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrices.png"),dpi=200); plt.close()

    # Feature importance bar
    fig,ax=plt.subplots(figsize=(10,8),dpi=200)
    ax.barh(range(15),mean_shap[top15],
            color=['#FF9800' if feature_names[i] not in orig_set else '#2196F3' for i in top15])
    ax.set_yticks(range(15)); ax.set_yticklabels([feature_names[i] for i in top15],fontsize=9)
    ax.set_xlabel('mean(|SHAP|)'); ax.set_title('Feature Importance (SHAP) — Orange=New, Blue=Original')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"feature_importance.png"),dpi=200); plt.close()

    # Excel
    pd.DataFrame(rows).to_excel(os.path.join(OUTPUT_DIR,"final_results.xlsx"),index=False)
    # Save SHAP
    sh_df = pd.DataFrame({'feature':feature_names,'mean_shap':mean_shap})
    sh_df.sort_values('mean_shap',ascending=False).to_excel(os.path.join(OUTPUT_DIR,"shap_importance.xlsx"),index=False)

    print(f"\n  Results -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
