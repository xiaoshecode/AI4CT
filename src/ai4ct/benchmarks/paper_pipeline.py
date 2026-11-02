#!/usr/bin/env python
# ai4ct/benchmarks/paper_pipeline.py
"""
Full paper-spec pipeline: DICOM → 30+ metrics → ArcCheck GPR labels
→ 4 model training (paper parameters) → evaluation.

Matches: "Quantum-Classical Hybrid Networks for PSQA Prediction"
"""
from __future__ import annotations
import os, json, time, math, random, warnings, pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, auc,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project imports
from ai4ct.features.metrics_33 import extract_33_metrics
from ai4ct.paths import DICOM_DIR, ARCCHECK_PATH, PAPER_OUTPUT

warnings.filterwarnings("ignore")

# ================================================================
#  Constants
# ================================================================
SEED = 42
OUTPUT_DIR = PAPER_OUTPUT
GPR_THRESHOLD = 95.0          # TG-218: 95% gamma pass rate
TEST_SIZE = 0.2
CONTAMINATION = 0.1            # Isolation Forest outlier fraction

# Paper-spec model parameters (33 features when available)
INPUT_DIM = None               # set dynamically after metric extraction


# ================================================================
#  1. Data Loading & Labeling
# ================================================================

def load_arccheck_labels(excel_path: str) -> pd.DataFrame:
    """Read ArcCheck Excel and return DataFrame with clean GPR labels."""
    df = pd.read_excel(excel_path)
    # Rename for clarity
    df.columns = ['row_idx','name','pid','plan_type','physicist','diagnosis',
                  'stage','site','plan_result','date','gpr_2_2','gpr_3_3']

    def _parse_gpr(val):
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace(',', '.').replace('%', ''))

    df['GPR'] = df['gpr_2_2'].apply(_parse_gpr)
    df['Label'] = (df['GPR'] >= GPR_THRESHOLD).astype(int)  # 1=Pass, 0=Fail
    return df


def build_dataset(dicom_dir: str, arccheck_df: pd.DataFrame,
                  max_files: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Match DICOM files with ArcCheck rows by position, extract metrics.

    Row i (0-indexed) of ArcCheck → RP_pID{i+1:03d}.dcm

    Returns
    -------
    X : (N, D) feature matrix
    y : (N,)  labels (1=Pass, 0=Fail)
    feature_names : list of str
    """
    import glob
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "RP_pID*.dcm")))
    if max_files:
        dcm_files = dcm_files[:max_files]

    all_features = []
    labels = []
    failed = []

    for i, dcm_path in enumerate(dcm_files):
        try:
            metrics = extract_33_metrics(dcm_path)
        except Exception as e:
            failed.append((dcm_path, str(e)))
            continue

        # Match to ArcCheck row (position-based)
        if i >= len(arccheck_df):
            continue
        label = arccheck_df.iloc[i]['Label']

        # Drop non-numeric keys
        pid = metrics.pop('patient_id', str(i))
        feat_dict = {k: v for k, v in metrics.items()
                     if isinstance(v, (int, float)) and not np.isnan(v)}

        all_features.append(feat_dict)
        labels.append(label)

    if not all_features:
        raise RuntimeError("No valid DICOM-ArcCheck pairs found!")

    if failed:
        print(f"  [{len(failed)} files skipped: {failed[:3]}...]")

    # Align feature keys across all samples
    all_keys = sorted(all_features[0].keys())
    X = np.array([[feat.get(k, 0.0) for k in all_keys] for feat in all_features],
                 dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Drop duplicate (PI=CA) and zero-variance (CLS, SAS2) features
    drop_keys = {'d-PI','p-PI','d-CLS','p-CLS','d-SAS2','p-SAS2'}
    keep_idx = [i for i, k in enumerate(all_keys) if k not in drop_keys]
    selected_keys = [all_keys[i] for i in keep_idx]
    X = X[:, keep_idx]

    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features (dropped {len(drop_keys)} redundant)")
    print(f"  Pass (1): {(y==1).sum()}  |  Fail (0): {(y==0).sum()}")
    return X, y, selected_keys


def remove_outliers_isolation_forest(X: np.ndarray, y: np.ndarray,
                                     contamination: float = 0.1
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outlier samples using Isolation Forest (paper method)."""
    iso = IsolationForest(contamination=contamination, random_state=SEED, n_jobs=-1)
    outlier_pred = iso.fit_predict(X)
    keep = outlier_pred == 1   # 1 = inlier, -1 = outlier
    print(f"  IsolationForest removed {(~keep).sum()}/{len(X)} outliers "
          f"({contamination*100:.0f}% contamination)")
    return X[keep], y[keep]


# ================================================================
#  2. Paper-Spec Models
# ================================================================

# --- 2a. Classical MLP ---
class PaperMLP(nn.Module):
    """33 → 23 → 10 → 2  (NO activation on output, paper spec)"""
    def __init__(self, input_dim=33):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 23)
        self.fc2 = nn.Linear(23, 10)
        self.fc3 = nn.Linear(10, 2)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)          # raw logits


# --- 2b. PQC-HybridNet (pure PyTorch quantum circuit) ---
class RealAmplitudesCircuit(nn.Module):
    """5-qubit RealAmplitudes PQC: H→RY1→CNOT→RY2→⟨Z⟩."""
    def __init__(self, num_qubits=5, num_layers=1):
        super().__init__()
        self.n = num_qubits
        self.dim = 1 << num_qubits
        self.num_layers = num_layers
        H1 = torch.tensor([[1.,1.],[1.,-1.]]) / math.sqrt(2.)
        Hn = H1
        for _ in range(1, num_qubits):
            Hn = torch.kron(
                torch.tensor([[1.,1.],[1.,-1.]]) / math.sqrt(2.), Hn)
        self.register_buffer('_Hmat', Hn.to(torch.complex64))

    def _hadamard_all(self, state):
        return state @ self._Hmat.to(state.device).T

    @staticmethod
    def _ry(state, theta, qubit):
        B, D = state.shape
        n = int(math.log2(D))
        sr = state.reshape(B, 1 << qubit, 2, 1 << (n - qubit - 1))
        ch, sh = torch.cos(theta/2)[:,None,None], torch.sin(theta/2)[:,None,None]
        a0, a1 = sr[:,:,0,:], sr[:,:,1,:]
        return torch.stack([ch*a0 - sh*a1, sh*a0 + ch*a1], dim=2).reshape(B, D)

    @staticmethod
    def _cnot(state, control, target):
        D = state.shape[1]
        idx = torch.arange(D, device=state.device)
        c_mask = (idx >> control) & 1
        return state[:, torch.where(c_mask.bool(), idx ^ (1 << target), idx)]

    @staticmethod
    def _expectation_z(state, qubit):
        probs = state.real**2 + state.imag**2
        D = state.shape[1]
        mask = torch.tensor([(j>>qubit)&1 for j in range(D)], dtype=torch.float32, device=state.device)
        return 1.0 - 2.0 * (probs * mask).sum(dim=1)

    def forward(self, x):
        B = x.shape[0]
        state = torch.zeros(B, self.dim, dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j
        state = self._hadamard_all(state)
        for q in range(self.n):
            state = self._ry(state, x[:, q], q)
        for _ in range(self.num_layers):
            for c in range(self.n - 1):
                for t in range(c + 1, self.n):
                    state = self._cnot(state, c, t)
        for q in range(self.n):
            state = self._ry(state, x[:, self.n + q], q)
        return torch.stack([(1.0 - self._expectation_z(state, q))/2.0
                            for q in range(self.n)], dim=1)


class PaperPQC(nn.Module):
    """33 → 23 → 10 → PQC(5q) → 2"""
    def __init__(self, input_dim=33):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 23)
        self.fc2 = nn.Linear(23, 10)
        self.act = nn.LeakyReLU()
        self.pqc = RealAmplitudesCircuit(num_qubits=5, num_layers=1)
        self.fc_out = nn.Linear(5, 2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.pqc(x)
        return self.fc_out(x)


# --- 2c. QFM-VQCNet ---
class PaperVQC(nn.Module):
    """QFM-VQCNet with PyVQNet backend."""
    def __init__(self, input_dim=33):
        super().__init__()
        import pyvqnet
        from pyvqnet.nn.module import Module as VQCModule
        from pyvqnet.qnn.vqc import QMachine, VQC_ZZFeatureMap, VQC_HardwareEfficientAnsatz, MeasureAll
        from pyvqnet.nn import ReLu as VQCReLu, Linear as VQCLinear
        from pyvqnet.tensor import tensor as VQCtensor

        self._vqc = VQCModule
        self._VQCLinear = VQCLinear
        self._VQCReLu = VQCReLu
        self._QMachine = QMachine
        self._ZZFeatureMap = VQC_ZZFeatureMap
        self._HEAnsatz = VQC_HardwareEfficientAnsatz
        self._MeasureAll = MeasureAll
        self._tensor = VQCtensor
        self.input_dim = input_dim

    def build(self):
        """Build the PyVQNet VQC model — classical→quantum→classical."""
        class VQCModule(self._vqc):
            def __init__(self2):
                super().__init__()
                self2.linear1 = self._VQCLinear(self.input_dim, 23)
                self2.linear2 = self._VQCLinear(23, 5)
                self2.act = self._VQCReLu()
                self2.qm = self._QMachine(5)
                self2.z_measures = [self._MeasureAll(obs={f"Z{i}": 1.0}) for i in range(5)]
                self2.linear3 = self._VQCLinear(5, 2)

            def forward(self2, x):
                # Classical front-end
                x = self2.act(self2.linear1(x))
                x = self2.act(self2.linear2(x))
                # Quantum feature map + ansatz
                bs = x.shape[0]
                self2.qm.reset_states(bs)
                self._ZZFeatureMap(input_feat=x, q_machine=self2.qm, rep=2)
                self._HEAnsatz(5, ["rx","ry","rz"], "cnot", "all", 3)(q_machine=self2.qm)
                z = self._tensor.concatenate(
                    [m(q_machine=self2.qm) for m in self2.z_measures], axis=1)
                return self2.linear3(z)
        return VQCModule()


# ================================================================
#  3. Training & Evaluation
# ================================================================

def train_mlp(model, X_tr, y_tr, X_te, y_te, epochs=200, lr=0.005, device="cpu"):
    """Paper-spec MLP training with class weighting + cosine annealing."""
    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt_tr = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xt_te = torch.tensor(X_te, dtype=torch.float32).to(device)
    yt_te = torch.tensor(y_te, dtype=torch.long).to(device)
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    class_w = torch.tensor([1.0, float(n_neg)/float(n_pos)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {'train_acc':[], 'test_acc':[], 'train_loss':[], 'test_loss':[]}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = criterion(model(Xt_tr), yt_tr); loss.backward(); opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xt_tr).argmax(1) == yt_tr).float().mean().item()
            te_out = model(Xt_te)
            te_loss = criterion(te_out, yt_te).item()
            te_acc = (te_out.argmax(1) == yt_te).float().mean().item()
        hist['train_acc'].append(tr_acc); hist['test_acc'].append(te_acc)
        hist['train_loss'].append(loss.item()); hist['test_loss'].append(te_loss)
        if (ep+1) % 40 == 0:
            print(f"  MLP  E{ep+1:>3d}/{epochs} loss={loss.item():.4f} tr_acc={tr_acc:.4f} te_acc={te_acc:.4f}")
    train_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xt_te), 1)[:,1].cpu().numpy()
        pred = model(Xt_te).argmax(1).cpu().numpy()
    return pred, prob, hist, train_time


def train_pqc(model, X_tr, y_tr, X_te, y_te, epochs=400, lr=0.003, device="cpu"):
    """Paper-spec PQC-HybridNet training with class weighting + cosine annealing."""
    Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt_tr = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xt_te = torch.tensor(X_te, dtype=torch.float32).to(device)
    yt_te = torch.tensor(y_te, dtype=torch.long).to(device)
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    class_w = torch.tensor([1.0, float(n_neg)/float(n_pos)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {'train_acc':[], 'test_acc':[], 'train_loss':[], 'test_loss':[]}
    t0 = time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = criterion(model(Xt_tr), yt_tr); loss.backward(); opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xt_tr).argmax(1) == yt_tr).float().mean().item()
            te_out = model(Xt_te)
            te_loss = criterion(te_out, yt_te).item()
            te_acc = (te_out.argmax(1) == yt_te).float().mean().item()
        hist['train_acc'].append(tr_acc); hist['test_acc'].append(te_acc)
        hist['train_loss'].append(loss.item()); hist['test_loss'].append(te_loss)
        if (ep+1) % 40 == 0:
            print(f"  PQC  E{ep+1:>3d}/{epochs} loss={loss.item():.4f} tr_acc={tr_acc:.4f} te_acc={te_acc:.4f}")
    train_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xt_te), 1)[:,1].cpu().numpy()
        pred = model(Xt_te).argmax(1).cpu().numpy()
    return pred, prob, hist, train_time


def evaluate(name: str, y_true, y_pred, y_prob, train_time: float) -> Dict:
    """Compute all 6 paper-spec metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)       # sensitivity = recall
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens + spec) / 2.0             # balanced accuracy
    auc_val = roc_auc_score(y_true, y_prob)

    print(f"\n  [{name}]")
    print(f"    Accuracy:      {acc:.4f}")
    print(f"    Sensitivity:   {sens:.4f}")
    print(f"    Specificity:   {spec:.4f}")
    print(f"    Precision:     {prec:.4f}")
    print(f"    Balanced Acc:  {bal_acc:.4f}")
    print(f"    AUC:           {auc_val:.4f}")
    print(f"    Train Time:    {train_time:.1f}s")
    print(f"    CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    return {
        "model": name,
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "precision": prec, "balanced_accuracy": bal_acc, "auc": auc_val,
        "train_time_s": train_time, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ================================================================
#  4. Main Pipeline
# ================================================================

def main():
    print("=" * 65)
    print("  PAPER-SPEC PSQA PREDICTION PIPELINE")
    print("=" * 65)

    # --- Load data ---
    print("\n[1] Loading ArcCheck ground truth...")
    arccheck = load_arccheck_labels(ARCCHECK_PATH)
    print(f"    {len(arccheck)} rows, Pass≥95%: {(arccheck['Label']==1).sum()}")

    print("\n[2] Extracting plan complexity metrics from DICOM...")
    X_full, y_full, feature_names = build_dataset(DICOM_DIR, arccheck)
    global INPUT_DIM
    INPUT_DIM = X_full.shape[1]

    # --- Skip outlier removal (use all 644 samples) ---
    print("\n[3] Skipping outlier removal (use all samples)...")
    X_clean, y_clean = X_full, y_full

    # --- Train/test split ---
    print("\n[4] Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=TEST_SIZE, random_state=SEED, stratify=y_clean)
    print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"    Features: {INPUT_DIM} (full list at end)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Model 1: GBDT (sklearn, paper spec) ---
    # Compute class weight (inverse frequency)
    n_pos = (y_train == 1).sum(); n_neg = (y_train == 0).sum()
    pos_weight = n_neg / n_pos  # weight for minority class (Pass)

    print("\n[5a] Training Classical-GBDT...")
    t0 = time.time()
    gbdt = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.2, max_depth=3,
        subsample=0.7, random_state=SEED)
    gbdt.fit(X_train, y_train, sample_weight=np.where(y_train==1, pos_weight, 1.0))
    gbdt_time = time.time() - t0
    gbdt_pred = gbdt.predict(X_test)
    gbdt_prob = gbdt.predict_proba(X_test)[:, 1]
    gbdt_res = evaluate("Classical-GBDT", y_test, gbdt_pred, gbdt_prob, gbdt_time)

    # --- Model 2: Classical-MLP ---
    print("\n[5b] Training Classical-MLP...")
    mlp_model = PaperMLP(input_dim=INPUT_DIM)
    mlp_pred, mlp_prob, mlp_hist, mlp_time = train_mlp(
        mlp_model, X_train, y_train, X_test, y_test, epochs=200, lr=0.005)
    mlp_res = evaluate("Classical-MLP", y_test, mlp_pred, mlp_prob, mlp_time)

    # --- Model 3: PQC-HybridNet ---
    print("\n[5c] Training PQC-HybridNet...")
    pqc_model = PaperPQC(input_dim=INPUT_DIM)
    pqc_pred, pqc_prob, pqc_hist, pqc_time = train_pqc(
        pqc_model, X_train, y_train, X_test, y_test, epochs=400, lr=0.003)
    pqc_res = evaluate("PQC-HybridNet", y_test, pqc_pred, pqc_prob, pqc_time)

    # --- Model 4: QFM-VQCNet ---
    print("\n[5d] Training QFM-VQCNet (PyVQNet)...")
    try:
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import CrossEntropyLoss, Softmax
        from pyvqnet.optim import Adam
        from pyvqnet.tensor import no_grad
        from pyvqnet.utils import set_random_seed

        set_random_seed(SEED)
        vqc_wrapper = PaperVQC(input_dim=INPUT_DIM)
        vqc_model = vqc_wrapper.build()

        Xt_tr = QTensor(X_train.astype(np.float32))
        yt_tr = QTensor(y_train.astype(np.int64))
        Xt_te = QTensor(X_test.astype(np.float32))
        yt_te = QTensor(y_test.astype(np.int64))

        opt = Adam(vqc_model.parameters(), lr=0.005)
        loss_fn = CrossEntropyLoss()
        softmax = Softmax(axis=1)

        t0 = time.time()
        vqc_model.train()
        n_samples = Xt_tr.shape[0]; bs = 48; n_batches = n_samples // bs
        for ep in range(12):
            ep_loss = 0.0
            for bi in range(n_batches):
                s, e = bi*bs, min(bi*bs+bs, n_samples)
                Xb, yb = Xt_tr[s:e], yt_tr[s:e]
                opt.zero_grad()
                out = vqc_model(Xb); l = loss_fn(yb, out); l.backward(); opt.step()
                ep_loss += l.item() * Xb.shape[0]
            if (ep+1) % 3 == 0:
                print(f"  VQC  E{ep+1:>2d}/12  loss={ep_loss/n_samples:.4f}")
        vqc_time = time.time() - t0

        vqc_model.eval()
        with no_grad():
            vqc_out = vqc_model(Xt_te)
            vqc_prob = softmax(vqc_out).numpy()[:, 1]
            vqc_pred = vqc_out.argmax(dim=1).numpy()
        vqc_res = evaluate("QFM-VQCNet", y_test, vqc_pred, vqc_prob, vqc_time)
    except Exception as e:
        import traceback
        print(f"  QFM-VQCNet FAILED: {e}")
        traceback.print_exc()
        vqc_res = {"model": "QFM-VQCNet", "accuracy": 0, "sensitivity": 0,
                   "specificity": 0, "precision": 0, "balanced_accuracy": 0,
                   "auc": 0, "train_time_s": 0, "tn": 0, "fp": 0, "fn": 0, "tp": 0}

    # --- Comparison ---
    all_res = [gbdt_res, mlp_res, pqc_res, vqc_res]
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f'  {"Model":<18s} {"Acc":>6s} {"Sens":>6s} {"Spec":>6s} {"Prec":>6s} {"BalAcc":>6s} {"AUC":>6s} {"Time":>7s}')
    print("  " + "-" * 64)
    for r in all_res:
        print(f'  {r["model"]:<18s} {r["accuracy"]:>6.4f} {r["sensitivity"]:>6.4f} '
              f'{r["specificity"]:>6.4f} {r["precision"]:>6.4f} '
              f'{r["balanced_accuracy"]:>6.4f} {r["auc"]:>6.4f} {r["train_time_s"]:>6.1f}s')

    # --- ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    colors = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    roc_data = [
        ("GBDT", gbdt_prob), ("MLP", mlp_prob),
        ("PQC", pqc_prob), ("VQC", vqc_prob if vqc_res["auc"] > 0 else None),
    ]
    for (name, prob), c in zip(roc_data, colors):
        if prob is None: continue
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={auc(fpr,tpr):.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — Paper-Spec Models"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"roc_curves.png"),dpi=200); plt.close()

    # --- Bar chart: all 6 metrics ---
    metrics_6 = ['accuracy','sensitivity','precision','specificity','balanced_accuracy','auc']
    labels_6 = ['Accuracy','Sensitivity','Precision','Specificity','Avg Accuracy','AUC']
    x = np.arange(len(labels_6)); w = 0.2
    fig, ax = plt.subplots(figsize=(14,7), dpi=200)
    colors_bar = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    for i, r in enumerate(all_res):
        vals = [r[m] for m in metrics_6]
        bars = ax.bar(x + i*w, vals, w, label=r['model'], color=colors_bar[i], edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Score'); ax.set_xticks(x + w*1.5); ax.set_xticklabels(labels_6, fontsize=11)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=10, loc='lower right'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Benchmark Comparison — 33 Metrics, 644 Samples', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"bar_comparison.png"),dpi=200); plt.close()

    # --- Confusion matrix grid ---
    fig, axes = plt.subplots(2, 2, figsize=(12,12), dpi=200)
    for ax_i, (r, c) in zip(axes.flat, zip(all_res, colors_bar)):
        cm = np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        rs = cm.sum(axis=1, keepdims=True); rs[rs==0]=1
        cm_n = cm.astype(float)/rs
        ax_i.imshow(cm_n, cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax_i.set_title(f"{r['model']}", fontsize=13, fontweight='bold', color=c)
        for i in range(2):
            for j in range(2):
                ax_i.text(j,i,f"{cm[i,j]}\n({cm_n[i,j]*100:.1f}%)",
                          ha='center',va='center',
                          color='white' if cm_n[i,j]>0.5 else 'black',fontsize=11)
        ax_i.set_xticks([0,1]); ax_i.set_yticks([0,1])
        ax_i.set_xticklabels(['Pred Fail','Pred Pass']); ax_i.set_yticklabels(['True Fail','True Pass'])
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrices.png"),dpi=200); plt.close()

    # --- Save results ---
    pd.DataFrame(all_res).to_excel(os.path.join(OUTPUT_DIR, "paper_results.xlsx"), index=False)
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump({"n_features": INPUT_DIM, "names": feature_names}, f, indent=2, ensure_ascii=False)

    print(f"\n  Feature list ({INPUT_DIM}): {feature_names}")
    print(f"  Results → {OUTPUT_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
