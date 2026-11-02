#!/usr/bin/env python
# ai4ct/benchmarks/run_threshold_90.py
"""
Benchmark with negative class threshold changed to gpr_2mm_2pct < 90%.
(Original threshold was 95%, now changed to 90%)

Features: 27 plan complexity metrics from DICOM (33 original - 6 redundant)
Models: Classical-GBDT, Classical-MLP, PQC-HybridNet, QFM-VQCNet
"""
import os, json, time, warnings, glob, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
)

from ai4ct.features.metrics_33 import extract_33_metrics
from ai4ct.paths import DICOM_DIR, LABEL_PATH, BENCHMARK_OUTPUT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================================================================
#  Constants
# ================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = BENCHMARK_OUTPUT / "threshold_90"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- THRESHOLD CHANGED: 95% → 90% -----
GPR_THRESHOLD = 90.0  # gpr_2mm_2pct >= 90 → Pass (1), < 90 → Fail (0)

# Redundant / zero-variance features to drop (same as original paper)
DROP_KEYS = {'d-PI', 'p-PI', 'd-CLS', 'p-CLS', 'd-SAS2', 'p-SAS2'}


# ================================================================
#  1. Data Loading
# ================================================================

def load_all():
    """Load DICOM features + ArcCheck labels, return X, y, feature_names."""
    from sklearn.preprocessing import RobustScaler

    with open(LABEL_PATH, encoding='utf-8') as f:
        labels = json.load(f)

    files = sorted(glob.glob(os.path.join(DICOM_DIR, "RP_pID*.dcm")))[:644]
    if not files:
        raise FileNotFoundError(f"No RP_pID*.dcm files found in {DICOM_DIR}; check AI4CT_DATA_DIR.")
    if len(labels) < len(files):
        raise ValueError("Fewer label rows than DICOM files; verify the cohort mapping before training.")

    X_list, y_list = [], []
    for i, f in enumerate(files):
        m = extract_33_metrics(f)
        m.pop('patient_id', None)

        # Keep only non-dropped keys
        keys = sorted(k for k in m if k not in DROP_KEYS)
        X_list.append([float(m[k]) for k in keys])

        # Label: gpr_2mm_2pct >= threshold → Pass (1), else Fail (0)
        label = 1 if labels[i]['gpr_2mm_2pct'] >= GPR_THRESHOLD else 0
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Apply RobustScaler normalization (same as ai4ct/benchmarks/data.py)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    n_pass = y.sum()
    n_fail = len(y) - n_pass
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Threshold gpr_2mm_2pct >= {GPR_THRESHOLD}%: Pass={n_pass}, Fail={n_fail}")
    return X, y, keys


# ================================================================
#  2. Models
# ================================================================

class PaperMLP(nn.Module):
    """33→23→10→2 with LeakyReLU (paper architecture)."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 23), nn.LeakyReLU(),
            nn.Linear(23, 10), nn.LeakyReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.net(x)


class PaperPQC(nn.Module):
    """33→23→10→PQC(5q)→2 Hybrid Quantum-Classical (paper architecture)."""
    def __init__(self, input_dim):
        super().__init__()
        from .models.pqc_hybrid import RealAmplitudesCircuit
        self.fc1 = nn.Linear(input_dim, 23)
        self.fc2 = nn.Linear(23, 10)
        self.act = nn.LeakyReLU()
        self.pqc = RealAmplitudesCircuit(num_qubits=5, num_layers=1)
        self.fc_out = nn.Linear(5, 2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc_out(self.pqc(x))


# ================================================================
#  3. Training
# ================================================================

def train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=200, lr=0.005):
    """MLP training on combined train+val, cosine annealing, no class weighting (60/40 ≈ balanced)."""
    Xt_tr = torch.tensor(np.vstack([X_tr, X_val]), dtype=torch.float32).to(DEVICE)
    yt_tr = torch.tensor(np.concatenate([y_tr, y_val]), dtype=torch.long).to(DEVICE)

    # No class weighting at 90% threshold — class balance is ~60/40
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {'train_acc': [], 'train_loss': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt_tr)
        loss = criterion(out, yt_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        acc = (out.argmax(1) == yt_tr).float().mean().item()
        hist['train_acc'].append(acc)
        hist['train_loss'].append(loss.item())

        if (ep + 1) % 50 == 0:
            print(f"    MLP E{ep+1:>3d}/{epochs} loss={loss.item():.4f} acc={acc:.4f}")

    model.eval()
    return time.time() - t0


def train_pqc(model, X_tr, y_tr, X_val, y_val, epochs=400, lr=0.003):
    """PQC training on combined train+val, cosine annealing, no class weighting."""
    Xt_tr = torch.tensor(np.vstack([X_tr, X_val]), dtype=torch.float32).to(DEVICE)
    yt_tr = torch.tensor(np.concatenate([y_tr, y_val]), dtype=torch.long).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt_tr)
        loss = criterion(out, yt_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        if (ep + 1) % 50 == 0:
            acc = (out.argmax(1) == yt_tr).float().mean().item()
            print(f"    PQC E{ep+1:>3d}/{epochs} loss={loss.item():.4f} acc={acc:.4f}")

    model.eval()
    return time.time() - t0


def evaluate_model(model, X_te, y_te, name):
    """Evaluate model and return metrics dict."""
    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(Xt), 1)[:, 1].cpu().numpy()
        pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_te, pred)
    sens = recall_score(y_te, pred)  # sensitivity = recall
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_te, pred, zero_division=0)
    bal = (sens + spec) / 2.0  # balanced accuracy = average accuracy
    auc_v = roc_auc_score(y_te, prob)

    return {
        'model': name,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'balanced_accuracy': bal,
        'auc': auc_v,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'prob': prob,
        'pred': pred,
    }


# ================================================================
#  4. Main Pipeline
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"  BENCHMARK: Negative Class = gpr_2mm_2pct < {GPR_THRESHOLD}%")
    print("=" * 70)

    # --- Load data ---
    print("\n[1] Loading DICOM features + ArcCheck labels...")
    X, y, feature_names = load_all()

    # --- Train/val/test split (60/20/20) ---
    print("\n[2] Train/Val/Test split (60/20/20)...")
    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, random_state=SEED, stratify=y_trval
    )
    print(f"    Train: {len(y_tr)}  |  Val: {len(y_val)}  |  Test: {len(y_te)}")
    print(f"    Train Pass/Fail: {(y_tr==1).sum()}/{(y_tr==0).sum()}")
    print(f"    Test  Pass/Fail: {(y_te==1).sum()}/{(y_te==0).sum()}")

    results = {}

    # ================================================================
    #  Model 1: Classical-GBDT (with grid search)
    # ================================================================
    print("\n[3a] Classical-GBDT — grid search...")
    best_auc = 0
    best_cfg = None
    for n_est in [100, 200, 300, 500]:
        for lr in [0.05, 0.1, 0.2]:
            for d in [3, 5, 7]:
                for sub in [0.6, 0.7, 0.8]:
                    gbdt = GradientBoostingClassifier(
                        n_estimators=n_est, learning_rate=lr, max_depth=d,
                        subsample=sub, random_state=SEED
                    )
                    gbdt.fit(X_tr, y_tr)
                    prob = gbdt.predict_proba(X_val)[:, 1]
                    auc_v = roc_auc_score(y_val, prob)
                    if auc_v > best_auc:
                        best_auc = auc_v
                        best_cfg = (n_est, lr, d, sub)

    n_est, lr, d, sub = best_cfg
    print(f"    Best: n_est={n_est}, lr={lr}, depth={d}, sub={sub} (val_auc={best_auc:.4f})")

    gbdt = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr, max_depth=d,
        subsample=sub, random_state=SEED
    )
    gbdt.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    t0 = time.time()
    prob = gbdt.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    gbdt_time = time.time() - t0

    results['Classical-GBDT'] = compute_metrics_dict(
        y_te, pred, prob, 'Classical-GBDT', gbdt_time
    )

    # ================================================================
    #  Model 2: Classical-MLP (paper-spec training)
    # ================================================================
    print("\n[3b] Classical-MLP — training (paper-spec, no early stopping)...")
    mlp = PaperMLP(input_dim=X.shape[1]).to(DEVICE)
    t0 = train_mlp(mlp, X_tr, y_tr, X_val, y_val, epochs=200, lr=0.005)
    results['Classical-MLP'] = evaluate_model(mlp, X_te, y_te, 'Classical-MLP')
    results['Classical-MLP']['train_time_s'] = t0

    # ================================================================
    #  Model 3: PQC-HybridNet (paper-spec training)
    # ================================================================
    print("\n[3c] PQC-HybridNet — training (paper-spec, no early stopping)...")
    pqc = PaperPQC(input_dim=X.shape[1]).to(DEVICE)
    t0 = train_pqc(pqc, X_tr, y_tr, X_val, y_val, epochs=400, lr=0.003)
    results['PQC-HybridNet'] = evaluate_model(pqc, X_te, y_te, 'PQC-HybridNet')
    results['PQC-HybridNet']['train_time_s'] = t0

    # ================================================================
    #  Model 4: QFM-VQCNet (PyVQNet backend)
    # ================================================================
    print("\n[3d] QFM-VQCNet — training (extended epochs)...")
    try:
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import CrossEntropyLoss, Softmax, ReLu, Linear
        from pyvqnet.optim import AdamW
        from pyvqnet.tensor import no_grad
        from pyvqnet.qnn.vqc import (
            QMachine, VQC_ZZFeatureMap, VQC_HardwareEfficientAnsatz, MeasureAll
        )
        from pyvqnet.nn.module import Module
        from pyvqnet.tensor import tensor as vqc_tensor
        from pyvqnet.utils import set_random_seed

        set_random_seed(SEED)

        class VQCModule(Module):
            def __init__(self):
                super().__init__()
                self.l1 = Linear(X.shape[1], 23)
                self.l2 = Linear(23, 5)
                self.act = ReLu()
                self.qm = QMachine(5)
                self.zm = [MeasureAll(obs={f"Z{i}": 1.0}) for i in range(5)]
                self.l3 = Linear(5, 2)

            def forward(self, x):
                x = self.act(self.l1(x))
                x = self.act(self.l2(x))
                bs = x.shape[0]
                self.qm.reset_states(bs)
                VQC_ZZFeatureMap(input_feat=x, q_machine=self.qm, rep=2)
                VQC_HardwareEfficientAnsatz(5, ["rx", "ry", "rz"], "cnot", "all", 3)(
                    q_machine=self.qm
                )
                z = vqc_tensor.concatenate(
                    [m(q_machine=self.qm) for m in self.zm], axis=1
                )
                return self.l3(z)

        vqc_mod = VQCModule()
        # Oversample minority class (Fail=0) to balance training data
        X_trv = np.vstack([X_tr, X_val])
        y_trv = np.concatenate([y_tr, y_val])
        pass_idx = np.where(y_trv == 1)[0]
        fail_idx = np.where(y_trv == 0)[0]
        n_pass = len(pass_idx)
        n_fail = len(fail_idx)
        if n_pass > n_fail:
            # Oversample fails
            oversampled_fail = np.random.choice(fail_idx, size=n_pass - n_fail, replace=True)
            bal_idx = np.concatenate([np.arange(len(y_trv)), oversampled_fail])
        else:
            oversampled_pass = np.random.choice(pass_idx, size=n_fail - n_pass, replace=True)
            bal_idx = np.concatenate([np.arange(len(y_trv)), oversampled_pass])
        np.random.shuffle(bal_idx)
        X_bal = X_trv[bal_idx]
        y_bal = y_trv[bal_idx]
        print(f"    Balanced: {len(y_bal)} samples (Pass={(y_bal==1).sum()}, Fail={(y_bal==0).sum()})")

        Xt_trv = QTensor(X_bal.astype(np.float32))
        yt_trv = QTensor(y_bal.astype(np.int64))
        Xt_te = QTensor(X_te.astype(np.float32))

        opt = AdamW(vqc_mod.parameters(), lr=0.005)
        loss_fn = CrossEntropyLoss()
        sm = Softmax(axis=1)

        vqc_mod.train()
        bs = 48
        n_s = Xt_trv.shape[0]
        nb = n_s // bs

        t0 = time.time()
        for ep in range(30):
            el = 0.0
            for bi in range(nb):
                s, e = bi * bs, min(bi * bs + bs, n_s)
                Xb, yb = Xt_trv[s:e], yt_trv[s:e]
                opt.zero_grad()
                o = vqc_mod(Xb)
                l = loss_fn(yb, o)
                l.backward()
                opt.step()
                el += l.item() * Xb.shape[0]
            if (ep + 1) % 10 == 0:
                print(f"    VQC E{ep+1:>2d}/30 loss={el/n_s:.4f}")
        vqc_time = time.time() - t0

        vqc_mod.eval()
        with no_grad():
            prob_v = sm(vqc_mod(Xt_te)).numpy()[:, 1]
            pred_v = vqc_mod(Xt_te).argmax(dim=1).numpy()

        results['QFM-VQCNet'] = compute_metrics_dict(
            y_te, pred_v, prob_v, 'QFM-VQCNet', vqc_time
        )

    except Exception as e:
        import traceback
        print(f"    QFM-VQCNet FAILED: {e}")
        traceback.print_exc()
        results['QFM-VQCNet'] = {
            'model': 'QFM-VQCNet',
            'accuracy': 0, 'sensitivity': 0, 'specificity': 0,
            'precision': 0, 'balanced_accuracy': 0, 'auc': 0,
            'train_time_s': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
            'prob': np.zeros(len(y_te)), 'pred': np.zeros(len(y_te)),
        }

    # ================================================================
    #  5. Results Table
    # ================================================================
    print("\n")
    print("=" * 90)
    print(f"  RESULTS — Negative Class: gpr_2mm_2pct < {GPR_THRESHOLD}%")
    print("=" * 90)

    # Print formatted table
    hdr = (
        f"  {'Models':<18s} {'Accuracy':>9s} {'Sensitivity':>12s} "
        f"{'Precision':>10s} {'Specificity':>12s} "
        f"{'Avg Accuracy':>13s} {'AUC':>7s}"
    )
    print(hdr)
    print("  " + "-" * 85)

    ordered_names = ['Classical-MLP', 'Classical-GBDT', 'PQC-HybridNet', 'QFM-VQCNet']
    table_rows = []

    for name in ordered_names:
        r = results.get(name, results.get('QFM-VQCNet'))
        if r['accuracy'] > 0:
            print(
                f"  {r['model']:<18s} "
                f"{r['accuracy']*100:>7.1f}%   "
                f"{r['sensitivity']*100:>10.1f}%   "
                f"{r['precision']*100:>8.1f}%   "
                f"{r['specificity']*100:>10.1f}%    "
                f"{r['balanced_accuracy']*100:>11.1f}%   "
                f"{r['auc']:>6.3f}"
            )
        else:
            print(
                f"  {r['model']:<18s} "
                f"{'FAILED':>9s}   {'-':>10s}   {'-':>8s}   "
                f"{'-':>10s}    {'-':>11s}   {'-':>6s}"
            )

        table_rows.append({
            'Model': r['model'],
            'Accuracy': f"{r['accuracy']*100:.1f}%",
            'Sensitivity': f"{r['sensitivity']*100:.1f}%",
            'Precision': f"{r['precision']*100:.1f}%",
            'Specificity': f"{r['specificity']*100:.1f}%",
            'Avg Accuracy': f"{r['balanced_accuracy']*100:.1f}%",
            'AUC': f"{r['auc']:.3f}",
        })

    print("  " + "-" * 85)

    # Also print a nicer markdown-style table
    print("\n  --- Markdown Table ---")
    print(f"  | Models | Accuracy | Sensitivity | Precision | Specificity | Avg Accuracy | AUC |")
    print(f"  |---|---|---|---|---|---|---|")
    for row in table_rows:
        print(
            f"  | {row['Model']} | {row['Accuracy']} | {row['Sensitivity']} | "
            f"{row['Precision']} | {row['Specificity']} | {row['Avg Accuracy']} | "
            f"{row['AUC']} |"
        )

    # ================================================================
    #  6. Confusion Matrix Details
    # ================================================================
    print("\n  --- Confusion Matrix Details ---")
    for name in ordered_names:
        r = results.get(name, {})
        if r.get('accuracy', 0) > 0:
            print(
                f"  {r['model']:<18s}  "
                f"TN={r['tn']:<4d} FP={r['fp']:<4d} "
                f"FN={r['fn']:<4d} TP={r['tp']:<4d}"
            )

    # ================================================================
    #  7. Save results
    # ================================================================
    output_xlsx = os.path.join(OUTPUT_DIR, "threshold_90_results.xlsx")
    pd.DataFrame(table_rows).to_excel(output_xlsx, index=False)
    print(f"\n  Results saved to: {output_xlsx}")

    # Also save a detailed JSON
    output_json = os.path.join(OUTPUT_DIR, "threshold_90_results.json")
    json_results = {}
    for name in ordered_names:
        r = results.get(name, {})
        json_results[name] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in r.items()
            if k not in ('prob', 'pred')
        }
    with open(output_json, 'w') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  Detailed results saved to: {output_json}")

    print("\n  Done!")
    return results


def compute_metrics_dict(y_te, pred, prob, name, train_time):
    """Helper to compute metrics for non-torch models (GBDT, VQC)."""
    acc = accuracy_score(y_te, pred)
    sens = recall_score(y_te, pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_te, pred, zero_division=0)
    bal = (sens + spec) / 2.0
    auc_v = roc_auc_score(y_te, prob)
    return {
        'model': name,
        'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
        'precision': prec, 'balanced_accuracy': bal, 'auc': auc_v,
        'train_time_s': train_time,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'prob': prob, 'pred': pred,
    }


if __name__ == "__main__":
    main()
