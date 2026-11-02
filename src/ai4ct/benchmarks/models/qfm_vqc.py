# ai4ct/benchmarks/models/qfm_vqc.py
"""
QFM-VQCNet — Quantum Fuzzy Model with VQC

Uses PyVQNet (a quantum machine learning framework) to build a
hybrid quantum-classical network:

    Input(33) → Linear(23) → ReLU → Linear(5) → ReLU
    → VQC(5 qubits, ZZFeatureMap + HardwareEfficientAnsatz)
    → Linear(2)

The VQC uses ZZFeatureMap for quantum encoding and a
HardwareEfficientAnsatz as the trainable variational circuit.
"""
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import pyvqnet
from pyvqnet.nn.module import Module
from pyvqnet.optim.adam import Adam
from pyvqnet.optim import AdamW
from pyvqnet.qnn.vqc import (
    QMachine, VQC_HardwareEfficientAnsatz,
    VQC_ZZFeatureMap, MeasureAll,
)
from pyvqnet.tensor import tensor, QTensor
from pyvqnet.nn import ReLu, Linear, CrossEntropyLoss, Softmax
from pyvqnet.tensor import no_grad
from pyvqnet.utils import set_random_seed

from ..config import SEED, QFMVQCConfig


# ============================================================
#  Quantum Circuit
# ============================================================

class VariationQuantumCircuit(Module):
    """VQC with ZZFeatureMap encoding + HardwareEfficientAnsatz."""

    def __init__(self, num_qubits: int = 5, reps: int = 3):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.qm = QMachine(self.num_qubits)

        self.feature_map = VQC_ZZFeatureMap
        self.ansatz = VQC_HardwareEfficientAnsatz(
            self.num_qubits,
            ["rx", "ry", "rz"],
            "cnot",
            "all",
            self.reps,
        )
        self.z_measures = [
            MeasureAll(obs={f"Z{i}": 1.0}) for i in range(self.num_qubits)
        ]

    def forward(self, x):
        batch_size = x.shape[0]
        self.qm.reset_states(batch_size)
        self.feature_map(input_feat=x, q_machine=self.qm, rep=2)
        self.ansatz(q_machine=self.qm)
        z_list = [m(q_machine=self.qm) for m in self.z_measures]
        return tensor.concatenate(z_list, axis=1)


# ============================================================
#  Full Hybrid Model
# ============================================================

class QFM_VQCNet(Module):
    """QFM-VQCNet: Classical → Quantum → Classical."""

    def __init__(self, input_dim: int, hidden_dim1: int = 23,
                 hidden_dim2: int = 5, output_dim: int = 2,
                 num_qubits: int = 5, reps: int = 3):
        super().__init__()
        self.linear1 = Linear(input_dim, hidden_dim1)
        self.linear2 = Linear(hidden_dim1, hidden_dim2)
        self.act = ReLu()
        self.vqc = VariationQuantumCircuit(num_qubits=num_qubits, reps=reps)
        self.linear3 = Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.vqc(x)
        x = self.linear3(x)
        return x


# ============================================================
#  Data loader (PyVQNet uses QTensor)
# ============================================================

def _load_qvqc_data(file_path: str):
    """Load data and convert to PyVQNet QTensor format.
    Uses config-defined feature/label column ranges so it works with
    any Excel layout, not just the original 38-column format."""
    from ..config import FEATURE_COLS, LABEL_COL
    if file_path is None or not Path(file_path).is_file():
        raise FileNotFoundError(
            f"Feature table not found: {file_path}. Restore data/processed or pass --data."
        )
    data = pd.read_excel(file_path)
    X = data.iloc[:, FEATURE_COLS[0]:FEATURE_COLS[1]].values.astype(np.float32)
    y = data.iloc[:, LABEL_COL].values.astype(np.int64)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return (
        QTensor(X_train), QTensor(X_test),
        QTensor(y_train), QTensor(y_test),
    )


# ============================================================
#  Training function
# ============================================================

def train_qfm_vqc(
    model: QFM_VQCNet,
    X_train, X_test, y_train, y_test,
    epochs: int = 12,
    batch_size: int = 48,
    lr: float = 0.005,
    print_every: int = 10,
    use_adamw: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], float]:
    """Train QFM-VQCNet and return diagnostic history + training time.

    Returns
    -------
    train_accs, test_accs, train_losses, test_losses,
    train_time
    """
    opt_cls = AdamW if use_adamw else Adam
    optimizer = opt_cls(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    softmax = Softmax(axis=1)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            Xb, yb = X_train[start:end], y_train[start:end]

            optimizer.zero_grad()
            output = model(Xb)
            loss = loss_fn(yb, output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * Xb.shape[0]
            pred = output.argmax(1)
            epoch_correct += (pred == yb).sum().item()

        avg_train_loss = epoch_loss / n_samples
        train_acc = epoch_correct / n_samples

        model.eval()
        with no_grad():
            test_output = model(X_test)
            test_loss = loss_fn(y_test, test_output).item()
            test_probs = softmax(test_output)
            test_preds = test_probs.argmax(1).numpy()
            y_true = y_test.numpy().astype(int)
            test_acc = np.mean(test_preds == y_true)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"  VQC  Epoch {epoch+1:>2d}/{epochs}  "
              f"train_loss={avg_train_loss:.4f}  test_loss={test_loss:.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    train_time = time.time() - t0
    return train_accs, test_accs, train_losses, test_losses, train_time


# ============================================================
#  Evaluation
# ============================================================

def evaluate_qfm_vqc(model, X_test, y_test):
    """Return (test_accuracy, precision, recall, auc, ap, TN, FP, FN, TP)."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    model.eval()
    with no_grad():
        outputs = model(X_test)
        probs = Softmax(axis=1)(outputs).numpy()[:, 1]
        preds = outputs.argmax(dim=1).numpy()
        y_true = y_test.numpy().astype(int)

    test_acc = accuracy_score(y_true, preds)
    test_prec = precision_score(y_true, preds, zero_division=0)
    test_rec = recall_score(y_true, preds, zero_division=0)
    test_auc = roc_auc_score(y_true, probs)
    test_ap = average_precision_score(y_true, probs)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return test_acc, test_prec, test_rec, test_auc, test_ap, int(tn), int(fp), int(fn), int(tp)


# ============================================================
#  High-level runner (matches the original notebook's flow)
# ============================================================

def run_qfm_vqc(config: QFMVQCConfig = None, data_path: str = None):
    """Full training + evaluation of QFM-VQCNet.  Returns metrics dict."""
    cfg = config or QFMVQCConfig()
    X_train, X_test, y_train, y_test = _load_qvqc_data(data_path)

    model = QFM_VQCNet(
        input_dim=cfg.input_dim,
        hidden_dim1=cfg.hidden_dim1,
        hidden_dim2=cfg.hidden_dim2,
        output_dim=cfg.output_dim,
        num_qubits=cfg.num_qubits,
        reps=cfg.reps,
    )

    print(">>> Training QFM-VQCNet ...")
    train_accs, test_accs, train_losses, test_losses, train_time = train_qfm_vqc(
        model, X_train, X_test, y_train, y_test,
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        use_adamw=(cfg.optimizer == "adamw"),
    )

    t1 = time.time()
    test_acc, test_prec, test_rec, test_auc, test_ap, tn, fp, fn, tp = \
        evaluate_qfm_vqc(model, X_test, y_test)
    inference_time = time.time() - t1

    return {
        "train_score": train_accs[-1],
        "test_score": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "auc": test_auc,
        "ap": test_ap,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "train_time_s": train_time,
        "inference_time_s": inference_time,
        "train_acc_list": train_accs,
        "test_acc_list": test_accs,
        "train_loss_list": train_losses,
        "test_loss_list": test_losses,
        "epochs": cfg.epochs,
    }
