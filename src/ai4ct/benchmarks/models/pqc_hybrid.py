# ai4ct/benchmarks/models/pqc_hybrid.py
"""
PQC-HybridNet — Parameterized Quantum Circuit + Classical MLP

Architecture:
    Input → FC(hidden1) → FC(hidden2) → RealAmplitudesCircuit → FC(2)

The quantum circuit is implemented in **pure PyTorch** (no TensorCircuit
or TensorFlow dependency).  It simulates:

    |0...0⟩ → H⊗ⁿ → RY(θ₁) → CNOT cascade → RY(θ₂) → ⟨Z⟩

using explicit state-vector manipulation for up to ~5 qubits (dim ≤ 32).
"""
from __future__ import annotations
import time, random, math
import numpy as np
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..config import SEED, PQCHybridConfig


# ================================================================
#  Pure-PyTorch Real Amplitudes Quantum Circuit
# ================================================================

class RealAmplitudesCircuit(nn.Module):
    """RealAmplitudes ansatz as a pure-PyTorch nn.Module.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (circuit dimension = 2**num_qubits).
    num_layers : int
        Number of CNOT-entanglement repetitions.
    """

    def __init__(self, num_qubits: int, num_layers: int = 1):
        super().__init__()
        self.n = num_qubits
        self.num_layers = num_layers
        self.dim = 1 << num_qubits   # 2**n
        # Precompute H⊗ⁿ once (small: ≤ 32×32)
        self.register_buffer(
            "_Hmat",
            self._build_hadamard(num_qubits),
        )

    @staticmethod
    def _build_hadamard(n: int) -> torch.Tensor:
        H = torch.tensor([[1., 1.], [1., -1.]]) / math.sqrt(2.)
        for _ in range(1, n):
            H = torch.kron(
                torch.tensor([[1., 1.], [1., -1.]]) / math.sqrt(2.), H)
        return H.to(torch.complex64)

    # ------------------------------------------------------------------
    #  Gate helpers
    # ------------------------------------------------------------------

    def _hadamard_all(self, state: torch.Tensor) -> torch.Tensor:
        """Apply H⊗ⁿ using the precomputed matrix buffer."""
        H = self._Hmat.to(device=state.device)
        return state @ H.T

    @staticmethod
    def _ry(state: torch.Tensor, theta: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply RY(theta_q) to a single qubit.  theta: (B,)."""
        B, D = state.shape
        n = int(math.log2(D))
        # reshape to expose qubit:  (B, 2^q, 2, 2^{n-q-1})
        state_r = state.reshape(B, 1 << qubit, 2, 1 << (n - qubit - 1))

        ch = torch.cos(theta / 2.0)       # (B,)
        sh = torch.sin(theta / 2.0)       # (B,)
        # expand for broadcasting
        ch = ch[:, None, None]            # (B, 1, 1)
        sh = sh[:, None, None]

        a0 = state_r[:, :, 0, :]          # amplitude of |0⟩ on qubit q
        a1 = state_r[:, :, 1, :]          # amplitude of |1⟩

        new0 = ch * a0 - sh * a1
        new1 = sh * a0 + ch * a1
        return torch.stack([new0, new1], dim=2).reshape(B, D)

    @staticmethod
    def _cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """CNOT: flip target iff control == 1.  Implemented as index permutation."""
        D = state.shape[1]
        idx = torch.arange(D, device=state.device)
        c_mask = (idx >> control) & 1          # 1 where control qubit is 1
        t_flip = idx ^ (1 << target)           # flip target bit
        perm = torch.where(c_mask.bool(), t_flip, idx)
        return state[:, perm]

    @staticmethod
    def _expectation_z(state: torch.Tensor, qubit: int) -> torch.Tensor:
        """⟨Z_q⟩ = P(q=0) - P(q=1) = 1 - 2 * P(q=1)."""
        probs = state.real ** 2 + state.imag ** 2   # (B, D)
        D = state.shape[1]
        mask = torch.tensor(
            [(j >> qubit) & 1 for j in range(D)],
            dtype=torch.float32, device=state.device,
        )
        p1 = (probs * mask).sum(dim=1)             # (B,)
        return 1.0 - 2.0 * p1

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2*n)  — first n columns = θ₁, last n columns = θ₂

        Returns
        -------
        (B, n) — mapped outputs in [0, 1]
        """
        B = x.shape[0]
        # |0...0⟩ initial state
        state = torch.zeros(B, self.dim, dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j

        # H⊗ⁿ
        state = self._hadamard_all(state)

        # RY layer 1
        for q in range(self.n):
            state = self._ry(state, x[:, q], q)

        # CNOT cascade
        for _ in range(self.num_layers):
            for c in range(self.n - 1):
                for t in range(c + 1, self.n):
                    state = self._cnot(state, c, t)

        # RY layer 2
        for q in range(self.n):
            state = self._ry(state, x[:, self.n + q], q)

        # ⟨Z⟩ measurements → map [-1,1] to [0,1]
        results = []
        for q in range(self.n):
            ez = self._expectation_z(state, q)    # (B,) in [-1, 1]
            results.append((1.0 - ez) / 2.0)       # → [0, 1]
        return torch.stack(results, dim=1)         # (B, n)


# ================================================================
#  Hybrid Model
# ================================================================

class PQC_HybridNet(nn.Module):
    """Classical front-end + quantum circuit + linear output."""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int,
                 output_dim: int, num_layer: int,
                 activation: str = "leaky_relu"):
        super().__init__()
        act_fn = {"leaky_relu": nn.LeakyReLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act = act_fn()

        # hidden_dim2 must be even — each pair of dims maps to one qubit
        num_qubits = hidden_dim2 // 2
        self.pqc = RealAmplitudesCircuit(num_qubits=num_qubits, num_layers=num_layer)
        self.fc_out = nn.Linear(num_qubits, output_dim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.pqc(x)
        return self.fc_out(x)


# ================================================================
#  Trainer
# ================================================================

class PQCHybridModel:
    """PQC-HybridNet trainer (pure PyTorch backend)."""

    def __init__(self, cfg: PQCHybridConfig = None, device: str = "cpu"):
        self.cfg = cfg or PQCHybridConfig()
        self.device = torch.device(device)
        self._set_seed()

        self.model = PQC_HybridNet(
            input_dim=self.cfg.input_dim,
            hidden_dim1=self.cfg.hidden_dim1,
            hidden_dim2=self.cfg.hidden_dim2,
            output_dim=self.cfg.output_dim,
            num_layer=self.cfg.num_layer,
            activation=self.cfg.activation,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        opt_cls = optim.AdamW if self.cfg.optimizer == "adamw" else optim.Adam
        self.optimizer = opt_cls(self.model.parameters(), lr=self.cfg.lr)

        self._train_acc: List[float] = []
        self._test_acc: List[float] = []
        self._train_loss: List[float] = []
        self._test_loss: List[float] = []

    def _set_seed(self):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        Xt_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Xt_te = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        yt_tr = torch.tensor(y_train, dtype=torch.long).to(self.device)
        yt_te = torch.tensor(y_test, dtype=torch.long).to(self.device)

        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(Xt_tr)
            loss = self.criterion(outputs, yt_tr)
            loss.backward()
            self.optimizer.step()
            self._train_loss.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                tr_preds = self.model(Xt_tr).argmax(dim=1).cpu().numpy()
                self._train_acc.append(accuracy_score(y_train, tr_preds))

                te_out = self.model(Xt_te)
                self._test_loss.append(self.criterion(te_out, yt_te).item())
                te_preds = te_out.argmax(dim=1).cpu().numpy()
                self._test_acc.append(accuracy_score(y_test, te_preds))

            if (epoch + 1) % 20 == 0:
                print(f"  PQC  Epoch {epoch+1:>3d}/{self.cfg.epochs}  "
                      f"loss={loss.item():.4f}  train_acc={self._train_acc[-1]:.4f}  "
                      f"test_acc={self._test_acc[-1]:.4f}")

        train_time = time.time() - t0
        return {
            "train_time": train_time,
            "train_acc_list": self._train_acc,
            "test_acc_list": self._test_acc,
            "train_loss_list": self._train_loss,
            "test_loss_list": self._test_loss,
            "epochs": self.cfg.epochs,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(Xt).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(Xt), dim=1)[:, 1].cpu().numpy()

    @property
    def train_accuracies(self): return self._train_acc
    @property
    def test_accuracies(self):  return self._test_acc
    @property
    def train_losses(self):     return self._train_loss
    @property
    def test_losses(self):      return self._test_loss
