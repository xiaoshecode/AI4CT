# ai4ct/benchmarks/models/mlp.py
"""
Classical MLP — Multi-Layer Perceptron

A 3-layer fully-connected network (33→23→10→2) with LeakyReLU
activation, trained with CrossEntropyLoss and Adam.
"""
import time
import random
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..config import SEED, MLPConfig


def _get_activation(name: str) -> nn.Module:
    return {
        "leaky_relu": nn.LeakyReLU,
        "relu":       nn.ReLU,
        "gelu":       nn.GELU,
    }[name]()


class MLP(nn.Module):
    """3-layer fully-connected network."""

    def __init__(self, input_dim: int, hidden_dim1: int,
                 hidden_dim2: int, output_dim: int, activation: str = "leaky_relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.act = _get_activation(activation)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class MLPModel:
    """Classical MLP trainer wrapper."""

    def __init__(self, cfg: MLPConfig = None, device: str = "cpu"):
        self.cfg = cfg or MLPConfig()
        self.device = torch.device(device)
        self._set_seed()

        self.model = MLP(
            input_dim=self.cfg.input_dim,
            hidden_dim1=self.cfg.hidden_dim1,
            hidden_dim2=self.cfg.hidden_dim2,
            output_dim=self.cfg.output_dim,
            activation=self.cfg.activation,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

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

    def _get_optimizer(self):
        if self.cfg.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        return optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train the MLP and record per-epoch diagnostics."""
        Xt_tr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Xt_te = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        yt_tr = torch.tensor(y_train, dtype=torch.long).to(self.device)
        yt_te = torch.tensor(y_test, dtype=torch.long).to(self.device)

        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            # ---- train step ----
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(Xt_tr)
            loss = self.criterion(outputs, yt_tr)
            loss.backward()
            self.optimizer.step()
            self._train_loss.append(loss.item())

            # ---- eval ----
            self.model.eval()
            with torch.no_grad():
                train_preds = self.model(Xt_tr).argmax(dim=1).cpu().numpy()
                self._train_acc.append(accuracy_score(y_train, train_preds))

                test_out = self.model(Xt_te)
                self._test_loss.append(self.criterion(test_out, yt_te).item())
                test_preds = test_out.argmax(dim=1).cpu().numpy()
                self._test_acc.append(accuracy_score(y_test, test_preds))

            if (epoch + 1) % 10 == 0:
                print(f"  MLP  Epoch {epoch+1:>3d}/{self.cfg.epochs}  "
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
