# ai4ct/benchmarks/models/gbdt.py
"""
GBDT — Gradient Boosted Decision Trees

Uses sklearn's GradientBoostingClassifier for radiotherapy plan
quality classification.  Tracks per-tree accuracy and log-loss
for diagnostic curves.
"""
import time
import random
import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from ..config import SEED, GBDTConfig


class GBDTModel:
    """Gradient Boosting Decision Tree classifier wrapper."""

    def __init__(self, cfg: GBDTConfig = None):
        self.cfg = cfg or GBDTConfig()
        self.model = GradientBoostingClassifier(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            random_state=SEED,
            subsample=self.cfg.subsample,
        )
        self._train_acc = []
        self._test_acc = []
        self._train_loss = []
        self._test_loss = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train the model and record per-tree diagnostics.

        Returns
        -------
        dict with keys: train_time, train_acc_list, test_acc_list,
                        train_loss_list, test_loss_list, n_trees
        """
        t0 = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - t0

        # ---- per-tree accuracy ----
        for train_pred, test_pred in zip(
            self.model.staged_predict(X_train),
            self.model.staged_predict(X_test),
        ):
            self._train_acc.append(accuracy_score(y_train, train_pred))
            self._test_acc.append(accuracy_score(y_test, test_pred))

        # ---- per-tree log-loss ----
        for train_proba, test_proba in zip(
            self.model.staged_predict_proba(X_train),
            self.model.staged_predict_proba(X_test),
        ):
            self._train_loss.append(log_loss(y_train, train_proba))
            self._test_loss.append(log_loss(y_test, test_proba))

        return {
            "train_time": train_time,
            "train_acc_list": self._train_acc,
            "test_acc_list": self._test_acc,
            "train_loss_list": self._train_loss,
            "test_loss_list": self._test_loss,
            "n_trees": len(self._train_acc),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def train_accuracies(self) -> list:
        return self._train_acc

    @property
    def test_accuracies(self) -> list:
        return self._test_acc

    @property
    def train_losses(self) -> list:
        return self._train_loss

    @property
    def test_losses(self) -> list:
        return self._test_loss
