"""Auditable paper-inspired models, independent of the historical implementations.

The publication specifies the optimizer/rate/epochs and the circuits in Figs. 1/2.
Unspecified initialization, batching, optimizer defaults and front-end widths are
recorded below as implementation choices. No labels, scaling or split logic live
in this module. The quantum layers use differentiable, exact statevectors, with
qubit 0 consistently represented by the most significant basis-state bit.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


MODEL_NAMES = ("gbdt", "mlp", "pqc", "qfm")
MODEL_DEFAULTS = {
    "gbdt": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.2,
             "subsample": 0.7, "loss": "log_loss"},
    "mlp": {"hidden_dim": 23, "bottleneck_dim": 10, "epochs": 65,
            "learning_rate": 0.005, "optimizer": "adam", "weight_decay": 0.0,
            "negative_slope": 0.01},
    "pqc": {"hidden_dim": 23, "n_qubits": 5, "epochs": 205,
            "learning_rate": 0.005, "optimizer": "adamw", "weight_decay": 0.01,
            "negative_slope": 0.01},
    "qfm": {"hidden_dim": 23, "n_qubits": 5, "epochs": 12,
            "learning_rate": 0.005, "optimizer": "adamw", "weight_decay": 0.01,
            "feature_reps": 2, "ansatz_reps": 3},
}
IMPLEMENTATION_CHOICES = {
    "batching": "Full batch unless batch_size is explicitly supplied; no samples dropped.",
    "preprocessing": "None in model: publication reports no normalization.",
    "optimizer_defaults": "Adam/AdamW betas=(0.9,0.999), eps=1e-8; AdamW weight_decay=0.01.",
    "classical_initialization": "PyTorch Linear defaults, seeded before construction.",
    "pqc_qfm_hidden_width": "23 reused from the stated classical MLP; hybrid figure width is unspecified.",
    "pqc_readout": "Raw Pauli-Z expectation, not the historical (1-Z)/2 probability mapping.",
    "qfm_mapping": "phi(x)=x; phi(x,y)=(pi-x)*(pi-y): PyVQNet ZZFeatureMap default, not specified in paper.",
    "qfm_initialization": "45 trainable angles sampled uniformly from [0,2*pi).",
    "qfm_ansatz": "Figure 2: 3 rotation-plus-entanglement blocks; no extra terminal rotation block.",
    "leaky_relu": "Negative slope 0.01 (framework default, paper does not state slope).",
    "backend": "Exact PyTorch statevector, analytic Z expectations, no shot noise.",
}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Neural models need the optional 'deep' dependencies: pip install -e '.[deep]'.") from exc
    return torch


def _seed_torch(seed: int, threads: int = 1):
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch


def _single_qubit(state, qubit: int, n_qubits: int, m00, m01, m10, m11):
    """Apply a batched 2x2 gate; qubit 0 is the MSB for every gate/readout."""
    torch = _torch()
    view = state.reshape(state.shape[0], 1 << qubit, 2, 1 << (n_qubits - qubit - 1))
    a0, a1 = view[:, :, 0, :], view[:, :, 1, :]
    def broadcast(value):
        return value[:, None, None] if torch.is_tensor(value) and value.ndim == 1 else value
    values = [broadcast(value) for value in (m00, m01, m10, m11)]
    b0 = values[0] * a0 + values[1] * a1
    b1 = values[2] * a0 + values[3] * a1
    return torch.stack((b0, b1), dim=2).reshape(state.shape)


def _h(state, qubit, n_qubits):
    scale = 1 / math.sqrt(2)
    return _single_qubit(state, qubit, n_qubits, scale, scale, scale, -scale)


def _rotation(state, theta, qubit, n_qubits, axis):
    torch = _torch()
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    if axis == "y":
        return _single_qubit(state, qubit, n_qubits, c, -s, s, c)
    if axis == "x":
        return _single_qubit(state, qubit, n_qubits, c, -1j * s, -1j * s, c)
    if axis == "z":
        return _single_qubit(state, qubit, n_qubits, c - 1j * s, 0, 0, c + 1j * s)
    raise ValueError(f"Unknown rotation axis: {axis}")


def _phase(state, theta, qubit, n_qubits):
    torch = _torch()
    return _single_qubit(state, qubit, n_qubits, 1, 0, 0, torch.exp(1j * theta))


def _cnot(state, control, target, n_qubits):
    torch = _torch()
    index = torch.arange(1 << n_qubits, device=state.device)
    control_set = ((index >> (n_qubits - control - 1)) & 1).bool()
    permutation = torch.where(control_set, index ^ (1 << (n_qubits - target - 1)), index)
    return state[:, permutation]


def _read_z(state, n_qubits):
    torch = _torch()
    probabilities = state.abs().square()
    index = torch.arange(1 << n_qubits, device=state.device)
    signs = torch.stack([1 - 2 * ((index >> (n_qubits - q - 1)) & 1) for q in range(n_qubits)])
    return probabilities @ signs.to(probabilities.dtype).T


def build_pqc_circuit(n_qubits: int = 5):
    """Figure 1: H, data RY, ordered full CNOT cascade, data RY, Z."""
    torch = _torch()
    class PQCCircuit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_qubits = n_qubits

        def statevector(self, x):
            if x.ndim != 2 or x.shape[1] != 2 * self.n_qubits:
                raise ValueError("PQC expects exactly two input angles per qubit.")
            state = x.new_zeros((len(x), 1 << self.n_qubits))
            state[:, 0] = 1
            for q in range(self.n_qubits):
                state = _h(state, q, self.n_qubits)
                state = _rotation(state, x[:, q], q, self.n_qubits, "y")
            for control in range(self.n_qubits - 1):
                for target in range(control + 1, self.n_qubits):
                    state = _cnot(state, control, target, self.n_qubits)
            for q in range(self.n_qubits):
                state = _rotation(state, x[:, self.n_qubits + q], q, self.n_qubits, "y")
            return state

        def forward(self, x):
            return _read_z(self.statevector(x), self.n_qubits)
    return PQCCircuit()


def build_qfm_circuit(n_qubits: int = 5, feature_reps: int = 2, ansatz_reps: int = 3):
    """Figure 2 circuit, with the explicitly documented standard ZZ map choice."""
    torch = _torch()
    class QFMCircuit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_qubits = n_qubits
            self.angles = torch.nn.Parameter(torch.empty(ansatz_reps, n_qubits, 3))
            torch.nn.init.uniform_(self.angles, 0, 2 * math.pi)

        def feature_statevector(self, x):
            if x.ndim != 2 or x.shape[1] != self.n_qubits:
                raise ValueError("QFM expects one input value per qubit.")
            dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
            state = torch.zeros((len(x), 1 << self.n_qubits), dtype=dtype, device=x.device)
            state[:, 0] = 1
            for _ in range(feature_reps):
                for q in range(self.n_qubits):
                    state = _h(state, q, self.n_qubits)
                    state = _phase(state, 2 * x[:, q], q, self.n_qubits)
                for control in range(self.n_qubits - 1):
                    target = control + 1
                    state = _cnot(state, control, target, self.n_qubits)
                    angle = 2 * (math.pi - x[:, control]) * (math.pi - x[:, target])
                    state = _phase(state, angle, target, self.n_qubits)
                    state = _cnot(state, control, target, self.n_qubits)
            return state

        def statevector(self, x):
            state = self.feature_statevector(x)
            for layer in range(ansatz_reps):
                for q in range(self.n_qubits):
                    for axis_index, axis in enumerate(("x", "y", "z")):
                        state = _rotation(state, self.angles[layer, q, axis_index], q, self.n_qubits, axis)
                for control in range(self.n_qubits - 1):
                    for target in range(control + 1, self.n_qubits):
                        state = _cnot(state, control, target, self.n_qubits)
            return state

        def forward(self, x):
            return _read_z(self.statevector(x), self.n_qubits)
    return QFMCircuit()


def build_torch_model(name: str, n_features: int, config: dict | None = None, seed: int = 42):
    """Construct a seeded model with an input width appropriate to this dataset."""
    if name not in ("mlp", "pqc", "qfm"):
        raise ValueError(f"Not a neural model: {name}")
    cfg = {**MODEL_DEFAULTS[name], **(config or {})}
    torch = _seed_torch(seed, cfg.get("num_threads", 1))
    nn = torch.nn
    if name == "mlp":
        return nn.Sequential(nn.Linear(n_features, cfg["hidden_dim"]),
                             nn.LeakyReLU(cfg["negative_slope"]),
                             nn.Linear(cfg["hidden_dim"], cfg["bottleneck_dim"]),
                             nn.LeakyReLU(cfg["negative_slope"]),
                             nn.Linear(cfg["bottleneck_dim"], 2))
    qubits = cfg["n_qubits"]
    act = nn.LeakyReLU(cfg["negative_slope"]) if name == "pqc" else nn.ReLU()
    circuit = (build_pqc_circuit(qubits) if name == "pqc" else
               build_qfm_circuit(qubits, cfg["feature_reps"], cfg["ansatz_reps"]))
    width = 2 * qubits if name == "pqc" else qubits
    return nn.Sequential(nn.Linear(n_features, cfg["hidden_dim"]), act,
                         nn.Linear(cfg["hidden_dim"], width),
                         nn.LeakyReLU(cfg["negative_slope"]) if name == "pqc" else nn.ReLU(),
                         circuit, nn.Linear(qubits, 2))


def _validate_x(x, n_features=None):
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X must be a 2D matrix with at least one feature.")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {values.shape[1]}.")
    if not np.isfinite(values).all():
        raise ValueError("X contains nonfinite values; handle them within the training fold explicitly.")
    return values


@dataclass
class FittedModel:
    name: str
    model: Any
    n_features: int
    seed: int
    config: dict
    history: list
    train_time_s: float

    def predict_proba(self, x) -> np.ndarray:
        """Return P(y=1), with no label remapping or threshold selection."""
        values = _validate_x(x, self.n_features)
        if len(values) == 0:
            return np.empty(0, dtype=np.float64)
        if self.name == "gbdt":
            return self.model.predict_proba(values)[:, 1]
        torch = _torch()
        self.model.eval()
        outputs = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for start in range(0, len(values), 256):
                batch = torch.as_tensor(values[start:start + 256], device=device)
                outputs.append(torch.softmax(self.model(batch), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, x) -> np.ndarray:
        # argmax([p0, p1]) chooses class 0 at an exact tie.
        return (self.predict_proba(x) > 0.5).astype(np.int64)

    def save(self, path) -> None:
        """Save local model artifacts and every effective configuration value."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "name": self.name, "n_features": self.n_features,
                    "seed": self.seed, "config": self.config, "train_time_s": self.train_time_s,
                    "implementation_choices": IMPLEMENTATION_CHOICES}
        if self.name == "gbdt":
            import joblib
            joblib.dump(self.model, path / "model.joblib")
        else:
            torch = _torch()
            state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
            torch.save(state, path / "weights.pt")
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (path / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def train_model(name, X_train, y_train, seed=42, config=None) -> FittedModel:
    """Fit on the supplied training rows only; callers own all folds and evaluation."""
    aliases = {"pqc_hybrid": "pqc", "qfm_vqc": "qfm"}
    name = aliases.get(name, name)
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}.")
    cfg = {**MODEL_DEFAULTS[name], "device": "cpu", "batch_size": None,
           "shuffle": False, "num_threads": 1, **(config or {})}
    x = _validate_x(X_train)
    y = np.asarray(y_train)
    if y.ndim != 1 or len(y) != len(x) or set(np.unique(y)) != {0, 1}:
        raise ValueError("y_train must contain both binary classes 0/1 and match X_train rows.")
    y = y.astype(np.int64)
    started = time.perf_counter()
    history = []
    if name == "gbdt":
        params = {key: cfg[key] for key in MODEL_DEFAULTS[name]}
        model = GradientBoostingClassifier(**params, random_state=seed)
        model.fit(x, y)
        history = [{"iteration": i + 1, "train_deviance": float(loss)}
                   for i, loss in enumerate(model.train_score_)]
    else:
        torch = _torch()
        if int(cfg["epochs"]) < 1:
            raise ValueError("epochs must be positive.")
        batch_size = len(x) if cfg["batch_size"] is None else int(cfg["batch_size"])
        if batch_size < 1:
            raise ValueError("batch_size must be positive or None for full batch.")
        model = build_torch_model(name, x.shape[1], cfg, seed).to(cfg["device"])
        tensor_x = torch.as_tensor(x, device=cfg["device"])
        tensor_y = torch.as_tensor(y, dtype=torch.long, device=cfg["device"])
        optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg["optimizer"]]
        optimizer = optimizer_class(model.parameters(), lr=cfg["learning_rate"],
                                    weight_decay=cfg["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.default_rng(seed)
        for epoch in range(int(cfg["epochs"])):
            model.train()
            order = rng.permutation(len(x)) if cfg["shuffle"] else np.arange(len(x))
            total_loss = 0.0
            seen = 0
            for start in range(0, len(x), batch_size):
                indices = torch.as_tensor(order[start:start + batch_size], device=cfg["device"])
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tensor_x[indices]), tensor_y[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name}: nonfinite loss at epoch {epoch + 1}.")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"{name}: nonfinite gradient at epoch {epoch + 1}.")
                optimizer.step()
                count = len(indices)
                total_loss += float(loss.detach()) * count
                seen += count
            history.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                            "samples_seen": seen})
    return FittedModel(name, model, x.shape[1], int(seed), cfg, history, time.perf_counter() - started)


def load_model(path, device="cpu") -> FittedModel:
    """Reload an artifact created by save; only load trusted local joblib files."""
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != 1:
        raise ValueError("Unsupported reproduction model format.")
    cfg = {**meta["config"], "device": device}
    if meta["name"] == "gbdt":
        import joblib
        model = joblib.load(path / "model.joblib")
    else:
        torch = _torch()
        model = build_torch_model(meta["name"], meta["n_features"], cfg, meta["seed"]).to(device)
        model.load_state_dict(torch.load(path / "weights.pt", map_location=device, weights_only=True))
        model.eval()
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    return FittedModel(meta["name"], model, meta["n_features"], meta["seed"], cfg,
                       history, meta["train_time_s"])
