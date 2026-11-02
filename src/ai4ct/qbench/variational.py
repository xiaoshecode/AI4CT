"""PennyLane/Torch classifiers for the frozen qbench architecture catalogue.

Inputs are raw, finite 17-column features. Preprocessors are fitted only on the
rows supplied to ``fit_variational``. The caller owns grouping, labels and IF:
this module never splits, filters or reorders samples. Optional dependencies are
loaded when a model is constructed, not when this module is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from types import SimpleNamespace
import hashlib
import json
import math
import os
from pathlib import Path
from .atomic import replace_with_retry
import random
import time
from typing import Any

import numpy as np


VARIATIONAL_IDS = (
    "vqc_pca5_angle_5q", "reupload_full17_5q", "reupload_full17_1q",
    "dressed_vqc_5q", "residual_vqc_5q", "ttn_pca8_8q",
    "qcnn_pca8_8q", "amplitude_vqc_5q",
    "iqp_vqc_5q", "separable_vqc_5q", "mps_pca8_8q",
    "gibbs_qbm_3q", "gibbs_separable_3q", "dressed_classical_5d",
    "dressed_separable_5q", "mera_pca8_8q",
    "anchor_paper_pqc_5q", "anchor_paper_qfm_5q",
)
ANCHOR_IDS = VARIATIONAL_IDS[-2:]
FORMAT_VERSION = 1


def _dependencies():
    try:
        import torch
        import pennylane as qml
    except ImportError as exc:
        raise ImportError("qbench variational models require PennyLane and PyTorch in the dedicated qbench environment") from exc
    return torch, qml


def _validate_x(x):
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 17:
        raise ValueError("Expected a two-dimensional raw 17-feature matrix")
    if not np.isfinite(value).all():
        raise ValueError("Features contain nonfinite values; no samples were silently discarded")
    return value


def _resolved_config(model_id, supplied=None):
    if model_id not in VARIATIONAL_IDS:
        raise ValueError(f"Unknown variational model: {model_id}")
    anchor = model_id in ANCHOR_IDS
    epochs = 205 if model_id == ANCHOR_IDS[0] else 12 if anchor else 30
    cfg = dict(epochs=epochs, lr=.005, dtype="float32" if anchor else "float64",
               threads=1, device="cpu", optimizer="adamw", weight_decay=.01,
               betas=[.9, .999], eps=1e-8, batch_size=None, shuffle=False,
               clip_grad_norm=None, max_wall_seconds=None, resume_from=None)
    supplied = dict(supplied or {})
    for old, new in (("learning_rate", "lr"), ("num_threads", "threads")):
        if old in supplied:
            if new in supplied and supplied[new] != supplied[old]:
                raise ValueError(f"Conflicting {old}/{new}")
            supplied[new] = supplied.pop(old)
    unknown = set(supplied) - set(cfg)
    if unknown:
        raise ValueError(f"Unrecognized variational configuration keys: {sorted(unknown)}")
    cfg.update(supplied)
    if cfg["device"] != "cpu":
        raise ValueError("This validated reference adapter supports device='cpu' only")
    if cfg["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not isinstance(cfg["epochs"], int) or cfg["epochs"] < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(cfg["threads"], int) or cfg["threads"] < 1:
        raise ValueError("threads must be a positive integer")
    if cfg["optimizer"] not in ("adam", "adamw") or not np.isfinite(cfg["lr"]) or cfg["lr"] <= 0 or not np.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("Invalid optimizer, learning rate or weight decay")
    if cfg["batch_size"] is not None and (not isinstance(cfg["batch_size"], int) or cfg["batch_size"] < 1):
        raise ValueError("batch_size must be None or a positive integer")
    if len(cfg["betas"]) != 2 or not all(0 <= v < 1 for v in cfg["betas"]) or cfg["eps"] <= 0:
        raise ValueError("Invalid Adam betas or epsilon")
    for key in ("clip_grad_norm", "max_wall_seconds"):
        if cfg[key] is not None and (not np.isfinite(cfg[key]) or cfg[key] <= 0):
            raise ValueError(f"{key} must be positive or None")
    if cfg["resume_from"] is not None:
        cfg["resume_from"] = str(cfg["resume_from"])
    return cfg


@dataclass
class FeatureTransform:
    """Portable numerical preprocessing state; no pickled estimator needed."""
    model_id: str
    state: dict

    @classmethod
    def fit(cls, model_id, x):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        values = _validate_x(x)
        if not len(values):
            raise ValueError("Cannot fit a transform on no rows")
        state = {"fit_rows": len(values), "kind": "raw", "zero_amplitude_training_rows": 0}
        if model_id in ANCHOR_IDS:
            return cls(model_id, state)
        scaler = StandardScaler().fit(values)
        state.update(kind="standard", mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                     variance=scaler.var_.tolist(), constant_columns=np.flatnonzero(scaler.var_ == 0).tolist())
        z = scaler.transform(values)
        k = 5 if model_id in ("vqc_pca5_angle_5q", "iqp_vqc_5q", "separable_vqc_5q") else 8 if model_id in ("ttn_pca8_8q", "qcnn_pca8_8q", "mps_pca8_8q", "mera_pca8_8q") else None
        if k:
            if len(values) < k:
                raise ValueError(f"PCA{k} requires at least {k} training rows")
            pca = PCA(n_components=k, whiten=False, svd_solver="full").fit(z)
            projected = pca.transform(z)
            std = projected.std(axis=0, ddof=0)
            degenerate = std < 1e-12
            std[degenerate] = 1
            state.update(kind="pca_angle", components=pca.components_.tolist(),
                         pca_mean=pca.mean_.tolist(), pc_scale=std.tolist(),
                         explained_variance=pca.explained_variance_.tolist(),
                         explained_variance_ratio=np.nan_to_num(pca.explained_variance_ratio_).tolist(),
                         degenerate_components=np.flatnonzero(degenerate).tolist())
        elif model_id.startswith("reupload_"):
            state["kind"] = "angle"
        elif model_id == "amplitude_vqc_5q":
            state["kind"] = "amplitude"
            state["zero_amplitude_training_rows"] = int(np.count_nonzero(np.linalg.norm(z, axis=1) == 0))
        return cls(model_id, state)

    def transform(self, x):
        values = _validate_x(x)
        s = self.state
        if s["kind"] == "raw":
            return values.copy()
        z = (values - np.asarray(s["mean"])) / np.asarray(s["scale"])
        if s["kind"] == "pca_angle":
            z = ((z - np.asarray(s["pca_mean"])) @ np.asarray(s["components"]).T) / np.asarray(s["pc_scale"])
            return 2 * np.arctan(z)
        if s["kind"] == "angle":
            return 2 * np.arctan(z)
        if s["kind"] == "amplitude":
            padded = np.zeros((len(z), 32), dtype=np.float64)
            padded[:, :17] = z
            norm = np.linalg.norm(padded, axis=1)
            zero = norm == 0
            padded[zero, 0] = 1
            norm[zero] = 1
            return padded / norm[:, None]
        return z

    def to_dict(self):
        return {"model_id": self.model_id, "state": self.state}

    @classmethod
    def from_dict(cls, value):
        return cls(value["model_id"], value["state"])


def _apply_circuit(qml, model_id, inputs, weights):
    """Explicit catalogue gates, using last-axis indexing for data broadcasts."""
    def ry_input(q, index=None):
        qml.RY(inputs[..., q if index is None else index], wires=q)

    def rot_layer(w, n=5):
        for q in range(n):
            qml.Rot(*w[q], wires=q)
        if model_id not in ("separable_vqc_5q", "dressed_separable_5q"):
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])

    def full_cnot():
        for c in range(4):
            for t in range(c + 1, 5):
                qml.CNOT(wires=[c, t])

    if model_id == "anchor_paper_pqc_5q":
        for q in range(5):
            qml.Hadamard(wires=q)
            ry_input(q)
        full_cnot()
        for q in range(5):
            ry_input(q, q + 5)
    elif model_id == "anchor_paper_qfm_5q":
        for _ in range(2):
            for q in range(5):
                qml.Hadamard(wires=q)
                qml.PhaseShift(2 * inputs[..., q], wires=q)
            for q in range(4):
                qml.CNOT(wires=[q, q + 1])
                qml.PhaseShift(2 * (math.pi - inputs[..., q]) * (math.pi - inputs[..., q + 1]), wires=q + 1)
                qml.CNOT(wires=[q, q + 1])
        for layer in range(3):
            for q in range(5):
                qml.RX(weights[layer, q, 0], wires=q)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)
            full_cnot()
    elif model_id == "reupload_full17_5q":
        for repetition in range(2):
            for block in range(4):
                for q in range(5):
                    index = block * 5 + q
                    qml.RY(inputs[..., index] if index < 17 else 0., wires=q)
                rot_layer(weights[repetition, block])
    elif model_id == "reupload_full17_1q":
        for repetition in range(2):
            for block in range(6):
                angles = [inputs[..., block * 3 + j] if block * 3 + j < 17 else 0. for j in range(3)]
                qml.Rot(*angles, wires=0)
                qml.Rot(*weights[repetition, block], wires=0)
    elif model_id in ("ttn_pca8_8q", "mps_pca8_8q", "mera_pca8_8q"):
        for q in range(8):
            ry_input(q)
        pairs = (((1, 2), (3, 4), (5, 6), (0, 1), (2, 3), (4, 5), (6, 7),
                  (3, 5), (1, 3), (5, 7), (3, 7)) if model_id.startswith("mera_") else
                 ((0, 1), (2, 3), (4, 5), (6, 7), (1, 3), (5, 7), (3, 7)) if model_id.startswith("ttn_") else tuple((q, q + 1) for q in range(7)))
        for b, (left, right) in enumerate(pairs):
            qml.Rot(*weights[b, :3], wires=left)
            qml.Rot(*weights[b, 3:], wires=right)
            qml.CNOT(wires=[left, right])
    elif model_id == "qcnn_pca8_8q":
        for q in range(8):
            ry_input(q)
        active = list(range(8))
        for level in range(3):
            w = weights[level * 18:(level + 1) * 18]
            pairs = [(active[i], active[i + 1]) for i in range(0, len(active), 2)]
            if len(active) > 2:
                pairs += [(active[i], active[(i + 1) % len(active)]) for i in range(1, len(active), 2)]
            for a, b in pairs:
                qml.U3(*w[:3], wires=a)
                qml.U3(*w[3:6], wires=b)
                qml.IsingXX(w[6], wires=[a, b])
                qml.IsingYY(w[7], wires=[a, b])
                qml.IsingZZ(w[8], wires=[a, b])
                qml.U3(*w[9:12], wires=a)
                qml.U3(*w[12:15], wires=b)
            for i in range(0, len(active), 2):
                qml.ctrl(qml.U3, control=active[i + 1])(*w[15:18], wires=active[i])
            active = active[::2]
        qml.Rot(*weights[54:57], wires=active[0])
    else:
        if model_id == "amplitude_vqc_5q":
            qml.AmplitudeEmbedding(inputs, wires=range(5), normalize=False)
        elif model_id == "iqp_vqc_5q":
            qml.IQPEmbedding(inputs, wires=range(5), n_repeats=2, pattern=[(i, j) for i in range(5) for j in range(i + 1, 5)])
        else:
            for q in range(5):
                ry_input(q)
        for layer in range(3):
            rot_layer(weights[layer])


def build_variational_model(model_id, *, seed=42, config=None):
    """Build a Torch module; transformed inputs are the responsibility of fit/load."""
    cfg = _resolved_config(model_id, config)
    torch, qml = _dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    dtype = getattr(torch, cfg["dtype"])
    if model_id in ("gibbs_qbm_3q", "gibbs_separable_3q", "dressed_classical_5d"):
        from .second_batch import build_second_batch_model
        return build_second_batch_model(model_id, cfg)
    n_qubits = 1 if model_id.endswith("_1q") else 8 if model_id.endswith("_8q") else 5
    measured = [7] if model_id.startswith(("ttn_", "mps_", "mera_")) else [0] if model_id.startswith("qcnn_") else list(range(n_qubits))
    shape = (() if model_id == ANCHOR_IDS[0] else (2, 4, 5, 3) if model_id == "reupload_full17_5q"
             else (2, 6, 3) if model_id == "reupload_full17_1q" else (7, 6) if model_id.startswith(("ttn_", "mps_"))
             else (11, 6) if model_id.startswith("mera_") else (57,) if model_id.startswith("qcnn_") else (3, 5, 3))

    class VariationalClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_id = model_id
            self.config = cfg
            self.quantum_device = qml.device("default.qubit", wires=n_qubits, shots=None)
            if shape:
                self.angles = torch.nn.Parameter(torch.empty(shape, dtype=dtype))
                torch.nn.init.uniform_(self.angles, 0, 2 * math.pi)
            else:
                self.register_parameter("angles", None)
            self.front = None
            self.residual = None
            if model_id in ANCHOR_IDS:
                width = 10 if model_id == ANCHOR_IDS[0] else 5
                act = lambda: torch.nn.LeakyReLU(.01) if width == 10 else torch.nn.ReLU()
                self.front = torch.nn.Sequential(torch.nn.Linear(17, 23), act(), torch.nn.Linear(23, width), act())
            elif model_id in ("dressed_vqc_5q", "residual_vqc_5q", "dressed_separable_5q"):
                self.front = torch.nn.Linear(17, 5)
            self.readout = torch.nn.Linear(len(measured), 2)
            if model_id == "residual_vqc_5q":
                self.residual = torch.nn.Sequential(torch.nn.Linear(17, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2))
                self.alpha = torch.nn.Parameter(torch.tensor(.1, dtype=dtype))
            self.to(dtype=dtype)

            @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
            def expectations(inputs, weights):
                _apply_circuit(qml, model_id, inputs, weights)
                return [qml.expval(qml.PauliZ(q)) for q in measured]

            @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
            def statevector(inputs, weights):
                _apply_circuit(qml, model_id, inputs, weights)
                return qml.state()

            self.qnode = expectations
            self.state_qnode = statevector

        def encoded(self, inputs):
            if self.front is None:
                return inputs
            projected = self.front(inputs)
            return projected if model_id in ANCHOR_IDS else math.pi * torch.tanh(projected)

        def quantum_features(self, inputs, *, already_encoded=False):
            encoded = inputs if already_encoded else self.encoded(inputs)
            result = self.qnode(encoded, self.angles)
            return torch.stack(result, dim=-1).to(dtype=inputs.dtype)

        def statevector(self, inputs, *, already_encoded=False):
            return self.state_qnode(inputs if already_encoded else self.encoded(inputs), self.angles)

        def forward(self, inputs):
            logits = self.readout(self.quantum_features(inputs))
            if self.residual is not None:
                return self.residual(inputs) + self.alpha * logits
            return logits

    return VariationalClassifier()


def load_paper_reference_state(model, state_dict):
    """Map a trusted frozen reproduction state_dict into the PennyLane anchor."""
    if model.model_id not in ANCHOR_IDS:
        raise ValueError("Reference mapping is only defined for paper anchors")
    mapping = {"0.weight": "front.0.weight", "0.bias": "front.0.bias",
               "2.weight": "front.2.weight", "2.bias": "front.2.bias",
               "5.weight": "readout.weight", "5.bias": "readout.bias"}
    if model.model_id == ANCHOR_IDS[1]:
        mapping["4.angles"] = "angles"
    if set(state_dict) != set(mapping):
        raise ValueError("Unexpected paper reference parameter keys")
    model.load_state_dict({mapping[key]: value for key, value in state_dict.items()}, strict=True)
    return model


def _circuit_spec(model, example):
    torch, qml = _dependencies()
    if hasattr(model, "model_spec"):
        return model.model_spec
    with torch.no_grad():
        encoded = model.encoded(example)
        tape = qml.tape.make_qscript(lambda: _apply_circuit(qml, model.model_id, encoded, model.angles))()
    return {
        "wire_order": list(model.quantum_device.wires), "qubit_0_statevector_bit": "most_significant",
        "quantum_parameter_shape": [] if model.angles is None else list(model.angles.shape),
        "operations_in_order": [{"name": op.name, "wires": list(op.wires),
                                 "parameter_shapes": [list(qml.math.shape(p)) for p in op.parameters]}
                                for op in tape.operations],
        "classical_module": str(model),
        "input_embedding": model.model_id,
        "iqp_repeats": 2 if model.model_id == "iqp_vqc_5q" else None,
        "iqp_pattern": [[i, j] for i in range(5) for j in range(i + 1, 5)] if model.model_id == "iqp_vqc_5q" else None,
        "specification_source": "docs/quantum_algorithm_catalog.md plus frozen variational.py source hash",
        "parameter_value_storage": "complete initial/epoch/best/last/final state_dict checkpoints",
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_digest(x, y):
    digest = hashlib.sha256()
    for a in (np.asarray(x, dtype="<f8"), np.asarray(y, dtype="<i8")):
        digest.update(str(a.shape).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def _rng_state(torch):
    n = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]]}


def _restore_rng(torch, value):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


@dataclass
class FittedVariational:
    model_id: str
    model: Any
    preprocessor: FeatureTransform
    config: dict
    history: list
    output_dir: Path

    def _logits(self, raw):
        torch, _ = _dependencies()
        values = self.preprocessor.transform(raw)
        if not len(values):
            return np.empty((0, 2), dtype=np.float64)
        self.model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                x = torch.as_tensor(values[start:start + 256], dtype=getattr(torch, self.config["dtype"]))
                parts.append(self.model(x).cpu().numpy())
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite prediction logits")
        return result

    def predict_score(self, raw):
        logits = self._logits(raw)
        return logits[:, 1] - logits[:, 0]

    def predict_proba(self, raw):
        logits = self._logits(raw).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True) if len(logits) else 0
        exp = np.exp(logits)
        return exp[:, 1] / exp.sum(axis=1)

    def predict(self, raw):
        return (self.predict_proba(raw) > .5).astype(np.int64)


def fit_variational(model_id, x_train_raw17, y_train, *, seed, output_dir, config=None):
    """Fit final-epoch weights; save every epoch and never inspect held-out data.

    ``resume_from`` may name a last/epoch checkpoint in a different run. The new
    output directory must be empty; data/seed/config are checked before restoring
    optimizer and RNG. ``epochs`` is the total desired epoch count, not extra
    epochs. Prior source artifacts are never overwritten.
    """
    cfg = _resolved_config(model_id, config)
    x = _validate_x(x_train_raw17)
    y = np.asarray(y_train)
    if y.shape != (len(x),) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training labels must be aligned and contain both 0/1 classes")
    y = y.astype(np.int64)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty model output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    started = time.perf_counter()
    metadata = {"format_version": FORMAT_VERSION, "model_id": model_id, "status": "running",
                "seed": int(seed), "config": cfg, "training_rows": len(x),
                "training_data_sha256": _data_digest(x, y),
                "source_sha256": _sha(Path(__file__)), "history": [],
                "implementation_source_sha256": {name: _sha(Path(__file__).with_name(name))
                                                  for name in ("variational.py", "second_batch.py")},
                "selection": "final_epoch; no validation, early stopping or threshold tuning",
                "probability_kind": "uncalibrated_softmax_pass_probability",
                "prediction_rule": "pass_probability > 0.5; exact tie predicts Fail",
                "outlier_filter": "none inside adapter; caller owns any paper-compatible IF",
                "backend": "pennylane.default.qubit", "shots": None, "diff_method": "backprop",
                "quantum_initialization": "uniform[0,2*pi); seed before model construction",
                "classical_initialization": "seeded torch.nn.Linear defaults, converted to configured dtype"}
    _write_json(output / "metadata.json", metadata)
    history, checkpoint_index = [], {}
    try:
        torch, qml = _dependencies()
        metadata["versions"] = {"torch": str(torch.__version__), "pennylane": str(qml.__version__), "numpy": np.__version__}
        model = build_variational_model(model_id, seed=seed, config=cfg)
        if model_id.startswith("gibbs_"):
            metadata.update(backend="torch.linalg.matrix_exp", diff_method="torch_autograd_exact_Gibbs",
                            probability_kind="exact_Gibbs_marginal_pass_probability; floor_1e-12_and_renormalize",
                            classical_initialization="Hamiltonian fields W~Normal(0,.05), b=0; see circuit_spec",
                            quantum_initialization="Hamiltonian fields/couplings/Gamma as recorded in circuit_spec")
        elif model_id == "dressed_classical_5d":
            metadata.update(backend="torch", diff_method="torch_autograd", quantum_initialization="not_applicable")
        optimizer_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=tuple(cfg["betas"]), eps=cfg["eps"])
        start_epoch = 0
        if cfg["resume_from"]:
            source = Path(cfg["resume_from"])
            saved = torch.load(source, map_location="cpu", weights_only=True)
            if saved["model_id"] != model_id or saved["seed"] != seed or saved["training_data_sha256"] != metadata["training_data_sha256"]:
                raise ValueError("Resume model, seed or exact ordered training data does not match")
            if saved["source_sha256"] != metadata["source_sha256"]:
                raise ValueError("Resume requires the identical variational implementation source")
            if saved.get("implementation_source_sha256") != metadata["implementation_source_sha256"]:
                raise ValueError("Resume requires all implementation module sources to match")
            comparison_keys = set(cfg) - {"epochs", "resume_from", "max_wall_seconds"}
            if any(cfg[key] != saved["config"][key] for key in comparison_keys):
                raise ValueError("Resume configuration changes numerical training settings")
            if saved["epoch"] >= cfg["epochs"]:
                raise ValueError("Resume target epochs must exceed checkpoint epoch")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            _restore_rng(torch, saved["rng_state"])
            preprocessor = FeatureTransform.from_dict(saved["preprocessor"])
            history = saved["history"]
            start_epoch = saved["epoch"]
            metadata["resume_source"] = {"sha256": _sha(source), "epoch": start_epoch,
                                         "relative_checkpoint_path": os.path.relpath(source.resolve(), output.resolve())}
        else:
            preprocessor = FeatureTransform.fit(model_id, x)
        transformed = preprocessor.transform(x)
        tensor_x = torch.as_tensor(transformed, dtype=getattr(torch, cfg["dtype"]))
        tensor_y = torch.as_tensor(y, dtype=torch.long)
        metadata["preprocessor"] = preprocessor.to_dict()
        metadata["parameter_counts"] = {"total": sum(p.numel() for p in model.parameters()),
                                         "quantum": 0 if model.angles is None else model.angles.numel()}
        if model_id.startswith("gibbs_"):
            metadata["parameter_counts"].update(
                quantum=None, quantum_circuit_gate_parameters=None,
                hamiltonian_total=sum(p.numel() for p in model.parameters()),
                fields=sum(p.numel() for p in model.fields.parameters()),
                couplings=3 if model_id == "gibbs_qbm_3q" else 0,
                transverse=model.angles.numel(),
                note="Hamiltonian parameters; quantum circuit gate parameter counts are not applicable")
        metadata["circuit_spec"] = _circuit_spec(model, tensor_x[:1])
        _write_json(output / "preprocessor.json", preprocessor.to_dict())

        def save_checkpoint(filename, epoch):
            payload = {"format_version": FORMAT_VERSION, "model_id": model_id, "seed": seed,
                       "config": cfg, "epoch": epoch, "training_data_sha256": metadata["training_data_sha256"],
                       "source_sha256": metadata["source_sha256"],
                       "implementation_source_sha256": metadata["implementation_source_sha256"],
                       "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       "optimizer_state": optimizer.state_dict(), "rng_state": _rng_state(torch),
                       "preprocessor": preprocessor.to_dict(), "history": list(history)}
            path = checkpoints / filename
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            replace_with_retry(temporary, path)
            checkpoint_index[filename] = {"epoch": epoch, "sha256": _sha(path), "bytes": path.stat().st_size}
            _write_json(output / "checkpoint_index.json", checkpoint_index)

        save_checkpoint("initial.pt", start_epoch)
        save_checkpoint("last.pt", start_epoch)
        _write_json(output / "metadata.json", metadata)
        batch_size = cfg["batch_size"] or len(x)
        for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
            if cfg["max_wall_seconds"] is not None and time.perf_counter() - started > cfg["max_wall_seconds"]:
                raise TimeoutError("Predeclared variational wall-clock budget exhausted between epochs")
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(x)) if cfg["shuffle"] else torch.arange(len(x))
            total_loss, gradient_sq = 0., 0.
            with (qml.Tracker(model.quantum_device) if hasattr(model, "quantum_device") else nullcontext(SimpleNamespace(totals={}))) as tracker:
                for start in range(0, len(x), batch_size):
                    ids = order[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tensor_x[ids])
                    loss = torch.nn.functional.cross_entropy(logits, tensor_y[ids])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                        raise FloatingPointError("Missing or nonfinite training gradients")
                    gradient_sq += sum(float(g.detach().square().sum()) for g in gradients)
                    if cfg["clip_grad_norm"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(ids)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Optimizer produced nonfinite parameters")
            history.append({"epoch": epoch, "train_loss": total_loss / len(x),
                            "gradient_l2_over_batches": math.sqrt(gradient_sq),
                            "seconds": time.perf_counter() - epoch_start,
                            "device_executions": int(tracker.totals.get("executions", 0)),
                            "device_simulations": int(tracker.totals.get("simulations", 0))})
            save_checkpoint(f"epoch_{epoch:04d}.pt", epoch)
            save_checkpoint("last.pt", epoch)
            segment_history = [row for row in history if row["epoch"] > start_epoch]
            if history[-1]["train_loss"] <= min(row["train_loss"] for row in segment_history):
                save_checkpoint("best.pt", epoch)
                metadata["best_training_loss_epoch"] = epoch
                metadata["best_checkpoint_role"] = "diagnostic only; minimum observed training epoch loss; predictions use final"
                metadata["best_checkpoint_scope"] = "epochs trained in this output directory; prior segment snapshots remain in resume lineage"
            _write_json(output / "history.json", history)
        save_checkpoint("final.pt", cfg["epochs"])
        metadata.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                        epochs_completed=cfg["epochs"], history=history)
        _write_json(output / "metadata.json", metadata)
        return FittedVariational(model_id, model, preprocessor, cfg, history, output)
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.perf_counter() - started,
                        epochs_completed=history[-1]["epoch"] if history else 0, history=history)
        _write_json(output / "metadata.json", metadata)
        raise


def load_variational(output_dir):
    """Load only a completed local fit and verify final checkpoint content hash."""
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != FORMAT_VERSION or metadata["status"] != "completed":
        raise ValueError("Only completed supported variational artifacts can be loaded")
    index = json.loads((output / "checkpoint_index.json").read_text(encoding="utf-8"))
    path = output / "checkpoints" / "final.pt"
    if _sha(path) != index["final.pt"]["sha256"]:
        raise ValueError("Final checkpoint hash mismatch")
    torch, _ = _dependencies()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_id"] != metadata["model_id"] or checkpoint["epoch"] != metadata["config"]["epochs"]:
        raise ValueError("Checkpoint metadata mismatch")
    model = build_variational_model(metadata["model_id"], seed=metadata["seed"], config=metadata["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return FittedVariational(metadata["model_id"], model,
                             FeatureTransform.from_dict(checkpoint["preprocessor"]),
                             metadata["config"], checkpoint["history"], output)
