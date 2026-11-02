# Copyright 2024 Xanadu Quantum Technologies Inc.
# Adaptation and additional matched controls for AI4CT, 2026.
# Licensed under the Apache License, Version 2.0; see LICENSE_Xanadu.txt.
"""Explicit Torch adaptation of five Xanadu re-uploading variants and two controls.

Upstream circuit semantics are preserved, including coupled changes in NoScaling
and Separable, trainable omega in NoTrainableEmbedding, and the all-qubit summed
NoCost objective. Optimization settings and RNG differ from the JAX reference.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
import torch

from ai4ct.qbench.common import seed_everything, sha256, validate_x, validate_xy, write_json
from ai4ct.qbench.variational import _data_digest, _restore_rng, _rng_state


SOURCE_COMMIT = "c226205f5442e49df609904ce8c37e1bb4d3ad98"
SOURCE_SHA256 = "4e591d839e6c16283e8d13a9362c84817a581e84fcb9ad642abec38c28f87e16"
SOURCE_URL = f"https://github.com/XanaduAI/qml-benchmarks/blob/{SOURCE_COMMIT}/src/qml_benchmarks/models/data_reuploading.py"
MODEL_VARIANTS = {
    "reupload_reference_base_6q": "base",
    "reupload_reference_no_scaling_6q": "no_scaling",
    "reupload_reference_no_cost_6q": "no_cost",
    "reupload_reference_no_trainable_embedding_6q": "no_trainable_embedding",
    "reupload_reference_separable_6q": "separable",
    "reupload_matched_no_scaling_6q": "matched_no_scaling",
    "reupload_matched_separable_6q": "matched_separable",
}
DIFFERENCES = {
    "base": "upstream split encoding/processing Rot, compressed final Rot",
    "no_scaling": "upstream composite change: remove omega AND merge Rot(x)Rot(theta) into Rot(x+theta)",
    "no_cost": "upstream loss replacement: all-qubit sum, no alpha contribution; negative finite loss is valid",
    "no_trainable_embedding": "upstream reordered circuit: all encoding before processing; omega remains trainable; final encoding and processing are separate",
    "separable": "upstream composite change: remove CZ AND merge encoding/processing rotations",
    "matched_no_scaling": "single factor versus base: omega fixed to one; split rotations preserved",
    "matched_separable": "single factor versus base: CZ removed; split rotations and trainable omega preserved",
}


def input_support(variant, n_layers, observable_weight, n_qubits=6):
    """Structural light cone of the fixed prediction readout (zero-based columns)."""
    wires = set(range(observable_weight))
    if variant not in {"separable", "matched_separable"}:
        for layer in reversed(range(n_layers)):
            for first in range(layer % 2, n_qubits-1, 2):
                if wires & {first, first+1}:
                    wires.update((first, first+1))
    return [3*wire+offset for wire in sorted(wires) for offset in range(3) if 3*wire+offset < 17]


def resolved_config(config=None):
    value = dict(epochs=30, n_layers=3, observable_type="single", lr=.005,
                 optimizer="adam", weight_decay=0.0, betas=[.9, .999], eps=1e-8,
                 dtype="float64", device="cpu", threads=1, batch_size=None,
                 scaling=1.0, max_wall_seconds=1700.0, resume_from=None)
    supplied = {} if config is None else dict(config)
    if set(supplied) - set(value):
        raise ValueError(f"Unknown config keys: {sorted(set(supplied)-set(value))}")
    value.update(supplied)
    if type(value["epochs"]) is not int or value["epochs"] < 1 or type(value["n_layers"]) is not int or value["n_layers"] < 1:
        raise ValueError("Positive integer epochs/layers required")
    if value["observable_type"] not in {"single", "half", "full"}:
        raise ValueError("Unknown observable range")
    if value["dtype"] != "float64" or value["device"] != "cpu" or value["optimizer"] != "adam" or value["batch_size"] is not None:
        raise ValueError("This registered adapter uses CPU float64 full-batch Adam")
    if value["weight_decay"] != 0 or value["scaling"] != 1 or value["threads"] != 1:
        raise ValueError("Registered weight_decay=0, scaling=1 and threads=1 required")
    if not np.isfinite(value["lr"]) or value["lr"] <= 0 or not np.isfinite(value["max_wall_seconds"]) or value["max_wall_seconds"] <= 0:
        raise ValueError("Positive finite learning rate and wall budget required")
    return value


class InputTransform:
    def __init__(self, scale, offset, data_min, data_max):
        self.scale = np.asarray(scale, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        self.data_min = np.asarray(data_min, dtype=float)
        self.data_max = np.asarray(data_max, dtype=float)

    @classmethod
    def fit(cls, raw):
        fitted = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2), clip=False).fit(validate_x(raw))
        return cls(fitted.scale_, fitted.min_, fitted.data_min_, fitted.data_max_)

    def transform(self, raw):
        x = validate_x(raw)
        return np.pad(x * self.scale + self.offset, ((0, 0), (0, 1)))

    def to_dict(self):
        return {"kind": "train_minmax17_pad18", "clip": False, "n_features_in": 17,
                "scale": self.scale.tolist(), "offset": self.offset.tolist(),
                "data_min": self.data_min.tolist(), "data_max": self.data_max.tolist()}

    @classmethod
    def from_dict(cls, value):
        if value["kind"] != "train_minmax17_pad18" or value["clip"] is not False:
            raise ValueError("Preprocessor contract differs")
        return cls(value["scale"], value["offset"], value["data_min"], value["data_max"])


class ReferenceCircuit(torch.nn.Module):
    def __init__(self, variant, n_layers=3, n_qubits=6, observable_type="single"):
        super().__init__()
        self.variant, self.n_layers, self.n_qubits = variant, n_layers, n_qubits
        self.observable_weight = {"single": 1, "half": max(1, n_qubits//2), "full": n_qubits}[observable_type]
        self.thetas = torch.nn.Parameter(torch.rand(n_qubits, n_layers+1, 3, dtype=torch.float64))
        if variant == "no_scaling":
            self.register_parameter("omegas", None)
        elif variant == "matched_no_scaling":
            self.register_buffer("omegas", torch.ones(n_layers+1, 3*n_qubits, dtype=torch.float64))
        else:
            self.omegas = torch.nn.Parameter(torch.rand(n_layers+1, 3*n_qubits, dtype=torch.float64))
        self.alphas = torch.nn.Parameter(torch.ones(2, self.observable_weight, dtype=torch.float64))
        device = qml.device("default.qubit", wires=n_qubits)

        def rotation(angles, wire):
            qml.Rot(angles[..., 0], angles[..., 1], angles[..., 2], wires=wire)

        def entangle(layer):
            if variant in {"separable", "matched_separable"}:
                return
            for first in range(layer % 2, n_qubits-1, 2):
                qml.CZ(wires=[first, first+1])

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(x, thetas, omegas):
            if variant == "no_trainable_embedding":
                for layer in range(n_layers+1):
                    for wire in range(n_qubits):
                        start = wire * 3
                        rotation(x[..., start:start+3] * omegas[layer, start:start+3], wire)
                for layer in range(n_layers):
                    for wire in range(n_qubits):
                        rotation(thetas[wire, layer], wire)
                    entangle(layer)
                for wire in range(n_qubits):
                    rotation(thetas[wire, n_layers], wire)
            else:
                for layer in range(n_layers):
                    for wire in range(n_qubits):
                        start = wire * 3
                        inputs = x[..., start:start+3]
                        if variant != "no_scaling":
                            inputs = inputs * omegas[layer, start:start+3]
                        if variant in {"no_scaling", "separable"}:
                            rotation(inputs + thetas[wire, layer], wire)
                        else:
                            rotation(inputs, wire)
                            rotation(thetas[wire, layer], wire)
                    entangle(layer)
                for wire in range(n_qubits):
                    start = wire * 3
                    inputs = x[..., start:start+3]
                    if variant != "no_scaling":
                        inputs = inputs * omegas[n_layers, start:start+3]
                    rotation(inputs + thetas[wire, n_layers], wire)
            return [qml.expval(qml.Z(wire)) for wire in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        return torch.stack(self.circuit(x, self.thetas, self.omegas), dim=-1)

    def loss(self, z, y):
        if self.variant == "no_cost":
            return ((1 - (2*y-1) * z.sum(dim=-1)) / 2).mean()
        p0, p1 = (1-z[:, :self.observable_weight])/2, (1+z[:, :self.observable_weight])/2
        return .5 * (((self.alphas[0]*p0-(1-y[:, None]))**2
                      + (self.alphas[1]*p1-y[:, None])**2).sum() / len(y))


class FittedReupload:
    def __init__(self, model, preprocessor, metadata):
        self.model, self.preprocessor, self.metadata = model, preprocessor, metadata

    def predict_score(self, raw):
        self.model.eval()
        transformed = self.preprocessor.transform(raw)
        scores = []
        with torch.no_grad():
            for start in range(0, len(transformed), 256):
                z = self.model(torch.as_tensor(transformed[start:start+256], dtype=torch.float64))
                scores.append(z[:, :self.model.observable_weight].mean(dim=-1).numpy())
        return np.concatenate(scores)

    def predict_proba(self, raw):
        probability = (1+self.predict_score(raw))/2
        if np.any(probability < -1e-10) or np.any(probability > 1+1e-10):
            raise FloatingPointError("Quantum probabilities out of bounds")
        return np.clip(probability, 0, 1)


def fit_reupload(model_id, x_train_raw17, y_train, *, seed, output_dir, config=None):
    cfg = resolved_config(config)
    variant = MODEL_VARIANTS[model_id]
    x, y = validate_xy(x_train_raw17, y_train)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refuse to overwrite prior model artifacts")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    seed_everything(seed, threads=1)
    model = ReferenceCircuit(variant, cfg["n_layers"], 6, cfg["observable_type"])
    preprocessor = InputTransform.fit(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=0)
    fingerprint = _data_digest(x, y)
    metadata = {"schema": "qbench-reupload-reference-v1", "status": "running", "model_id": model_id,
                "variant": variant, "seed": seed, "config": cfg, "n_train": len(y),
                "training_data_sha256": fingerprint, "source_sha256": sha256(__file__),
                "upstream_source_url": SOURCE_URL, "upstream_source_sha256": SOURCE_SHA256,
                "semantic_difference": DIFFERENCES[variant], "observable_weight": model.observable_weight,
                "n_qubits": 6, "feature_representation": "same train-only 17D MinMax[-pi/2,pi/2], clip=False, pad18 for every variant",
                "implementation_adaptations": ["Torch RNG instead of JAX; same U[0,1) distribution, no seed-level upstream equivalence",
                    "L=3 instead of upstream default4", "Adam .005 full-batch 30epoch budget instead of upstream .05, batch32, max10000/convergence stopping",
                    "higher Z means Pass; upstream fidelity basis comments are reversed relative to PauliZ convention"],
                "no_cost_scope": "all six qubits summed; negative finite loss is valid" if variant == "no_cost" else None,
                "unused_trainable_parameters": ["alphas"] if variant == "no_cost" else [],
                "prediction_structural_input_columns": input_support(variant, cfg["n_layers"], model.observable_weight),
                "loss_structural_input_columns": list(range(17)) if variant == "no_cost" else input_support(variant, cfg["n_layers"], model.observable_weight),
                "parameter_count_note": "registered tensor entries include zero-padded omega and parameters outside the prediction light cone",
                "selection": "final epoch; best alias is diagnostic training-loss only", "probability_kind": "uncalibrated_mean_PauliZ_affine_pass_probability",
                "checkpoint_epoch_snapshots": "saved", "resume_mode": "epoch_boundary", "history": [],
                "versions": {"torch": str(torch.__version__), "pennylane": qml.__version__, "numpy": np.__version__}}
    history, index, start_epoch, best_epoch, best_loss = [], {}, 0, None, float("inf")
    if cfg["resume_from"]:
        source = Path(cfg["resume_from"])
        saved = torch.load(source, map_location="cpu", weights_only=True)
        if saved["model_id"] != model_id or saved["seed"] != seed or saved["training_data_sha256"] != fingerprint:
            raise ValueError("Resume model/seed/training data differs")
        keys = set(cfg)-{"epochs", "resume_from", "max_wall_seconds"}
        if any(cfg[key] != saved["config"][key] for key in keys) or saved["epoch"] >= cfg["epochs"]:
            raise ValueError("Resume configuration differs or target already reached")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        _restore_rng(torch, saved["rng_state"])
        preprocessor = InputTransform.from_dict(saved["preprocessor"])
        history, start_epoch = list(saved["history"]), saved["epoch"]
        best_epoch, best_loss = saved["best_epoch"], saved["best_loss"]
        for name in ["initial.pt", *[f"epoch_{n:04d}.pt" for n in range(1, start_epoch+1)]]:
            shutil.copy2(source.parent / name, checkpoints / name)
            index[name] = {"sha256": sha256(checkpoints/name), "copied_from_resume_lineage": True}
        metadata["resume_lineage"] = {"checkpoint_sha256": sha256(source), "epoch": start_epoch}
    metadata["preprocessor"] = preprocessor.to_dict()
    metadata["parameter_counts"] = {"trainable": sum(p.numel() for p in model.parameters()),
                                     "trainable_excluding_unused_alpha": sum(p.numel() for name, p in model.named_parameters() if not (variant=="no_cost" and name=="alphas"))}
    write_json(output / "metadata.json", metadata)
    tensor_x = torch.as_tensor(preprocessor.transform(x), dtype=torch.float64)
    tensor_y = torch.as_tensor(y, dtype=torch.float64)
    started = time.perf_counter()

    def save(name, epoch):
        payload = {"model_id": model_id, "seed": seed, "config": cfg, "epoch": epoch,
                   "training_data_sha256": fingerprint, "model_state": copy.deepcopy(model.state_dict()),
                   "optimizer_state": optimizer.state_dict(), "scheduler_state": None, "amp_scaler_state": None,
                   "rng_state": _rng_state(torch), "sampler_state": {"kind": "full_batch_no_shuffle", "order": list(range(len(y))), "cursor": 0},
                   "preprocessor": preprocessor.to_dict(), "history": list(history), "best_epoch": best_epoch, "best_loss": best_loss}
        # Each tensor snapshot has a unique name; only the JSON index is replaced.
        path = checkpoints / name
        if path.exists():
            raise FileExistsError("Checkpoint name already exists")
        torch.save(payload, path)
        index[name] = {"epoch": epoch, "sha256": sha256(path), "bytes": path.stat().st_size}
        write_json(output / "checkpoint_index.json", index)

    if not cfg["resume_from"]:
        save("initial.pt", 0)
    try:
        for epoch in range(start_epoch+1, cfg["epochs"]+1):
            if time.perf_counter()-started > cfg["max_wall_seconds"]:
                raise TimeoutError("Registered training wall-clock budget exhausted")
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(tensor_x), tensor_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Nonfinite or missing gradients")
            optimizer.step()
            with torch.no_grad():
                after = float(model.loss(model(tensor_x), tensor_y))
            if not np.isfinite(after) or not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite post-update state")
            history.append({"epoch": epoch, "train_loss_pre_update": float(loss.detach()), "train_loss": after,
                            "gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in gradients))),
                            "seconds": time.perf_counter()-tick})
            if after < best_loss:
                best_epoch, best_loss = epoch, after
            save(f"epoch_{epoch:04d}.pt", epoch)
            write_json(checkpoints / "aliases.json", {"last": f"epoch_{epoch:04d}.pt", "best": f"epoch_{best_epoch:04d}.pt",
                                                       "best_rule": "minimum post-update training loss; final checkpoint used for prediction"})
        save("final.pt", cfg["epochs"])
        metadata.update(status="completed", history=history, epochs_completed=cfg["epochs"], fit_seconds=time.perf_counter()-started)
        write_json(output / "metadata.json", metadata)
        return FittedReupload(model, preprocessor, metadata)
    except Exception as exc:
        metadata.update(status="failed", history=history, error_type=type(exc).__name__, error=str(exc))
        write_json(output / "metadata.json", metadata)
        raise


def load_reupload(output_dir):
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed":
        raise ValueError("Only completed fits can be loaded")
    path = output / "checkpoints/final.pt"
    index = json.loads((output / "checkpoint_index.json").read_text(encoding="utf-8"))
    if sha256(path) != index["final.pt"]["sha256"]:
        raise ValueError("Final checkpoint hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    if saved["model_id"] != metadata["model_id"] or saved["seed"] != metadata["seed"] or saved["epoch"] != cfg["epochs"]:
        raise ValueError("Checkpoint metadata differs")
    model = ReferenceCircuit(MODEL_VARIANTS[saved["model_id"]], cfg["n_layers"], 6, cfg["observable_type"])
    model.load_state_dict(saved["model_state"])
    return FittedReupload(model, InputTransform.from_dict(saved["preprocessor"]), metadata)
