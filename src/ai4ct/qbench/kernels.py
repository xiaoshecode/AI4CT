"""PennyLane quantum kernels and fixed feature maps with classical readouts.

All source representations, kernel learning and readout fits see training data
only. Exact statevectors are used to avoid unnecessary pairwise circuit calls.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

from .common import Representation, ensure_model_directory, seed_everything, validate_xy, write_json

MODEL_IDS = ("kernel_fidelity_5q", "kernel_projected_5q", "kernel_trainable_5q",
             "random_quantum_features_5q", "static_reservoir_5q", "kernel_iqp_5q",
             "kernel_separable_5q", "kernel_amplitude_5q", "quantum_metric_5q",
             "quantum_extreme_learning_5q", "kernel_knn_5q")
N_QUBITS = 5


def state_circuit(kind, trainable=False):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(a, theta=None):
        if kind == "amplitude":
            qml.AmplitudeEmbedding(a, wires=range(N_QUBITS), normalize=False)
        elif kind == "separable":
            for i in range(N_QUBITS):
                qml.RY(a[..., i], wires=i)
        elif kind == "iqp":
            qml.IQPEmbedding(a, wires=range(N_QUBITS), n_repeats=2)
        else:
            for layer in range(2):
                if trainable:
                    for i in range(N_QUBITS):
                        qml.RY(theta[layer, i], wires=i)
                for i in range(N_QUBITS):
                    qml.Hadamard(wires=i)
                    qml.PhaseShift(2 * a[..., i], wires=i)
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i+1])
                    qml.PhaseShift(2 * (np.pi-a[..., i]) * (np.pi-a[..., i+1]), wires=i+1)
                    qml.CNOT(wires=[i, i+1])
        return qml.state()
    return circuit


def _amplitudes(z):
    a = np.pad(z, ((0, 0), (0, 32-z.shape[1])))
    norm = np.linalg.norm(a, axis=1)
    zero = norm < 1e-14
    a[zero, 0] = 1
    norm[zero] = 1
    return a / norm[:, None]


def fidelity_kernel(a, b):
    return np.abs(np.asarray(a) @ np.asarray(b).conj().T) ** 2


def local_bloch(states):
    """X,Y,Z in wire-major order; PennyLane wire zero is MSB."""
    states = np.asarray(states)
    index = np.arange(2**N_QUBITS)
    output = []
    for wire in range(N_QUBITS):
        bit = (index >> (N_QUBITS-1-wire)) & 1
        flip = index ^ (1 << (N_QUBITS-1-wire))
        output.extend([
            (states.conj()*states[:, flip]).sum(1).real,
            (states.conj()*(-1j*(1-2*bit))*states[:, flip]).sum(1).real,
            (np.abs(states)**2*(1-2*bit)).sum(1).real,
        ])
    return np.column_stack(output)


def rbf(a, b, gamma):
    distance = (a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T
    return np.exp(-gamma * np.maximum(distance, 0))


class QuantumMapping:
    def __init__(self, model_id, representation, seed, config):
        self.model_id, self.representation, self.seed, self.config = model_id, representation, seed, config
        self.theta = None
        rng = np.random.default_rng(seed)
        if model_id in ("random_quantum_features_5q", "quantum_extreme_learning_5q"):
            self.weights = rng.normal(0, 1/np.sqrt(17), (16, 5, 17))
            self.bias = rng.uniform(-np.pi, np.pi, (16, 5))
            self.rotations = rng.uniform(-np.pi, np.pi, (16, 2, 5, 3))
        elif model_id == "static_reservoir_5q":
            self.j = rng.uniform(-1, 1, 4)
            self.h = rng.uniform(-1, 1, 5)
            self.times = [.5, 1., 1.5, 2.]

    def _kind(self):
        return {"kernel_iqp_5q": "iqp", "kernel_separable_5q": "separable",
                "kernel_amplitude_5q": "amplitude"}.get(self.model_id, "zz")

    def states(self, x):
        import torch
        a = self.representation.transform(x)
        if self._kind() == "amplitude":
            a = _amplitudes(a)
        circuit = state_circuit(self._kind(), self.theta is not None)
        theta = None if self.theta is None else torch.as_tensor(self.theta, dtype=torch.float64)
        with torch.no_grad():
            return np.asarray(circuit(torch.as_tensor(a, dtype=torch.float64), theta).detach().numpy())

    def features(self, x):
        import torch
        import pennylane as qml
        z = self.representation.transform(x)
        if self.model_id == "kernel_projected_5q":
            return local_bloch(self.states(x))
        dev = qml.device("default.qubit", wires=5, shots=None)
        if self.model_id in ("random_quantum_features_5q", "quantum_extreme_learning_5q"):
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def quantum(a, angles):
                for i in range(5):
                    qml.RY(a[..., i], wires=i)
                for layer in range(2):
                    for i in range(5):
                        qml.Rot(*angles[layer, i], wires=i)
                    for i in range(4):
                        qml.CNOT(wires=[i, i+1])
                return [qml.expval(qml.PauliZ(i)) for i in range(5)]
            outputs = []
            with torch.no_grad():
                for w, bias, angles in zip(self.weights, self.bias, self.rotations):
                    a = torch.as_tensor(2*np.arctan(z@w.T+bias), dtype=torch.float64)
                    values = quantum(a, torch.as_tensor(angles, dtype=torch.float64))
                    outputs.append(torch.stack(values, dim=1).numpy())
            return np.concatenate(outputs, axis=1)
        if self.model_id != "static_reservoir_5q":
            raise ValueError("No fixed feature representation for this kernel model")
        terms = [qml.PauliZ(i)@qml.PauliZ(i+1) for i in range(4)] + [qml.PauliX(i) for i in range(5)]
        hamiltonian = qml.Hamiltonian(np.concatenate([self.j, self.h]), terms)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def reservoir(a, t):
            for i in range(5):
                qml.RY(a[..., i], wires=i)
            qml.ApproxTimeEvolution(hamiltonian, t, n=20)
            return ([qml.expval(qml.PauliZ(i)) for i in range(5)] +
                    [qml.expval(qml.PauliZ(i)@qml.PauliZ(i+1)) for i in range(4)])
        with torch.no_grad():
            a = torch.as_tensor(z, dtype=torch.float64)
            return np.concatenate([torch.stack(reservoir(a, t), dim=1).numpy() for t in self.times], axis=1)


class KernelFitted:
    def __init__(self, mapping, estimator, train_embedding, y_train, config):
        self.mapping, self.estimator, self.train_embedding = mapping, estimator, train_embedding
        self.y_train, self.config = y_train, config

    def _features(self, x):
        cache_key = hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest()
        if getattr(self, "_cached_key", None) == cache_key:
            return self._cached_features
        if self.mapping.model_id == "kernel_projected_5q":
            values = rbf(self.mapping.features(x), self.train_embedding, float(self.config.get("gamma", 1/15)))
        elif self.mapping.model_id in ("random_quantum_features_5q", "static_reservoir_5q", "quantum_extreme_learning_5q"):
            values = self.mapping.features(x)
        else:
            values = fidelity_kernel(self.mapping.states(x), self.train_embedding)
        self._cached_key, self._cached_features = cache_key, values
        return values

    def predict_proba(self, x):
        if self.mapping.model_id == "kernel_knn_5q":
            k = min(int(self.config.get("neighbors", 5)), len(self.y_train))
            similarity = self._features(x)
            indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
            return self.y_train[indices].mean(axis=1)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self._features(x))[:, list(self.estimator.classes_).index(1)]
        return None

    def predict_score(self, x):
        if self.mapping.model_id == "kernel_knn_5q":
            return self.predict_proba(x)-.5
        return np.asarray(self.estimator.decision_function(self._features(x)))


def fit_kernel(model_id, x_train, y_train, *, seed, output_dir, config):
    import torch
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown quantum kernel/feature model {model_id}")
    x, y = validate_xy(x_train, y_train)
    output_dir = ensure_model_directory(output_dir)
    config = dict(config)
    seed_everything(seed, int(config.get("threads", 1)))
    start = time.perf_counter()
    full = model_id in ("kernel_amplitude_5q", "random_quantum_features_5q", "quantum_extreme_learning_5q")
    rep = Representation(components=None if full else 5, angle=not full).fit(x)
    mapping = QuantumMapping(model_id, rep, seed, config)
    history = []
    trainable = model_id in ("kernel_trainable_5q", "quantum_metric_5q")
    if trainable:
        theta = torch.nn.Parameter(torch.zeros((2, 5), dtype=torch.float64))
        optimizer = torch.optim.Adam([theta], lr=float(config.get("lr", .01)))
        a = torch.as_tensor(rep.transform(x), dtype=torch.float64)
        labels = torch.as_tensor(2*y-1, dtype=torch.float64)
        circuit = state_circuit("zz", True)
        rng = np.random.default_rng(seed)
        training_digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        start_epoch = 0
        if config.get("resume_from"):
            previous = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
            if previous.get("training_digest") != training_digest or previous.get("model_id") != model_id or previous.get("seed") != seed:
                raise ValueError("Resume kernel/data/seed mismatch")
            for key, default in (("lr",.01),("alignment_batch_size",64)):
                if previous["config"].get(key,default) != config.get(key,default):
                    raise ValueError("Resume kernel optimizer/batch configuration mismatch")
            with torch.no_grad():
                theta.copy_(previous["theta"])
            optimizer.load_state_dict(previous["optimizer"])
            torch.set_rng_state(previous["torch_rng"])
            rng.bit_generator.state=previous["numpy_rng"]
            random.setstate(previous["python_rng"])
            start_epoch=int(previous["epoch"])
        if int(config.get("alignment_epochs",30)) <= start_epoch:
            raise ValueError("Epoch budget must exceed restored epoch")
        checkpoints = output_dir / "checkpoints"
        checkpoints.mkdir()
        def save(path, epoch):
            torch.save({"epoch": epoch, "theta": theta.detach(), "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                        "python_rng": random.getstate(), "config": config, "representation": rep,
                        "training_digest": training_digest,"model_id":model_id,"seed":seed}, path)
        save(checkpoints / "initial.pt", start_epoch)
        for epoch in range(start_epoch+1, int(config.get("alignment_epochs", 30))+1):
            # Training-only minibatch sampling; both classes are guaranteed explicitly.
            limit = min(int(config.get("alignment_batch_size", 64)), len(y))
            first = [int(rng.choice(np.flatnonzero(y == c))) for c in (0, 1)]
            remaining = np.setdiff1d(np.arange(len(y)), first)
            indices = np.asarray(first + list(rng.choice(remaining, size=limit-2, replace=False)))
            states = circuit(a[indices], theta)
            matrix = torch.abs(states@states.conj().T)**2
            target = labels[indices, None]*labels[None, indices]
            if model_id == "quantum_metric_5q":
                mask = ~torch.eye(len(indices), dtype=torch.bool)
                desired = (target+1)/2
                loss = ((matrix[mask]-desired[mask])**2).mean()
            else:
                loss = -(matrix*target).sum() / (torch.linalg.vector_norm(matrix)*torch.linalg.vector_norm(target)).clamp_min(1e-12)
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite kernel objective")
            loss.backward()
            if theta.grad is None or not torch.isfinite(theta.grad).all():
                raise FloatingPointError("Invalid kernel gradient")
            grad_norm = float(torch.linalg.vector_norm(theta.grad))
            optimizer.step()
            history.append({"epoch": epoch, "train_loss_pre_update": float(loss.detach()), "gradient_norm": grad_norm,
                            "training_batch_indices": indices.tolist()})
            save(checkpoints / f"epoch_{epoch:04d}.pt", epoch)
            write_json(output_dir / "history.json", history)
        save(checkpoints / "final.pt", epoch)
        write_json(checkpoints / "aliases.json", {"best": "final.pt", "last": "final.pt", "best_rule": "fixed final epoch; no held-out selection"})
        mapping.theta = theta.detach().numpy().copy()
    fixed_features = model_id in ("random_quantum_features_5q", "static_reservoir_5q", "quantum_extreme_learning_5q")
    if fixed_features:
        train_embedding = mapping.features(x)
        design = train_embedding
        estimator = (RidgeClassifier(alpha=float(config.get("alpha", 1))) if model_id == "quantum_extreme_learning_5q" else
                     LogisticRegression(C=float(config.get("C", 1)), max_iter=2000, solver="lbfgs", random_state=seed))
    elif model_id == "kernel_projected_5q":
        train_embedding = mapping.features(x)
        design = rbf(train_embedding, train_embedding, float(config.get("gamma", 1/15)))
        estimator = SVC(kernel="precomputed", C=float(config.get("C", 1)), probability=False, random_state=seed)
    else:
        train_embedding = mapping.states(x)
        design = fidelity_kernel(train_embedding, train_embedding)
        if not np.allclose(np.diag(design), 1, atol=1e-8) or not np.allclose(design, design.T, atol=1e-8):
            raise FloatingPointError("Invalid fidelity kernel")
        estimator = None if model_id == "kernel_knn_5q" else SVC(kernel="precomputed", C=float(config.get("C", 1)), probability=False, random_state=seed)
    if estimator is not None:
        estimator.fit(design, y)
    np.save(output_dir / "training_embedding.npy", train_embedding)
    np.save(output_dir / "training_design.npy", design)
    bundle = KernelFitted(mapping, estimator, train_embedding, y.copy(), config)
    joblib.dump(bundle, output_dir / "model.joblib")
    write_json(output_dir / "metadata.json", {
        "schema": "qbench-quantum-map-v1", "model_id": model_id, "family": "kernel", "seed": seed,
        "config": config, "n_train": len(y), "backend": "pennylane.default.qubit", "shots": None,
        "precision": "float64/complex128", "n_qubits": 5, "quantum_trainable_parameters": 10 if trainable else 0,
        "kernel_learning": "pairwise fidelity MSE" if model_id == "quantum_metric_5q" else "unbalanced target alignment" if trainable else "none",
        "readout_params": None if estimator is None else estimator.get_params(),
        "fit_seconds": time.perf_counter()-start, "probability_available": model_id in ("kernel_knn_5q", "random_quantum_features_5q", "static_reservoir_5q"),
        "resume_mode": "epoch_boundary" if trainable else "restart_trial", "checkpoint_epoch_snapshots": "saved" if trainable else "not_applicable",
        "calibration": "none", "kernel_centering": "none", "preprocessing_fit_scope": "supplied training samples only",
    })
    return bundle


def load_kernel(output_dir):
    return joblib.load(Path(output_dir)/"model.joblib")
