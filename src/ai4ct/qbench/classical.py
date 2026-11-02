"""Classical controls with complete fitted parameters and explicit score types."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .common import Representation, ensure_model_directory, seed_everything, validate_xy, write_json

MODEL_IDS = ("always_pass", "logistic", "linear_svm", "rbf_svm", "gbdt", "random_forest", "small_mlp", "rff_linear")


def _network():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(17, 23), torch.nn.LeakyReLU(0.01),
                               torch.nn.Linear(23, 10), torch.nn.LeakyReLU(0.01), torch.nn.Linear(10, 2)).double()


class ClassicalFitted:
    def __init__(self, model_id, representation, estimator=None, rff=None, network=None):
        self.model_id, self.representation, self.estimator = model_id, representation, estimator
        self.rff, self.network = rff, network

    def _x(self, x):
        z = self.representation.transform(x)
        return self.rff.transform(z) if self.rff is not None else z

    def predict_proba(self, x):
        z = self._x(x)
        if self.model_id == "always_pass":
            return np.ones(len(z))
        if self.network is not None:
            import torch
            with torch.no_grad():
                return self.network(torch.as_tensor(z, dtype=torch.float64)).softmax(-1)[:, 1].numpy()
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(z)[:, list(self.estimator.classes_).index(1)]
        return None

    def predict_score(self, x):
        if self.model_id in ("linear_svm", "rbf_svm"):
            return np.asarray(self.estimator.decision_function(self._x(x)))
        p = self.predict_proba(x)
        return p - 0.5


def fit_classical(model_id, x_train, y_train, *, seed, output_dir, config):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown classical model {model_id}")
    x, y = validate_xy(x_train, y_train)
    output_dir = ensure_model_directory(output_dir)
    config = dict(config)
    start = time.perf_counter()
    rep = Representation().fit(x)
    z = rep.transform(x)
    rff = estimator = network = None
    if model_id == "logistic":
        estimator = LogisticRegression(C=float(config.get("C", 1)), max_iter=2000, solver="lbfgs", random_state=seed)
    elif model_id in ("linear_svm", "rbf_svm"):
        estimator = SVC(kernel="linear" if model_id == "linear_svm" else "rbf", C=float(config.get("C", 1)),
                        gamma=config.get("gamma", "scale"), probability=False, random_state=seed)
    elif model_id == "gbdt":
        estimator = GradientBoostingClassifier(n_estimators=int(config.get("n_estimators", 200)), max_depth=3,
                                               learning_rate=.2, subsample=.7, random_state=seed)
    elif model_id == "random_forest":
        estimator = RandomForestClassifier(n_estimators=int(config.get("n_estimators", 200)), max_depth=None,
                                           min_samples_leaf=2, n_jobs=1, random_state=seed)
    elif model_id == "rff_linear":
        rff = RBFSampler(gamma=float(config.get("gamma", 1/17)), n_components=int(config.get("n_components", 80)), random_state=seed).fit(z)
        z = rff.transform(z)
        estimator = LogisticRegression(C=float(config.get("C", 1)), max_iter=2000, solver="lbfgs", random_state=seed)
    elif model_id == "small_mlp":
        import torch
        seed_everything(seed, int(config.get("threads", 1)))
        network = _network()
        optimizer = torch.optim.Adam(network.parameters(), lr=float(config.get("lr", .005)), weight_decay=0)
        tx, ty = torch.as_tensor(z, dtype=torch.float64), torch.as_tensor(y, dtype=torch.int64)
        training_digest = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        start_epoch = 0
        if config.get("resume_from"):
            previous = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
            if previous.get("training_digest") != training_digest or previous.get("seed") != seed:
                raise ValueError("Resume data/seed mismatch")
            if float(previous["configuration"].get("lr", .005)) != float(config.get("lr", .005)):
                raise ValueError("Resume optimizer configuration mismatch")
            network.load_state_dict(previous["state_dict"])
            optimizer.load_state_dict(previous["optimizer"])
            torch.set_rng_state(previous["torch_rng"])
            np.random.set_state(previous["numpy_rng"])
            random.setstate(previous["python_rng"])
            start_epoch = int(previous["epoch"])
        if int(config.get("epochs",30)) <= start_epoch:
            raise ValueError("Epoch budget must exceed the restored epoch")
        checkpoints = output_dir / "checkpoints"
        checkpoints.mkdir()
        def save(path, epoch):
            torch.save({"epoch": epoch, "state_dict": network.state_dict(), "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate(), "sampler": "deterministic_full_batch",
                        "configuration": config, "representation": rep, "training_digest": training_digest,
                        "seed": seed}, path)
        save(checkpoints / "initial.pt", start_epoch)
        history = []
        for epoch in range(start_epoch + 1, int(config.get("epochs", 30)) + 1):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(network(tx), ty)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classical neural loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite classical neural gradient")
            optimizer.step()
            history.append({"epoch": epoch, "train_loss_pre_update": float(loss.detach())})
            save(checkpoints / f"epoch_{epoch:04d}.pt", epoch)
            write_json(output_dir / "history.json", history)
        save(checkpoints / "final.pt", epoch)
        write_json(checkpoints / "aliases.json", {"last": "final.pt", "best": "final.pt", "best_rule": "fixed final epoch; no validation selection"})
    if estimator is not None:
        estimator.fit(z, y)
    joblib.dump({"representation": rep, "estimator": estimator, "rff": rff}, output_dir / "model.joblib")
    write_json(output_dir / "metadata.json", {
        "schema": "qbench-classical-v1", "model_id": model_id, "family": "classical", "seed": seed,
        "config": config, "estimator_params": None if estimator is None else estimator.get_params(),
        "n_train": len(y), "fit_seconds": time.perf_counter() - start,
        "probability_available": model_id not in ("linear_svm", "rbf_svm"),
        "resume_mode": "epoch_boundary" if network is not None else "restart_trial",
        "checkpoint_epoch_snapshots": "saved" if network is not None else "not_applicable",
        "feature_representation": "train_standard_scaler", "calibration": "none",
    })
    return ClassicalFitted(model_id, rep, estimator, rff, network)


def load_classical(output_dir):
    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    data = joblib.load(output_dir / "model.joblib")
    network = None
    if meta["model_id"] == "small_mlp":
        import torch
        network = _network()
        checkpoint = torch.load(output_dir / "checkpoints/final.pt", map_location="cpu", weights_only=False)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
    return ClassicalFitted(meta["model_id"], data["representation"], data["estimator"], data["rff"], network)
