"""Shared validation, train-only representations and local artifact utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .atomic import replace_with_retry
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_x(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 17 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a finite, nonempty n by 17 raw feature matrix")
    return x


def validate_xy(x, y):
    x = validate_x(x)
    y = np.asarray(y)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Training labels must be aligned binary values with both classes")
    return x, y.astype(np.int64)


def seed_everything(seed, threads=1):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Representation:
    """A fitted training-only scaler and optional PCA; no refit on inference."""
    def __init__(self, components=None, angle=False, raw=False):
        self.components, self.angle, self.raw = components, angle, raw

    def fit(self, x):
        x = validate_x(x)
        self.n_features_in_ = x.shape[1]
        if self.raw:
            return self
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if self.components is not None:
            self.pca = PCA(n_components=self.components, whiten=False, svd_solver="full").fit(z)
            pc = self.pca.transform(z)
            self.pc_scale = pc.std(axis=0, ddof=0)
            self.pc_scale[self.pc_scale < 1e-12] = 1.0
        return self

    def transform(self, x):
        x = validate_x(x)
        if self.raw:
            return x.copy()
        z = self.scaler.transform(x)
        if self.components is not None:
            z = self.pca.transform(z) / self.pc_scale
        return 2 * np.arctan(z) if self.angle else z


def ensure_model_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "metadata.json").exists() or (path / "model.joblib").exists():
        raise FileExistsError("Model artifact exists; use a new trial directory")
    return path
