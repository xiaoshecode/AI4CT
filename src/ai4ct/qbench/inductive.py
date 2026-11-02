"""Staged classical QUBO and exact interference-similarity adapters.

No hardware execution or quantum speed-up is claimed. QUBO uses a fixed training
dictionary and exhaustive classical search, not the full iterative QBoost paper.
The interference adapter uses the closed form of a six-qubit pair circuit, not
the original coherent training-index architecture. No module import writes files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np

from ai4ct.qbench.common import Representation, ensure_model_directory, sha256, validate_xy, write_json

MODEL_IDS = ("qboost_qubo_12_classical", "interference_distance_6q")
SCHEMA = "qbench-inductive-v1"


def _hash_array(value):
    value = np.ascontiguousarray(value)
    prefix = f"{value.dtype.str}:{value.shape}:".encode("ascii")
    return hashlib.sha256(prefix + value.tobytes()).hexdigest()


def _configuration(model_id, config):
    defaults = ({"max_weak_learners": 12, "lambda": .01, "quantiles": [.25, .5, .75]}
                if model_id == MODEL_IDS[0] else {"bias_amplitude": 1.0})
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported inductive configuration: {sorted(unknown)}")
    result = {**defaults, **dict(config)}
    if model_id == MODEL_IDS[0]:
        k = result["max_weak_learners"]
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 12:
            raise ValueError("Exact QUBO enumeration requires 1 <= max_weak_learners <= 12")
        result["lambda"] = float(result["lambda"])
        if not np.isfinite(result["lambda"]) or result["lambda"] < 0:
            raise ValueError("lambda must be nonnegative and finite")
        quantiles = np.asarray(result["quantiles"], dtype=float)
        if quantiles.ndim != 1 or not len(quantiles) or not np.isfinite(quantiles).all():
            raise ValueError("quantiles must be a nonempty finite list")
        if np.any((quantiles <= 0) | (quantiles >= 1)) or len(set(quantiles)) != len(quantiles):
            raise ValueError("quantiles must be distinct and strictly inside (0, 1)")
        result["quantiles"] = sorted(quantiles.tolist())
    else:
        result["bias_amplitude"] = float(result["bias_amplitude"])
        if result["bias_amplitude"] != 1.0:
            raise ValueError("This version fixes the positive bias amplitude to 1")
    return result


def stump_matrix(z, stumps):
    """Columns are +/-1, with >= threshold assigned the positive polarity."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all() or not stumps:
        raise ValueError("Finite 2D data and at least one stump are required")
    columns = []
    for stump in stumps:
        polarity = int(stump["polarity"])
        if polarity not in (-1, 1):
            raise ValueError("Stump polarity must be +/-1")
        if stump["kind"] == "constant":
            column = np.full(len(z), polarity, dtype=float)
        elif stump["kind"] == "threshold":
            column = polarity * np.where(z[:, int(stump["feature"])] >= float(stump["threshold"]), 1., -1.)
        else:
            raise ValueError("Unknown stump kind")
        columns.append(column)
    return np.column_stack(columns)


def select_stumps(z, signed_y, max_weak_learners=12, quantiles=(.25, .5, .75)):
    """Build/train-select a fixed dictionary; all quantiles and errors use train only."""
    candidates = []
    for polarity in (-1, 1):
        candidates.append({"kind": "constant", "feature": -1, "threshold": 0., "quantile": None, "polarity": polarity})
    for feature in range(z.shape[1]):
        for quantile in quantiles:
            threshold = float(np.quantile(z[:, feature], quantile, method="linear"))
            for polarity in (-1, 1):
                candidates.append({"kind": "threshold", "feature": feature, "threshold": threshold,
                                   "quantile": float(quantile), "polarity": polarity})
    h = stump_matrix(z, candidates)
    errors = np.mean(h != signed_y[:, None], axis=0)
    order = sorted(range(len(candidates)), key=lambda i: (
        float(errors[i]), candidates[i]["feature"],
        -1. if candidates[i]["quantile"] is None else candidates[i]["quantile"], candidates[i]["polarity"]))
    seen, selected = set(), []
    for index in order:
        key = h[:, index].astype(np.int8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        selected.append({**candidates[index], "train_error": float(errors[index])})
        if len(selected) == max_weak_learners:
            break
    return selected


def qubo_coefficients(h, signed_y, regularization=.01):
    """E(w)=1+sum_i Qii wi+sum_i<j Qij wi wj for binary w.

    Exactly expands mean((H w/K-y)^2)+(lambda/K)*sum(w).
    Only the upper triangle is populated; off-diagonal terms occur once.
    """
    h = np.asarray(h, dtype=float)
    y = np.asarray(signed_y, dtype=float)
    if h.ndim != 2 or not h.size or y.shape != (len(h),):
        raise ValueError("Invalid weak-learner matrix/labels")
    if not np.isin(h, [-1., 1.]).all() or not np.isin(y, [-1., 1.]).all():
        raise ValueError("H and signed_y must use +/-1")
    regularization = float(regularization)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Invalid regularization")
    n, k = h.shape
    a = h.T @ h / (n * k * k)
    b = -2 * (h.T @ y) / (n * k) + regularization / k
    q = np.triu(2 * a, 1)
    np.fill_diagonal(q, np.diag(a) + b)
    return q, 1.0


def enumerate_qubo(q, constant=1.0):
    """Classically enumerate <=12 bits; exact float64 objective, smallest-code ties."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not 1 <= len(q) <= 12:
        raise ValueError("Expected a square QUBO of 1 to 12 bits")
    if not np.isfinite(q).all() or not np.allclose(np.tril(q, -1), 0, rtol=0, atol=0):
        raise ValueError("Q must be finite and upper triangular")
    codes = np.arange(2 ** len(q), dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(len(q), dtype=np.uint64)) & 1).astype(np.float64)
    energies = np.einsum("bi,ij,bj->b", bits, q, bits) + float(constant)
    index = int(np.argmin(energies))
    return {"solution": bits[index].astype(np.int64), "integer_code": int(codes[index]),
            "minimum_energy": float(energies[index]), "all_energies": energies}


def amplitude_vectors(z):
    """17 standardized features plus fixed positive bias, padded to 32 amplitudes."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 17 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected finite n by 17 standardized features")
    result = np.zeros((len(z), 32), dtype=np.float64)
    result[:, :17] = z
    result[:, 17] = 1.0
    # Scaling before the norm also handles very large finite standardized values.
    result /= np.max(np.abs(result), axis=1, keepdims=True)
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def interference_acceptance(query_states, training_states):
    """Exact P(ancilla=0) after H on (|0>q+|1>t)/sqrt(2).

    This depends on the real signed overlap, not its square. Matrix rows and
    columns correspond to queries and retained training examples respectively.
    """
    queries, train = np.asarray(query_states, dtype=float), np.asarray(training_states, dtype=float)
    if queries.ndim != 2 or train.ndim != 2 or queries.shape[1] != train.shape[1] or not len(train):
        raise ValueError("Expected compatible nonempty state matrices")
    for array in (queries, train):
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("Interference states must be finite and unit normalized")
    return np.clip((1 + queries @ train.T) / 2, 0., 1.)


def interference_pass_probability(query_states, training_states, y_train):
    labels = np.asarray(y_train)
    if labels.shape != (len(training_states),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must align and use 0/1")
    acceptance = interference_acceptance(query_states, training_states)
    denominator = acceptance.sum(axis=1)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise FloatingPointError("Conditional Born probability undefined: zero postselection acceptance")
    return acceptance[:, labels == 1].sum(axis=1) / denominator


class InductiveFitted:
    def __init__(self, model_id, representation, payload):
        self.model_id, self.representation, self.payload = model_id, representation, payload

    def predict_proba(self, x):
        if self.model_id == MODEL_IDS[0]:
            return None
        query = amplitude_vectors(self.representation.transform(x))
        return interference_pass_probability(query, self.payload["training_states"], self.payload["training_labels"])

    def predict_score(self, x):
        if self.model_id == MODEL_IDS[0]:
            h = stump_matrix(self.representation.transform(x), self.payload["stumps"])
            return h @ self.payload["solution"] / len(self.payload["stumps"])
        return 2 * self.predict_proba(x) - 1


def fit_inductive(model_id, x_train, y_train, *, seed, output_dir, config):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown inductive model {model_id}")
    x, y = validate_xy(x_train, y_train)
    config = _configuration(model_id, config)
    output_dir = ensure_model_directory(output_dir)
    started = time.perf_counter()
    rep = Representation().fit(x)
    z = rep.transform(x)
    payload = {}
    details = {}
    if model_id == MODEL_IDS[0]:
        signed_y = 2 * y - 1
        stumps = select_stumps(z, signed_y, config["max_weak_learners"], config["quantiles"])
        h = stump_matrix(z, stumps)
        q, constant = qubo_coefficients(h, signed_y, config["lambda"])
        solution = enumerate_qubo(q, constant)
        direct = float(np.mean((h @ solution["solution"] / len(stumps) - signed_y) ** 2)
                       + config["lambda"] / len(stumps) * solution["solution"].sum())
        if not np.isclose(direct, solution["minimum_energy"], atol=1e-12, rtol=0):
            raise AssertionError("QUBO expansion and direct objective disagree")
        payload = {"stumps": stumps, "qubo_upper": q, "qubo_constant": constant, **solution,
                   "training_weak_matrix": h, "training_labels": y, "training_standardized": z}
        details = {"stumps": stumps, "qubo_upper": q.tolist(), "qubo_constant": constant,
                   "solution": solution["solution"].tolist(), "solution_integer_code": solution["integer_code"],
                   "minimum_energy": solution["minimum_energy"], "direct_objective": direct,
                   "enumerated_states": len(solution["all_energies"]), "n_selected_stumps": int(solution["solution"].sum()),
                   "n_weak_learners": len(stumps), "quantum_hardware_used": False,
                   "backend": "numpy_float64_exhaustive_binary_search",
                   "adaptation": "Fixed-dictionary QUBO classical exact control; not full iterative QBoost or quantum annealing",
                   "score_type": "signed_ensemble_margin", "training_weak_matrix_sha256": _hash_array(h)}
    else:
        states = amplitude_vectors(z)
        payload = {"training_states": states, "training_labels": y, "training_standardized": z}
        details = {"training_states_sha256": _hash_array(states), "n_amplitudes": 32,
                   "reference_pair_circuit_qubits": 6, "quantum_hardware_used": False,
                   "backend": "numpy_float64_closed_form_equivalent_to_pairwise_statevector",
                   "adaptation": "Pairwise exact interference simulation; coherent training-index register is not implemented",
                   "score_type": "2_times_conditional_pass_born_probability_minus_1",
                   "class_weighting": "uniform_per_training_example; no class balancing",
                   "amplitude_encoding": "StandardScaler train-only; append +1 then zero-pad to32 and normalize",
                   "postselection_weight": "(1+signed_real_overlap)/2, not fidelity squared"}
    artifact = {"schema": SCHEMA, "model_id": model_id, "config": config, "seed": int(seed),
                "representation": rep, "payload": payload}
    joblib.dump(artifact, output_dir / "model.joblib")
    write_json(output_dir / "metadata.json", {
        "schema": SCHEMA, "model_id": model_id, "family": "inductive", "seed": int(seed),
        "seed_used_for_fitting": False, "deterministic_fit": True, "config": config,
        "n_train": len(y), "fit_seconds": time.perf_counter() - started,
        "probability_available": model_id == MODEL_IDS[1], "calibration": "none",
        "threshold_rule": "Pass iff score>0; conditional probability>0.5; ties are Fail",
        "resume_mode": "restart_trial", "checkpoint_epoch_snapshots": "not_applicable",
        "feature_representation": "train_standard_scaler", "training_features_sha256": _hash_array(x),
        "training_standardized_sha256": _hash_array(z), "training_labels_sha256": _hash_array(y),
        "artifact_sha256": sha256(output_dir / "model.joblib"), **details})
    return InductiveFitted(model_id, rep, payload)


def load_inductive(output_dir):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("model_id") not in MODEL_IDS:
        raise ValueError("Unsupported inductive artifact schema/model")
    if sha256(output_dir / "model.joblib") != metadata["artifact_sha256"]:
        raise ValueError("Inductive artifact hash mismatch")
    artifact = joblib.load(output_dir / "model.joblib")
    if artifact.get("schema") != SCHEMA or artifact["model_id"] != metadata["model_id"] or artifact["config"] != metadata["config"]:
        raise ValueError("Inductive metadata and serialized parameters disagree")
    return InductiveFitted(artifact["model_id"], artifact["representation"], artifact["payload"])
