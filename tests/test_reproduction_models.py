"""Quantum numerical oracles and saved-prediction regression checks (synthetic data)."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ai4ct.reproduction.models import (
    _cnot, _read_z, _rotation, build_pqc_circuit, build_qfm_circuit,
    build_torch_model, train_model, load_model,
)

try:
    import torch
except ImportError:
    torch = None


def dense_gate(gate, qubit, n):
    """Independent Kronecker-product oracle; most-significant qubit first."""
    result = np.array([[1]], dtype=complex)
    for q in range(n):
        result = np.kron(result, gate if q == qubit else np.eye(2))
    return result


def dense_cnot(control, target, n):
    result = np.zeros((2**n, 2**n), dtype=complex)
    for column in range(2**n):
        bits = list(format(column, f"0{n}b"))
        if bits[control] == "1":
            bits[target] = "1" if bits[target] == "0" else "0"
        result[int("".join(bits), 2), column] = 1
    return result


def rotation_matrix(angle, axis):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    if axis == "x":
        return np.array([[c, -1j*s], [-1j*s, c]])
    if axis == "y":
        return np.array([[c, -s], [s, c]])
    return np.diag([np.exp(-1j*angle/2), np.exp(1j*angle/2)])


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class QuantumCircuitTests(unittest.TestCase):
    def test_rotation_entanglement_and_measurement_share_qubit_order(self):
        state = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
        state = _rotation(state, torch.tensor([math.pi], dtype=torch.float64), 0, 2, "y")
        np.testing.assert_allclose(state.numpy(), [[0, 0, 1, 0]], atol=1e-12)
        state = _cnot(state, 0, 1, 2)
        np.testing.assert_allclose(state.numpy(), [[0, 0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(_read_z(state, 2).numpy(), [[-1, -1]], atol=1e-12)

    def test_single_qubit_pqc_matches_analytic_expectation(self):
        x = torch.tensor([[0.13, 0.37], [-0.4, 1.2]], dtype=torch.float64)
        got = build_pqc_circuit(1)(x).detach().numpy()[:, 0]
        np.testing.assert_allclose(got, -np.sin(x.numpy().sum(axis=1)), atol=1e-12)

    def test_pqc_matches_independent_dense_matrix_oracle(self):
        n = 3
        angles = np.random.default_rng(7).normal(size=2*n)
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for q in range(n):
            state = dense_gate(h, q, n) @ state
            state = dense_gate(rotation_matrix(angles[q], "y"), q, n) @ state
        for control in range(n - 1):
            for target in range(control + 1, n):
                state = dense_cnot(control, target, n) @ state
        for q in range(n):
            state = dense_gate(rotation_matrix(angles[n + q], "y"), q, n) @ state
        got = build_pqc_circuit(n).statevector(torch.tensor(angles[None], dtype=torch.float64))
        np.testing.assert_allclose(got.numpy()[0], state, atol=1e-12)
        self.assertAlmostEqual(float(got.square().sum()), 1, places=12)

    def test_qfm_matches_independent_dense_matrix_oracle(self):
        n = 2
        values = np.array([0.19, -0.27])
        circuit = build_qfm_circuit(n, feature_reps=2, ansatz_reps=2).double()
        angles = np.random.default_rng(8).normal(size=(2, n, 3))
        with torch.no_grad():
            circuit.angles.copy_(torch.tensor(angles))
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1
        for _ in range(2):
            for q in range(n):
                state = dense_gate(h, q, n) @ state
                state = dense_gate(np.diag([1, np.exp(2j * values[q])]), q, n) @ state
            state = dense_cnot(0, 1, n) @ state
            phi = (np.pi - values[0]) * (np.pi - values[1])
            state = dense_gate(np.diag([1, np.exp(2j*phi)]), 1, n) @ state
            state = dense_cnot(0, 1, n) @ state
        for layer in range(2):
            for q in range(n):
                for index, axis in enumerate(("x", "y", "z")):
                    state = dense_gate(rotation_matrix(angles[layer, q, index], axis), q, n) @ state
            state = dense_cnot(0, 1, n) @ state
        got = circuit.statevector(torch.tensor(values[None], dtype=torch.float64))
        np.testing.assert_allclose(got.detach().numpy()[0], state, atol=1e-12)
        self.assertAlmostEqual(float(got.detach().abs().square().sum()), 1, places=12)

    def test_qfm_has_exactly_three_rotation_layers(self):
        circuit = build_qfm_circuit()
        self.assertEqual(sum(p.numel() for p in circuit.parameters()), 45)

    def test_quantum_input_gradients_match_finite_differences(self):
        for circuit, width in ((build_pqc_circuit(2), 4), (build_qfm_circuit(2, 2, 1).double(), 2)):
            x = torch.linspace(0.1, 0.7, width, dtype=torch.float64).reshape(1, width).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(circuit, (x,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_qfm_trainable_angle_gradient_matches_finite_difference(self):
        circuit = build_qfm_circuit(2, 2, 1).double()
        x = torch.tensor([[0.15, 0.39]], dtype=torch.float64)
        circuit(x).sum().backward()
        analytic = float(circuit.angles.grad[0, 0, 0])
        original = float(circuit.angles.detach()[0, 0, 0])
        eps = 1e-6
        with torch.no_grad():
            circuit.angles[0, 0, 0] = original + eps
            upper = float(circuit(x).sum())
            circuit.angles[0, 0, 0] = original - eps
            lower = float(circuit(x).sum())
        self.assertAlmostEqual(analytic, (upper-lower)/(2*eps), places=5)

    def test_seed_controls_initialization_before_any_model_is_built(self):
        for name in ("mlp", "pqc", "qfm"):
            left = build_torch_model(name, 7, seed=29)
            right = build_torch_model(name, 7, seed=29)
            for a, b in zip(left.state_dict().values(), right.state_dict().values()):
                self.assertTrue(torch.equal(a, b), name)


class FittedModelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(11, 7)).astype(np.float32)
        self.y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    def test_gbdt_saved_predictions_roundtrip(self):
        result = train_model("gbdt", self.x, self.y, config={"n_estimators": 3})
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            restored = load_model(directory)
            np.testing.assert_array_equal(result.predict_proba(self.x), restored.predict_proba(self.x))
            self.assertTrue((Path(directory) / "metadata.json").is_file())

    def test_probability_tie_uses_class_zero_like_argmax(self):
        result = train_model("gbdt", self.x, self.y, config={"n_estimators": 2})
        with patch.object(result, "predict_proba", return_value=np.array([0.4, 0.5, 0.6])):
            np.testing.assert_array_equal(result.predict(self.x[:3]), [0, 0, 1])

    @unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
    def test_neural_training_keeps_remainder_and_roundtrips_predictions(self):
        for name in ("mlp", "pqc", "qfm"):
            with self.subTest(model=name):
                result = train_model(name, self.x, self.y, seed=12,
                                     config={"epochs": 2, "batch_size": 4})
                self.assertEqual([r["samples_seen"] for r in result.history], [11, 11])
                probability = result.predict_proba(self.x)
                self.assertEqual(probability.shape, (11,))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                with tempfile.TemporaryDirectory() as directory:
                    result.save(directory)
                    restored = load_model(directory)
                    np.testing.assert_allclose(probability, restored.predict_proba(self.x), atol=1e-7, rtol=0)

    def test_input_validation_rejects_ambiguous_labels_and_nonfinite_features(self):
        with self.assertRaises(ValueError):
            train_model("gbdt", self.x, np.ones(len(self.x)))
        x = self.x.copy()
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_model("gbdt", x, self.y)


if __name__ == "__main__":
    unittest.main()
