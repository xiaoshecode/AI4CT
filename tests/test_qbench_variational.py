"""Synthetic numerical and artifact tests for the PennyLane reference models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai4ct.qbench.variational import (
    ANCHOR_IDS, VARIATIONAL_IDS, FeatureTransform, _apply_circuit,
    build_variational_model, fit_variational, load_paper_reference_state,
    load_variational,
)


HAVE_QUANTUM = all(importlib.util.find_spec(name) is not None for name in ("torch", "pennylane"))


class TransformTests(unittest.TestCase):
    def test_transform_fit_is_train_only_and_roundtrips(self):
        x = np.random.default_rng(4).normal(size=(15, 17))
        x[:, 3] = 7
        transform = FeatureTransform.fit("vqc_pca5_angle_5q", x)
        state = json.dumps(transform.to_dict(), sort_keys=True)
        np.testing.assert_allclose(transform.state["mean"], x.mean(axis=0))
        self.assertIn(3, transform.state["constant_columns"])
        test = np.full((2, 17), 1e8)
        predicted = transform.transform(test)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertLess(np.max(np.abs(predicted)), np.pi)
        self.assertEqual(state, json.dumps(transform.to_dict(), sort_keys=True))
        restored = FeatureTransform.from_dict(json.loads(state))
        np.testing.assert_array_equal(predicted, restored.transform(test))

    def test_amplitude_zero_vector_and_raw_anchor(self):
        x = np.zeros((10, 17))
        transform = FeatureTransform.fit("amplitude_vqc_5q", x)
        result = transform.transform(x)
        np.testing.assert_array_equal(result[:, 0], np.ones(10))
        np.testing.assert_array_equal(result[:, 1:], np.zeros((10, 31)))
        self.assertEqual(transform.state["zero_amplitude_training_rows"], 10)
        anchor = FeatureTransform.fit(ANCHOR_IDS[0], x + 12)
        np.testing.assert_array_equal(anchor.transform(x + 33), x + 33)


@unittest.skipUnless(HAVE_QUANTUM, "Optional PennyLane/Torch environment is unavailable")
class VariationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import pennylane as qml
        cls.torch, cls.qml = torch, qml
        cls.x = np.random.default_rng(18).normal(size=(12, 17))
        cls.y = np.arange(12) % 2

    def test_all_structures_batched_forward_parameter_gradients_and_norm(self):
        expected = {"reupload_full17_5q": 120, "reupload_full17_1q": 36,
                    "ttn_pca8_8q": 42, "mps_pca8_8q": 42, "qcnn_pca8_8q": 57,
                    "anchor_paper_pqc_5q": 0, "mera_pca8_8q": 66,
                    "gibbs_qbm_3q": 3, "gibbs_separable_3q": 3,
                    "dressed_classical_5d": 0}
        for model_id in VARIATIONAL_IDS:
            with self.subTest(model_id=model_id):
                transform = FeatureTransform.fit(model_id, self.x)
                model = build_variational_model(model_id, seed=7, config={"dtype": "float64"})
                x = self.torch.tensor(transform.transform(self.x[:3]), dtype=self.torch.float64)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (3, 2))
                self.assertTrue(self.torch.isfinite(logits).all())
                self.torch.nn.functional.cross_entropy(logits, self.torch.tensor([0, 1, 0])).backward()
                gradients = [p.grad for p in model.parameters()]
                self.assertTrue(all(g is not None and self.torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                qcount = 0 if model.angles is None else model.angles.numel()
                self.assertEqual(qcount, expected.get(model_id, 45))
                if model.angles is not None:
                    self.assertGreater(float(model.angles.grad.abs().sum()), 1e-9)
                if hasattr(model, "statevector"):
                    state = model.statevector(x).detach().numpy()
                    np.testing.assert_allclose((np.abs(state) ** 2).sum(axis=-1), 1, atol=1e-10)
                elif hasattr(model, "density_matrix"):
                    rho = model.density_matrix(x).detach().numpy()
                    np.testing.assert_allclose(np.trace(rho, axis1=-2, axis2=-1), 1, atol=1e-10)
                else:
                    self.assertEqual(model_id, "dressed_classical_5d")
                single = model(x[:1]).detach().numpy()
                np.testing.assert_allclose(single, logits[:1].detach().numpy(), atol=1e-10)

    def test_paper_anchor_matches_independent_frozen_implementation(self):
        from ai4ct.reproduction.models import build_torch_model
        for anchor, name in zip(ANCHOR_IDS, ("pqc", "qfm")):
            with self.subTest(anchor=anchor):
                original = build_torch_model(name, 17, seed=12).double()
                target = build_variational_model(anchor, seed=24, config={"dtype": "float64"})
                load_paper_reference_state(target, original.state_dict())
                x1 = self.torch.tensor(self.x[:2] * .05, dtype=self.torch.float64, requires_grad=True)
                x2 = x1.detach().clone().requires_grad_()
                logits1, logits2 = original(x1), target(x2)
                np.testing.assert_allclose(logits1.detach(), logits2.detach(), atol=1e-10)
                logits1[:, 1].sum().backward()
                logits2[:, 1].sum().backward()
                np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-10)
                with self.torch.no_grad():
                    encoded = original[:4](x1.detach())
                    old_state = original[4].statevector(encoded)
                    new_state = target.statevector(encoded, already_encoded=True)
                np.testing.assert_allclose(old_state, new_state, atol=1e-10)

    def test_trainable_input_and_angle_gradients_match_finite_difference(self):
        model = build_variational_model("dressed_vqc_5q", seed=5)
        x = self.torch.tensor(self.x[:1] * .1, dtype=self.torch.float64, requires_grad=True)
        self.assertTrue(self.torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-5, rtol=1e-4))
        value = model(x)[0, 1]
        value.backward()
        index = (0, 1, 1)
        analytic = float(model.angles.grad[index])
        original = float(model.angles.detach()[index])
        with self.torch.no_grad():
            model.angles[index] = original + 1e-6
            upper = float(model(x)[0, 1])
            model.angles[index] = original - 1e-6
            lower = float(model(x)[0, 1])
            model.angles[index] = original
        self.assertAlmostEqual(analytic, (upper - lower) / 2e-6, places=6)

    def test_qcnn_deferred_pooling_matches_measurement_branches(self):
        qml, torch = self.qml, self.torch
        initial = np.random.default_rng(3).normal(size=4) + 1j * np.random.default_rng(8).normal(size=4)
        initial /= np.linalg.norm(initial)
        angles = [.2, -.8, .3]
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device)
        def coherent():
            qml.StatePrep(initial, wires=[0, 1])
            qml.ctrl(qml.U3, control=1)(*angles, wires=0)
            return qml.expval(qml.PauliZ(0))

        unitary = qml.matrix(qml.U3(*angles, wires=0))
        # Independent projective branches: wire1 is measured and then ignored.
        branch0, branch1 = initial.reshape(2, 2)[:, 0], initial.reshape(2, 2)[:, 1]
        branch1 = unitary @ branch1
        expected = sum(abs(b[0]) ** 2 - abs(b[1]) ** 2 for b in (branch0, branch1))
        self.assertAlmostEqual(float(coherent()), float(expected), places=12)

    def test_separable_has_no_two_qubit_gates_and_iqp_entangles(self):
        qml = self.qml
        weights = self.torch.zeros((3, 5, 3), dtype=self.torch.float64)
        angles = self.torch.tensor([.2, .4, .6, .8, 1.], dtype=self.torch.float64)
        tape = qml.tape.make_qscript(lambda: _apply_circuit(qml, "separable_vqc_5q", angles, weights))()
        self.assertTrue(all(len(op.wires) == 1 for op in tape.operations))
        model = build_variational_model("iqp_vqc_5q", seed=3)
        got = model.quantum_features(angles[None]).detach().numpy()
        self.assertEqual(got.shape, (1, 5))

    def test_all_models_fit_save_every_epoch_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model_id in VARIATIONAL_IDS:
                with self.subTest(model_id=model_id):
                    path = Path(tmp) / model_id
                    fitted = fit_variational(model_id, self.x, self.y, seed=11, output_dir=path,
                                             config={"epochs": 1, "dtype": "float64"})
                    loaded = load_variational(path)
                    np.testing.assert_array_equal(fitted.predict_proba(self.x), loaded.predict_proba(self.x))
                    np.testing.assert_array_equal(fitted.predict_score(self.x), loaded.predict_score(self.x))
                    for name in ("initial.pt", "epoch_0001.pt", "last.pt", "final.pt"):
                        self.assertTrue((path / "checkpoints" / name).is_file())
                    self.assertEqual(json.loads((path / "metadata.json").read_text())["status"], "completed")
                    self.assertEqual(loaded.predict_proba(np.empty((0, 17))).shape, (0,))
                    with self.assertRaises(FileExistsError):
                        fit_variational(model_id, self.x, self.y, seed=11, output_dir=path, config={"epochs": 1})

    def test_resume_in_new_directory_matches_uninterrupted_trajectory(self):
        model_id = "vqc_pca5_angle_5q"
        cfg = {"epochs": 2, "batch_size": 5, "shuffle": True}
        with tempfile.TemporaryDirectory() as tmp:
            full = fit_variational(model_id, self.x, self.y, seed=8, output_dir=Path(tmp) / "full", config=cfg)
            short = fit_variational(model_id, self.x, self.y, seed=8, output_dir=Path(tmp) / "short", config={**cfg, "epochs": 1})
            source = short.output_dir / "checkpoints" / "last.pt"
            resumed = fit_variational(model_id, self.x, self.y, seed=8, output_dir=Path(tmp) / "resumed",
                                      config={**cfg, "resume_from": str(source)})
            np.testing.assert_array_equal(full.predict_proba(self.x), resumed.predict_proba(self.x))
            np.testing.assert_array_equal([r["train_loss"] for r in full.history], [r["train_loss"] for r in resumed.history])
            with self.assertRaises(ValueError):
                fit_variational(model_id, self.x[::-1], self.y[::-1], seed=8, output_dir=Path(tmp) / "bad_resume",
                                config={**cfg, "resume_from": str(source)})
            failed = json.loads((Path(tmp) / "bad_resume" / "metadata.json").read_text())
            self.assertEqual(failed["status"], "failed")

    def test_tie_predicts_fail_and_seed_repeats_initial_parameters(self):
        first = build_variational_model("vqc_pca5_angle_5q", seed=9)
        second = build_variational_model("vqc_pca5_angle_5q", seed=9)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(self.torch.equal(a, b))
        with tempfile.TemporaryDirectory() as tmp:
            fitted = fit_variational("vqc_pca5_angle_5q", self.x, self.y, seed=1,
                                     output_dir=tmp, config={"epochs": 1})
            with self.torch.no_grad():
                fitted.model.readout.weight.zero_()
                fitted.model.readout.bias.zero_()
            np.testing.assert_array_equal(fitted.predict_proba(self.x), np.full(len(self.x), .5))
            np.testing.assert_array_equal(fitted.predict(self.x), np.zeros(len(self.x)))


if __name__ == "__main__":
    unittest.main()
