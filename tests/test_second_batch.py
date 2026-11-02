"""Independent synthetic checks; run only with staged-src PYTHONPATH."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm
import torch

from ai4ct.qbench.second_batch import SECOND_BATCH_IDS
from ai4ct.qbench.variational import (FeatureTransform, build_variational_model,
                                     fit_variational, load_variational)


class SecondBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.default_rng(79).normal(size=(12, 17))
        cls.y = np.arange(12) % 2

    def test_gibbs_matches_independent_scipy_matrix_exponential(self):
        for name in SECOND_BATCH_IDS[:2]:
            with self.subTest(name=name):
                model = build_variational_model(name, seed=5)
                x = torch.tensor(self.x[:3], dtype=torch.float64)
                hamiltonian = model.hamiltonian(x).detach().numpy()
                rho = model.density_matrix(x).detach().numpy()
                for h, actual in zip(hamiltonian, rho):
                    expected = expm(-h)
                    expected /= np.trace(expected)
                    np.testing.assert_allclose(actual, expected, atol=2e-12)
                    self.assertAlmostEqual(float(np.trace(actual)), 1., places=12)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(actual).min()), -1e-12)
                p = model.class_probabilities(x).detach().numpy()
                np.testing.assert_allclose(p[:, 1], np.diagonal(rho, axis1=-2, axis2=-1)[:, 4:].sum(-1), atol=2e-12)
                np.testing.assert_allclose(p.sum(-1), 1, atol=1e-12)

    def test_uncoupled_gibbs_analytic_factorization_and_hidden_invariance(self):
        model = build_variational_model("gibbs_separable_3q", seed=5)
        x = torch.tensor(self.x[:3], dtype=torch.float64)
        fields = model.fields(x).detach().numpy()[:, 0]
        gamma = float(torch.nn.functional.softplus(model.angles[0]).detach())
        radius = np.sqrt(fields ** 2 + gamma ** 2)
        expected = (1 - fields / radius * np.tanh(radius)) / 2
        initial = model.class_probabilities(x)[:, 1]
        np.testing.assert_allclose(initial.detach(), expected, atol=2e-12)
        initial.sum().backward()
        self.assertLess(float(model.fields.weight.grad[1:].abs().max()), 1e-12)
        self.assertLess(float(model.angles.grad[1:].abs().max()), 1e-12)
        with torch.no_grad():
            model.fields.bias[1:] += 4
            model.fields.weight[1:] *= 3
            model.angles[1:] += 2
        np.testing.assert_allclose(model.class_probabilities(x)[:, 1].detach(), expected, atol=2e-12)

    def test_gibbs_input_and_hamiltonian_parameter_derivatives(self):
        model = build_variational_model("gibbs_qbm_3q", seed=8)
        x = torch.tensor(self.x[:1] * .1, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=2e-6, rtol=2e-4))
        model(x)[0, 1].backward()
        for parameter, index in ((model.couplings, 0), (model.angles, 0), (model.fields.weight, (0, 2))):
            actual = float(parameter.grad[index])
            original = float(parameter.detach()[index])
            with torch.no_grad():
                parameter[index] = original + 1e-6
                upper = float(model(x)[0, 1])
                parameter[index] = original - 1e-6
                lower = float(model(x)[0, 1])
                parameter[index] = original
            self.assertAlmostEqual(actual, (upper - lower) / 2e-6, places=6)

    def test_control_dimensions_and_mera_norm_gradients(self):
        expected = {"dressed_classical_5d":151, "dressed_separable_5q":147,
                    "gibbs_qbm_3q":60, "gibbs_separable_3q":57,
                    "mera_pca8_8q":70}
        for name in SECOND_BATCH_IDS:
            with self.subTest(name=name):
                transform = FeatureTransform.fit(name, self.x)
                model = build_variational_model(name, seed=4)
                x = torch.tensor(transform.transform(self.x[:2]), dtype=torch.float64)
                result = model(x)
                self.assertEqual(tuple(result.shape), (2, 2))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])
                result.sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                if name == "mera_pca8_8q":
                    self.assertEqual(model.angles.numel(), 66)
                    state = model.statevector(x)
                    np.testing.assert_allclose(state.detach().abs().square().sum(-1), 1, atol=1e-11)
        quantum = build_variational_model("dressed_vqc_5q", seed=4)
        separable = build_variational_model("dressed_separable_5q", seed=4)
        for key in quantum.state_dict():
            torch.testing.assert_close(quantum.state_dict()[key], separable.state_dict()[key], atol=0, rtol=0)

    def test_all_five_fit_reload_and_resume_match_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in SECOND_BATCH_IDS:
                with self.subTest(name=name):
                    root = Path(directory) / name
                    base = {"epochs": 2, "batch_size": 5, "shuffle": True}
                    full = fit_variational(name, self.x, self.y, seed=9, output_dir=root / "full", config=base)
                    short = fit_variational(name, self.x, self.y, seed=9, output_dir=root / "short", config={**base,"epochs":1})
                    resumed = fit_variational(name, self.x, self.y, seed=9, output_dir=root / "resumed",
                                              config={**base,"resume_from":str(short.output_dir / "checkpoints" / "last.pt")})
                    loaded = load_variational(full.output_dir)
                    np.testing.assert_array_equal(full.predict_proba(self.x), loaded.predict_proba(self.x))
                    np.testing.assert_array_equal(full.predict_proba(self.x), resumed.predict_proba(self.x))
                    for checkpoint in ("initial.pt","epoch_0001.pt","epoch_0002.pt","best.pt","last.pt","final.pt"):
                        self.assertTrue((full.output_dir / "checkpoints" / checkpoint).exists())
                    meta = json.loads((full.output_dir / "metadata.json").read_text(encoding="utf-8"))
                    if name.startswith("gibbs_"):
                        self.assertEqual(meta["backend"], "torch.linalg.matrix_exp")
                        counts = meta["parameter_counts"]
                        self.assertIsNone(counts["quantum"])
                        self.assertIsNone(counts["quantum_circuit_gate_parameters"])
                        self.assertEqual(counts["fields"], 54)
                        self.assertEqual(counts["transverse"], 3)
                        self.assertEqual(counts["couplings"], 3 if name == "gibbs_qbm_3q" else 0)
                        self.assertEqual(counts["hamiltonian_total"], counts["total"])
                    elif name == "dressed_classical_5d":
                        self.assertEqual(meta["parameter_counts"]["quantum"], 0)
                    self.assertEqual(set(meta["implementation_source_sha256"]), {"variational.py", "second_batch.py"})


if __name__ == "__main__":
    unittest.main()
