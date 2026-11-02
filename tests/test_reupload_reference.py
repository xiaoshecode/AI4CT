"""Source-derived oracle and synthetic-only fit/resume checks."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pennylane as qml
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staged_reupload_reference", ROOT / "src/ai4ct/qbench/reupload_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fixtures = Path(__file__).parent / "fixtures"
upstream_tree = ast.parse((fixtures / "data_reuploading_reference.py").read_text(encoding="utf-8"))
broadcast_spec = importlib.util.spec_from_file_location("historical_broadcast", fixtures / "broadcast_reference.py")
historical = importlib.util.module_from_spec(broadcast_spec)
broadcast_spec.loader.exec_module(historical)


def oracle(model):
    classes = {"base": "DataReuploadingClassifier", "no_cost": "DataReuploadingClassifier",
               "no_scaling": "DataReuploadingClassifierNoScaling",
               "no_trainable_embedding": "DataReuploadingClassifierNoTrainableEmbedding",
               "separable": "DataReuploadingClassifierSeparable",
               "matched_no_scaling": "DataReuploadingClassifier", "matched_separable": "DataReuploadingClassifier"}
    definition = next(node for node in upstream_tree.body if isinstance(node, ast.ClassDef) and node.name == classes[model.variant])
    construct = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "construct_model")
    circuit = copy.deepcopy(next(node for node in construct.body if isinstance(node, ast.FunctionDef) and node.name == "circuit"))
    circuit.decorator_list = []

    class Proxy:
        def __getattr__(self, name):
            return getattr(qml, name)

        def broadcast(self, unitary, wires, pattern):
            if model.variant == "matched_separable":
                return None
            return historical.broadcast(unitary, wires=wires, pattern=pattern)

    namespace = {"qml": Proxy(), "self": SimpleNamespace(n_layers=model.n_layers, n_qubits_=model.n_qubits)}
    exec(compile(ast.Module(body=[circuit], type_ignores=[]), "official_circuit_oracle", "exec"), namespace)
    node = qml.QNode(namespace["circuit"], qml.device("default.qubit", wires=model.n_qubits), interface="torch", diff_method="backprop")
    return node


class ReuploadReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.x = np.random.default_rng(8).normal(size=(8, 17))
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    def tearDown(self):
        self.temp.cleanup()

    def test_all_seven_broadcast_forward_match_official_source_oracle(self):
        values = torch.tensor(np.pad(self.x[:3], ((0, 0), (0, 1))), dtype=torch.float64)
        for variant in module.MODEL_VARIANTS.values():
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                model = module.ReferenceCircuit(variant, n_layers=3)
                parameters = {"thetas": model.thetas, "omegas": model.omegas, "alphas": model.alphas}
                reference = oracle(model)
                expected = torch.stack([torch.stack(reference(parameters, row)) for row in values])
                torch.testing.assert_close(model(values), expected, rtol=1e-11, atol=1e-11)

    def test_weighted_loss_and_nocost_sum_definition(self):
        z = torch.tensor([[.2, .9, -.3, .1, .2, .3], [-.4, .2, .6, -.3, .1, .1]], dtype=torch.float64)
        labels = torch.tensor([0., 1.], dtype=torch.float64)
        base = module.ReferenceCircuit("base")
        expected = .5 * ((((1-z[:, 0])/2-(1-labels))**2 + ((1+z[:, 0])/2-labels)**2).mean())
        torch.testing.assert_close(base.loss(z, labels), expected)
        nocost = module.ReferenceCircuit("no_cost")
        torch.testing.assert_close(nocost.loss(z, labels), ((1-(2*labels-1)*z.sum(dim=1))/2).mean())
        self.assertEqual(float(nocost.loss(torch.ones(2, 6), torch.ones(2))), -2.5)

    def test_gradient_matches_central_difference_and_omega_still_trains(self):
        torch.manual_seed(17)
        model = module.ReferenceCircuit("base", n_layers=1)
        x = torch.tensor(np.pad(self.x[:2], ((0, 0), (0, 1))), dtype=torch.float64)
        y = torch.tensor(self.y[:2], dtype=torch.float64)
        loss = model.loss(model(x), y)
        loss.backward()
        gradient = float(model.thetas.grad[0, 0, 1])
        with torch.no_grad():
            original = float(model.thetas[0, 0, 1])
            model.thetas[0, 0, 1] = original + 1e-6
            plus = float(model.loss(model(x), y))
            model.thetas[0, 0, 1] = original - 1e-6
            minus = float(model.loss(model(x), y))
            model.thetas[0, 0, 1] = original
        self.assertAlmostEqual(gradient, (plus-minus)/2e-6, places=7)
        moved = module.ReferenceCircuit("no_trainable_embedding", n_layers=1)
        moved.loss(moved(x), y).backward()
        self.assertIsInstance(moved.omegas, torch.nn.Parameter)
        self.assertGreater(float(moved.omegas.grad.abs().sum()), 0)

    def test_single_factor_controls_and_compound_official_variants(self):
        x = torch.tensor(np.pad(self.x[:2], ((0, 0), (0, 1))), dtype=torch.float64)
        torch.manual_seed(42)
        base = module.ReferenceCircuit("base", n_layers=2)
        with torch.no_grad():
            base.omegas.fill_(1)
        fixed = module.ReferenceCircuit("matched_no_scaling", n_layers=2)
        fixed.load_state_dict(base.state_dict())
        torch.testing.assert_close(base(x), fixed(x), rtol=0, atol=0)
        self.assertNotIn("omegas", dict(fixed.named_parameters()))
        official = module.ReferenceCircuit("no_scaling", n_layers=2)
        with torch.no_grad():
            official.thetas.copy_(base.thetas)
        self.assertGreater(float((official(x)-base(x)).abs().max()), 1e-4)
        no_cz = module.ReferenceCircuit("matched_separable", n_layers=2)
        compressed = module.ReferenceCircuit("separable", n_layers=2)
        compressed.load_state_dict(no_cz.state_dict())
        self.assertGreater(float((compressed(x)-no_cz(x)).abs().max()), 1e-4)

    def test_preprocessor_train_only_no_clip_and_same_for_variants(self):
        transform = module.InputTransform.fit(self.x)
        before = transform.to_dict()
        extremes = self.x[:1] + 100
        self.assertTrue(np.any(np.abs(transform.transform(extremes)[:, :17]) > np.pi/2))
        self.assertEqual(before, transform.to_dict())
        self.assertTrue(np.array_equal(transform.transform(self.x)[:, -1], np.zeros(len(self.x))))
        np.testing.assert_array_equal(module.InputTransform.from_dict(before).transform(self.x), transform.transform(self.x))

    def test_single_readout_light_cone_is_explicit(self):
        x = torch.tensor(np.pad(self.x[:2], ((0, 0), (0, 1))), dtype=torch.float64)
        for variant in module.MODEL_VARIANTS.values():
            model = module.ReferenceCircuit(variant, n_layers=3, observable_type="single")
            support = module.input_support(variant, 3, 1)
            self.assertEqual(support, list(range(3 if variant in {"separable", "matched_separable"} else 12)))
            changed = x.clone()
            changed[:, [i for i in range(17) if i not in support]] += .71
            torch.testing.assert_close(model(x)[:, 0], model(changed)[:, 0], rtol=1e-11, atol=1e-11)

    def test_all_seven_fit_parameter_snapshots_reload_and_resume(self):
        preprocessing = None
        for i, model_id in enumerate(module.MODEL_VARIANTS):
            with self.subTest(model=model_id):
                options = {"epochs": 2, "n_layers": 1}
                full_path = self.root / f"full_{i}"
                full = module.fit_reupload(model_id, self.x, self.y, seed=17, output_dir=full_path, config=options)
                loaded = module.load_reupload(full_path)
                np.testing.assert_array_equal(full.predict_proba(self.x), loaded.predict_proba(self.x))
                index = json.loads((full_path / "checkpoint_index.json").read_text())
                self.assertEqual(set(index), {"initial.pt", "epoch_0001.pt", "epoch_0002.pt", "final.pt"})
                prefix_path = self.root / f"prefix_{i}"
                module.fit_reupload(model_id, self.x, self.y, seed=17, output_dir=prefix_path, config={"epochs": 1, "n_layers": 1})
                resumed = module.fit_reupload(model_id, self.x, self.y, seed=17, output_dir=self.root / f"resume_{i}",
                    config={**options, "resume_from": str(prefix_path / "checkpoints/final.pt")})
                np.testing.assert_array_equal(full.predict_proba(self.x), resumed.predict_proba(self.x))
                current = full.preprocessor.to_dict()
                self.assertTrue(preprocessing is None or preprocessing == current)
                preprocessing = current
                saved = torch.load(full_path / "checkpoints/final.pt", weights_only=True)
                self.assertIn("optimizer_state", saved)
                self.assertIn("rng_state", saved)
                self.assertEqual(saved["sampler_state"]["kind"], "full_batch_no_shuffle")
                self.assertTrue(all(np.isfinite(r["train_loss"]) for r in full.metadata["history"]))


if __name__ == "__main__":
    unittest.main()
