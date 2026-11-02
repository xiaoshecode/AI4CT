import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai4ct.qbench.common import Representation


@unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane optional environment required")
class QuantumMapTests(unittest.TestCase):
    def test_separable_kernel_analytic_and_bloch_oracle(self):
        import torch
        from ai4ct.qbench.kernels import state_circuit, fidelity_kernel, local_bloch
        a = np.array([[.2, -.3, .4, -.5, .6], [-.1, .7, -.8, .9, -.2]])
        states = state_circuit("separable")(torch.tensor(a)).detach().numpy()
        expected = np.prod(np.cos((a[0]-a[1])/2)**2)
        self.assertAlmostEqual(fidelity_kernel(states, states)[0, 1], expected, places=12)
        bloch = local_bloch(states)
        np.testing.assert_allclose(bloch[:, ::3], np.sin(a), atol=1e-12)
        np.testing.assert_allclose(bloch[:, 1::3], 0, atol=1e-12)
        np.testing.assert_allclose(bloch[:, 2::3], np.cos(a), atol=1e-12)
        state_y = np.zeros((1,32), dtype=complex)
        state_y[0,0], state_y[0,16] = 1/np.sqrt(2), 1j/np.sqrt(2)
        self.assertAlmostEqual(local_bloch(state_y)[0,1], 1, places=12)

    def test_trainable_kernel_gradient_and_fixed_zero_equivalence(self):
        import torch
        from ai4ct.qbench.kernels import state_circuit
        a = torch.tensor(np.random.default_rng(13).normal(size=(4, 5)), dtype=torch.float64)
        theta = torch.zeros((2,5),dtype=torch.float64,requires_grad=True)
        trainable, fixed = state_circuit("zz",True), state_circuit("zz")
        states = trainable(a,theta)
        torch.testing.assert_close(states, fixed(a), atol=1e-12, rtol=1e-12)
        value = abs((states[0].conj()*states[1]).sum())**2
        value.backward()
        self.assertGreater(float(theta.grad.norm()), 1e-8)
        epsilon=1e-6
        plus,minus = theta.detach().clone(),theta.detach().clone()
        plus[0,0]+=epsilon
        minus[0,0]-=epsilon
        def score(t):
            s=trainable(a,t)
            return float(abs((s[0].conj()*s[1]).sum())**2)
        self.assertAlmostEqual(float(theta.grad[0,0]),(score(plus)-score(minus))/(2*epsilon),places=6)

    def test_all_kernel_models_save_load_and_inference_does_not_refit(self):
        from ai4ct.qbench.kernels import MODEL_IDS, fit_kernel, load_kernel
        rng=np.random.default_rng(5)
        x=rng.normal(size=(12,17)); y=np.arange(12)%2
        test=rng.normal(size=(3,17))
        with tempfile.TemporaryDirectory() as temp:
            for name in MODEL_IDS:
                with self.subTest(model=name):
                    out=Path(temp)/name
                    fit=fit_kernel(name,x,y,seed=17,output_dir=out,config={"alignment_epochs":2,"alignment_batch_size":8})
                    original=fit.mapping.representation.scaler.mean_.copy()
                    score=fit.predict_score(test)
                    self.assertEqual(score.shape,(3,))
                    self.assertTrue(np.isfinite(score).all())
                    np.testing.assert_allclose(load_kernel(out).predict_score(test),score,atol=1e-12,rtol=1e-12)
                    np.testing.assert_array_equal(original,fit.mapping.representation.scaler.mean_)

    def test_kernel_epoch_resume_matches_continuous_training(self):
        from ai4ct.qbench.kernels import fit_kernel
        rng=np.random.default_rng(44)
        x=rng.normal(size=(12,17)); y=np.arange(12)%2
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); config={'alignment_epochs':3,'alignment_batch_size':8}
            full=fit_kernel('kernel_trainable_5q',x,y,seed=17,output_dir=folder/'full',config=config)
            fit_kernel('kernel_trainable_5q',x,y,seed=17,output_dir=folder/'part',config={**config,'alignment_epochs':1})
            resumed=fit_kernel('kernel_trainable_5q',x,y,seed=17,output_dir=folder/'resume',config={**config,'resume_from':str(folder/'part/checkpoints/final.pt')})
            np.testing.assert_array_equal(full.mapping.theta,resumed.mapping.theta)
            np.testing.assert_array_equal(full.predict_score(x),resumed.predict_score(x))


class RepresentationTests(unittest.TestCase):
    def test_transform_never_uses_heldout_statistics(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(10,17))
        transform=Representation(components=5,angle=True).fit(x)
        mean=transform.scaler.mean_.copy()
        out=transform.transform(np.full((2,17),1e6))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((abs(out)<np.pi).all())
        np.testing.assert_array_equal(mean,transform.scaler.mean_)
