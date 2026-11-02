"""Synthetic validation for staged adapters; never reads research data."""
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np


STAGED = Path(__file__).resolve().parents[1] / "src/ai4ct/qbench/inductive.py"
SPEC = importlib.util.spec_from_file_location("staged_qbench_inductive", STAGED)
inductive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inductive)


class QuboTests(unittest.TestCase):
    def test_hand_calculated_two_stumps(self):
        h = np.array([[1, 1], [-1, 1]])
        q, constant = inductive.qubo_coefficients(h, np.array([1, -1]), .01)
        np.testing.assert_allclose(q, [[-.745, 0], [0, .255]], atol=1e-15, rtol=0)
        result = inductive.enumerate_qubo(q, constant)
        np.testing.assert_array_equal(result["solution"], [1, 0])
        self.assertAlmostEqual(result["minimum_energy"], .255, places=14)

    def test_expansion_and_independent_enumeration(self):
        rng = np.random.default_rng(99)
        h = rng.choice([-1, 1], size=(11, 5))
        y = rng.choice([-1, 1], size=11)
        lam = .07
        q, constant = inductive.qubo_coefficients(h, y, lam)
        direct = []
        for code in range(32):
            w = np.array([(code >> i) & 1 for i in range(5)])
            loss = np.mean((h @ w / 5 - y)**2) + lam/5 * w.sum()
            expanded = constant + sum(q[i, i] * w[i] for i in range(5))
            expanded += sum(q[i, j] * w[i] * w[j] for i in range(5) for j in range(i+1, 5))
            self.assertAlmostEqual(loss, expanded, places=13)
            direct.append(loss)
        result = inductive.enumerate_qubo(q, constant)
        self.assertEqual(result["integer_code"], int(np.argmin(direct)))
        np.testing.assert_allclose(result["all_energies"], direct, atol=1e-13, rtol=0)

    def test_exact_ties_and_zero_margin_are_fail(self):
        result = inductive.enumerate_qubo(np.zeros((3, 3)))
        self.assertEqual(result["integer_code"], 0)
        np.testing.assert_array_equal(result["solution"], [0, 0, 0])
        x = np.zeros((4, 17))
        x[:, 0] = [-2, -1, 1, 2]
        with tempfile.TemporaryDirectory() as temp:
            fitted = inductive.fit_inductive(inductive.MODEL_IDS[0], x, [0,0,1,1],
                seed=42, output_dir=Path(temp)/'all_off', config={'lambda':100.})
            np.testing.assert_array_equal(fitted.predict_score(x), np.zeros(4))
            np.testing.assert_array_equal(fitted.predict_score(x)>0, np.zeros(4,dtype=bool))

    def test_dictionary_is_unique_and_thresholds_use_training_quantiles(self):
        z = np.column_stack([np.arange(8), np.arange(8), np.ones(8)])
        y = np.array([-1]*4 + [1]*4)
        stumps = inductive.select_stumps(z, y, max_weak_learners=12)
        h = inductive.stump_matrix(z, stumps)
        self.assertEqual(len(stumps), len({h[:,i].tobytes() for i in range(h.shape[1])}))
        for stump in stumps:
            if stump["kind"] == "threshold":
                self.assertEqual(stump["threshold"], np.quantile(z[:,stump["feature"]], stump["quantile"], method="linear"))
        self.assertEqual(stumps[0]["train_error"], 0.0)

    def test_invalid_qubo_size_and_signed_labels(self):
        with self.assertRaises(ValueError):
            inductive.enumerate_qubo(np.eye(13))
        with self.assertRaises(ValueError):
            inductive.qubo_coefficients(np.ones((2, 2)), [0, 1])
        with self.assertRaises(ValueError):
            inductive.enumerate_qubo(np.ones((2, 2)))


class InterferenceTests(unittest.TestCase):
    def test_six_qubit_pennylane_oracle(self):
        import pennylane as qml
        rng = np.random.default_rng(12)
        states = inductive.amplitude_vectors(rng.normal(size=(4, 17)))
        actual = inductive.interference_acceptance(states[:2], states[2:])
        @qml.qnode(qml.device("default.qubit", wires=6, shots=None))
        def circuit(pair_state):
            qml.StatePrep(pair_state, wires=range(6), normalize=False)
            qml.Hadamard(0)
            return qml.probs(wires=[0])
        for i, j in itertools.product(range(2), range(2)):
            pair_state = np.concatenate([states[i], states[j+2]]) / np.sqrt(2)
            self.assertAlmostEqual(float(circuit(pair_state)[0]), actual[i, j], places=13)
            # Independent norm derivation also catches squared-overlap replacements.
            self.assertAlmostEqual(np.dot(states[i]+states[j+2], states[i]+states[j+2])/4,
                                   actual[i,j], places=13)

    def test_equal_opposite_and_orthogonal_states(self):
        first, second = np.eye(32)[:2]
        training = np.stack([first, -first])
        queries = np.stack([first, -first, second])
        np.testing.assert_allclose(inductive.interference_acceptance(queries, training),
                                   [[1, 0], [0, 1], [.5, .5]], atol=1e-15, rtol=0)
        p = inductive.interference_pass_probability(queries, training, [1, 0])
        np.testing.assert_allclose(p, [1, 0, .5], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(p > .5, [True, False, False])
        with self.assertRaises(FloatingPointError):
            inductive.interference_pass_probability(first[None], np.stack([-first,-first]), [0,1])

    def test_probability_is_uniform_per_sample_not_class_balanced(self):
        first, second = np.eye(32)[:2]
        training = np.stack([first, first, first])
        p = inductive.interference_pass_probability(second[None], training, [1,1,0])
        self.assertAlmostEqual(p[0], 2/3, places=15)

    def test_bias_padding_norm_zero_and_large_vectors(self):
        z = np.zeros((3,17))
        z[1, 0] = 1e200
        z[2] = np.arange(17)
        state = inductive.amplitude_vectors(z)
        np.testing.assert_allclose(np.linalg.norm(state,axis=1), 1, atol=1e-15, rtol=0)
        self.assertEqual(state[0,17],1.)
        self.assertTrue((state[:,17] > 0).all())
        self.assertTrue((state[:,18:] == 0).all())


class AdapterTests(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(567)
        x = rng.normal(size=(24,17))
        y = (x[:,0] + .3*x[:,1] > 0).astype(int)
        return x,y,rng.normal(size=(5,17))

    def test_both_adapters_save_reload_batch_and_training_only(self):
        x,y,query=self.data()
        with tempfile.TemporaryDirectory() as temp:
            for model_id in inductive.MODEL_IDS:
                with self.subTest(model=model_id):
                    out=Path(temp)/model_id
                    fit=inductive.fit_inductive(model_id,x,y,seed=42,output_dir=out,config={})
                    np.testing.assert_allclose(fit.representation.scaler.mean_,x.mean(axis=0),atol=1e-15,rtol=0)
                    before=fit.representation.scaler.mean_.copy()
                    scores=fit.predict_score(query)
                    singles=np.concatenate([fit.predict_score(row[None]) for row in query])
                    np.testing.assert_allclose(scores,singles,atol=1e-14,rtol=0)
                    fit.predict_score(query+10000)
                    np.testing.assert_array_equal(before,fit.representation.scaler.mean_)
                    restored=inductive.load_inductive(out)
                    np.testing.assert_array_equal(scores,restored.predict_score(query))
                    metadata=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
                    artifact=joblib.load(out/'model.joblib')
                    self.assertEqual(metadata['n_train'],24)
                    self.assertEqual(metadata['calibration'],'none')
                    self.assertEqual(metadata['checkpoint_epoch_snapshots'],'not_applicable')
                    self.assertEqual(metadata['resume_mode'],'restart_trial')
                    self.assertFalse(metadata['quantum_hardware_used'])
                    self.assertEqual(artifact['config'],metadata['config'])
                    if model_id==inductive.MODEL_IDS[0]:
                        self.assertIsNone(fit.predict_proba(query))
                        self.assertLessEqual(metadata['enumerated_states'],4096)
                        self.assertIn('all_energies',artifact['payload'])
                    else:
                        p=fit.predict_proba(query)
                        self.assertTrue(((p>=0)&(p<=1)).all())
                        np.testing.assert_allclose(scores,2*p-1,atol=0,rtol=0)
                        self.assertEqual(artifact['payload']['training_states'].shape,(24,32))

    def test_deterministic_refit_and_artifact_tamper(self):
        x,y,query=self.data()
        with tempfile.TemporaryDirectory() as temp:
            for model_id in inductive.MODEL_IDS:
                first=inductive.fit_inductive(model_id,x,y,seed=42,output_dir=Path(temp)/(model_id+'a'),config={})
                second=inductive.fit_inductive(model_id,x,y,seed=43,output_dir=Path(temp)/(model_id+'b'),config={})
                np.testing.assert_array_equal(first.predict_score(query),second.predict_score(query))
                artifact=Path(temp)/(model_id+'a')/'model.joblib'
                with artifact.open('ab') as stream:
                    stream.write(b'tamper')
                with self.assertRaises(ValueError):
                    inductive.load_inductive(artifact.parent)

    def test_invalid_fit_contract_and_existing_directory(self):
        x,y,_=self.data()
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'fit'
            inductive.fit_inductive(inductive.MODEL_IDS[0],x,y,seed=42,output_dir=out,config={})
            with self.assertRaises(FileExistsError):
                inductive.fit_inductive(inductive.MODEL_IDS[0],x,y,seed=42,output_dir=out,config={})
            with self.assertRaises(ValueError):
                inductive.fit_inductive(inductive.MODEL_IDS[0],x,y,seed=42,output_dir=Path(temp)/'bad',config={'max_weak_learners':13})
            with self.assertRaises(ValueError):
                inductive.fit_inductive(inductive.MODEL_IDS[1],x,np.zeros(24),seed=42,output_dir=Path(temp)/'bad2',config={})


if __name__ == '__main__':
    unittest.main()
