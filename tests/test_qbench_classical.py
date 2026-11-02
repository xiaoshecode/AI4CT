from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai4ct.qbench.classical import MODEL_IDS, fit_classical, load_classical


class ClassicalTests(unittest.TestCase):
    def test_controls_roundtrip_and_score_semantics(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(20,17)); y=np.arange(20)%2
        with tempfile.TemporaryDirectory() as temp:
            for name in MODEL_IDS:
                with self.subTest(model=name):
                    folder=Path(temp)/name
                    fitted=fit_classical(name,x,y,seed=17,output_dir=folder,config={"epochs":2,"n_estimators":3})
                    score=fitted.predict_score(x[:3])
                    np.testing.assert_allclose(score,load_classical(folder).predict_score(x[:3]),atol=1e-12)
                    p=fitted.predict_proba(x[:3])
                    if name in ("linear_svm","rbf_svm"):
                        self.assertIsNone(p)
                    else:
                        np.testing.assert_array_equal(score>0,p>.5)
                    if name=="small_mlp":
                        self.assertTrue((folder/'checkpoints/initial.pt').exists())
                        self.assertTrue((folder/'checkpoints/epoch_0002.pt').exists())

    def test_invalid_labels_and_overwrite_are_rejected(self):
        x=np.ones((4,17))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                fit_classical('logistic',x,[0,1,.5,1],seed=1,output_dir=Path(temp)/'bad',config={})
            folder=Path(temp)/'good'
            fit_classical('always_pass',x,[0,1,0,1],seed=1,output_dir=folder,config={})
            with self.assertRaises(FileExistsError):
                fit_classical('always_pass',x,[0,1,0,1],seed=1,output_dir=folder,config={})

    def test_neural_epoch_resume_matches_continuous_training(self):
        rng=np.random.default_rng(41)
        x=rng.normal(size=(12,17)); y=np.arange(12)%2
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            full=fit_classical('small_mlp',x,y,seed=17,output_dir=folder/'full',config={'epochs':3})
            fit_classical('small_mlp',x,y,seed=17,output_dir=folder/'part',config={'epochs':1})
            resumed=fit_classical('small_mlp',x,y,seed=17,output_dir=folder/'resume',config={'epochs':3,'resume_from':str(folder/'part/checkpoints/final.pt')})
            np.testing.assert_array_equal(full.predict_score(x),resumed.predict_score(x))
