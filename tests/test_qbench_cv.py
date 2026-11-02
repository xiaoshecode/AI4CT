import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.linalg import expm
import torch

SOURCE=Path(__file__).resolve().parents[1]/'src/ai4ct/qbench/cv.py'
spec=importlib.util.spec_from_file_location('staged_cv',SOURCE)
cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)


def scipy_state(model,z,c):
    # Independently built Fock ladder, input coefficients and gate ordering.
    a=np.zeros((c,c),complex)
    for n in range(1,c):a[n-1,n]=math.sqrt(n)
    ad=a.conj().T;number=np.arange(c)
    u=.35*np.tanh(z@model.front_weight.detach().numpy().T+model.front_bias.detach().numpy())
    alpha=u[:,0]+1j*u[:,1]
    psi=np.array([[np.exp(-abs(v)**2/2)*v**n/math.sqrt(math.factorial(n)) for n in range(c)] for v in alpha])
    psi/=np.linalg.norm(psi,axis=1,keepdims=True)
    for phi1,r,phi2,br,bi,k in model.physical_gates().detach().numpy():
        psi=psi@np.diag(np.exp(1j*phi1*number))
        psi=psi@expm(r*(a@a-ad@ad)/2).T
        psi=psi@np.diag(np.exp(1j*phi2*number))
        beta=complex(br,bi)
        psi=psi@expm(beta*ad-beta.conjugate()*a).T
        if not model.gaussian:psi=psi@np.diag(np.exp(1j*k*number**2))
    moments=np.column_stack([np.einsum('bi,ij,bj->b',psi.conj(),op,psi).real for op in
                            ((a+ad)/np.sqrt(2),(a-ad)/(1j*np.sqrt(2)),np.diag(number))])
    return psi,moments


def gaussian_moments(model,z):
    u=.35*np.tanh(z@model.front_weight.detach().numpy().T+model.front_bias.detach().numpy())
    mean=np.sqrt(2)*u;cov=np.eye(2)/2
    for phi1,r,phi2,br,bi,_ in model.physical_gates().detach().numpy():
        for theta,squeeze in ((phi1,False),(r,True),(phi2,False)):
            mat=np.diag([np.exp(-theta),np.exp(theta)]) if squeeze else np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            mean=mean@mat.T;cov=mat@cov@mat.T
        mean+=np.sqrt(2)*np.array([br,bi])
    return np.column_stack([mean,(np.sum(mean*mean,axis=1)+np.trace(cov)-1)/2])


class CVMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.z=np.random.default_rng(7).normal(size=(5,17))

    def test_independent_scipy_states_moments_all_cutoffs(self):
        for model_id in cv.MODEL_IDS:
            model=cv.CVNetwork(model_id,6)
            with torch.no_grad():model.raw_gates.fill_(.7)
            for c in (24,32,40):
                _,m,s,_=model.simulate(torch.tensor(self.z),c)
                reference,expected=scipy_state(model,self.z,c)
                np.testing.assert_allclose(s.detach(),reference,atol=1e-11,rtol=0)
                np.testing.assert_allclose(m.detach(),expected,atol=1e-11,rtol=0)

    def test_gaussian_phase_space_oracle_and_zero_kerr_match(self):
        gaussian=cv.CVNetwork(cv.MODEL_IDS[1],9);kerr=cv.CVNetwork(cv.MODEL_IDS[0],9)
        with torch.no_grad():
            gaussian.raw_gates.fill_(.5);kerr.raw_gates[:,:5].copy_(gaussian.raw_gates);kerr.raw_gates[:,5].zero_()
        gm=gaussian.simulate(torch.tensor(self.z))[1]
        np.testing.assert_allclose(gm.detach(),gaussian_moments(gaussian,self.z),atol=1e-10,rtol=0)
        self.assertTrue(torch.equal(gaussian(torch.tensor(self.z)),kerr(torch.tensor(self.z))))

    def test_counts_and_same_seed_common_initialization(self):
        k,g=[cv.CVNetwork(i,31) for i in cv.MODEL_IDS]
        self.assertEqual(sum(p.numel() for p in k.parameters()),62)
        self.assertEqual(sum(p.numel() for p in g.parameters()),59)
        for key in ('front_weight','front_bias','read_weight','read_bias'):
            self.assertTrue(torch.equal(getattr(k,key),getattr(g,key)))
        self.assertTrue(torch.equal(k.raw_gates[:,:5],g.raw_gates))
        self.assertGreater(float((k(torch.tensor(self.z))-g(torch.tensor(self.z))).abs().max()),1e-9)

    def test_autograd_against_independent_scipy_finite_difference(self):
        model=cv.CVNetwork(cv.MODEL_IDS[0],8)
        z=torch.tensor(self.z,requires_grad=True)
        objective=(model.simulate(z)[1]*torch.tensor([.2,-.3,.1])).sum();objective.backward()
        eps=1e-6
        for tensor,index in [(model.raw_gates,(1,5)),(model.raw_gates,(0,1)),(model.raw_gates,(2,3)),(model.front_weight,(0,4))]:
            analytic=float(tensor.grad[index]);original=float(tensor[index].detach())
            values=[]
            for delta in (eps,-eps):
                with torch.no_grad():tensor[index]=original+delta
                values.append(float(np.sum(scipy_state(model,self.z,32)[1]*[.2,-.3,.1])))
            with torch.no_grad():tensor[index]=original
            self.assertAlmostEqual(analytic,(values[0]-values[1])/(2*eps),delta=2e-6)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_input_and_bounds_and_full_cutoff_contract(self):
        model=cv.CVNetwork(cv.MODEL_IDS[0],3)
        with torch.no_grad():model.raw_gates.fill_(20.)
        self.assertTrue(torch.all(model.physical_gates().abs()<=torch.tensor(cv.BOUNDS,dtype=torch.float64)))
        _,report=cv.cutoff_check(model,torch.tensor(self.z),True)
        self.assertEqual(set(report['cutoffs']),{'24','32','40'})
        with self.assertRaises(ValueError):model(torch.zeros(2,16,dtype=torch.float64))
        with self.assertRaises(FloatingPointError):model(torch.full((2,17),float('nan'),dtype=torch.float64))

    def test_reject_nonunitary_boundary_and_probability_divergence(self):
        model=cv.CVNetwork(cv.MODEL_IDS[0],3);z=torch.tensor(self.z)
        with mock.patch.object(torch.linalg,'matrix_exp',side_effect=lambda m:2*torch.eye(len(m),dtype=m.dtype)):
            with self.assertRaisesRegex(FloatingPointError,'norm'):model(z)
        with mock.patch.dict(cv.THRESHOLDS,{'tail32':-1}):
            with self.assertRaisesRegex(FloatingPointError,'tail'):model(z)
        original=model.simulate
        def modified(z,c=32):
            logits,m,state,diag=original(z,c)
            if c==40:logits=logits+torch.tensor([0.,.01])
            return logits,m,state,diag
        with mock.patch.object(model,'simulate',side_effect=modified):
            with self.assertRaisesRegex(FloatingPointError,'probability'):cv.cutoff_check(model,z)


class CVPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.x=np.random.default_rng(5).normal(size=(18,17));self.y=np.arange(18)%2
    def tearDown(self):self.tmp.cleanup()

    def fit(self,model_id,out,epochs=2,**extra):
        return cv.fit_cv(model_id,self.x,self.y,seed=42,output_dir=out,config={'epochs':epochs,**extra})

    def test_fit_reload_predictions_train_only_and_all_snapshots(self):
        for i,model_id in enumerate(cv.MODEL_IDS):
            path=self.root/str(i);fit=self.fit(model_id,path)
            p=fit.predict_proba(self.x[:5]+4);score=fit.predict_score(self.x[:5]+4)
            np.testing.assert_allclose(fit.preprocessor.values['mean'],self.x.mean(axis=0),atol=1e-14)
            self.assertTrue(np.array_equal(score>0,p>.5))
            other=cv.load_cv(path)
            np.testing.assert_array_equal(p,other.predict_proba(self.x[:5]+4))
            np.testing.assert_allclose(p,np.array([fit.predict_proba(row[None]+4)[0] for row in self.x[:5]]),atol=1e-14)
            self.assertIn('32_40',other.last_prediction_diagnostics[0]['comparisons'])
            for name in ('initial.pt','epoch_0001.pt','epoch_0002.pt','final.pt'):
                saved=torch.load(path/'checkpoints'/name,weights_only=True)
                self.assertIn('optimizer_state',saved);self.assertIn('rng_state',saved);self.assertIn('sampler_state',saved)
            self.assertEqual(fit.metadata['checkpoint_epoch_snapshots'],'saved')
            self.assertEqual(set(fit.metadata['training_cutoff_convergence']['cutoffs']),{'24','32','40'})

    def test_exact_resume_model_optimizer_rng_and_scores(self):
        for i,model_id in enumerate(cv.MODEL_IDS):
            full=self.fit(model_id,self.root/f'full{i}',4)
            self.fit(model_id,self.root/f'partial{i}',2)
            resumed=self.fit(model_id,self.root/f'resumed{i}',4,resume_from=str(self.root/f'partial{i}/checkpoints/epoch_0002.pt'))
            for key,value in full.model.state_dict().items():self.assertTrue(torch.equal(value,resumed.model.state_dict()[key]),key)
            np.testing.assert_array_equal(full.predict_score(self.x),resumed.predict_score(self.x))
            a=torch.load(self.root/f'full{i}/checkpoints/final.pt',weights_only=True)
            b=torch.load(self.root/f'resumed{i}/checkpoints/final.pt',weights_only=True)
            for key,state in a['optimizer_state']['state'].items():
                for name,t in state.items():self.assertTrue(torch.equal(t,b['optimizer_state']['state'][key][name]))
            self.assertTrue(torch.equal(a['rng_state']['torch'],b['rng_state']['torch']))
            self.assertEqual(a['rng_state']['python'],b['rng_state']['python'])
            self.assertEqual([h['train_loss'] for h in a['history']],[h['train_loss'] for h in b['history']])

    def test_reject_changed_data_and_tampered_checkpoint(self):
        path=self.root/'source';self.fit(cv.MODEL_IDS[0],path,1)
        with self.assertRaisesRegex(ValueError,'data'):
            cv.fit_cv(cv.MODEL_IDS[0],self.x+.01,self.y,seed=42,output_dir=self.root/'bad',config={'epochs':2,'resume_from':str(path/'checkpoints/epoch_0001.pt')})
        with (path/'checkpoints/final.pt').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):cv.load_cv(path)

    def test_reject_mutated_source_lineage_and_nonempty_output(self):
        path=self.root/'fit';self.fit(cv.MODEL_IDS[0],path,1)
        with self.assertRaises(FileExistsError):self.fit(cv.MODEL_IDS[0],path)
        with mock.patch.object(cv,'sha256',side_effect=lambda p:'changed' if Path(p)==SOURCE else __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError,'source'):cv.load_cv(path)

    def test_fixed_config_and_failed_diagnostics_preserved(self):
        for config in ({'cutoff':40},{'optimizer':'AdamW'},{'lr':.1},{'epochs':True},{'extra':1}):
            with self.assertRaises(ValueError):cv.resolved_config(config)
        with mock.patch.dict(cv.THRESHOLDS,{'tail32':-1}):
            with self.assertRaises(FloatingPointError):self.fit(cv.MODEL_IDS[0],self.root/'failed')
        metadata=json.loads((self.root/'failed/metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['status'],'failed')


if __name__=='__main__':unittest.main()
