"""Bounded one-mode finite-Fock hybrid adapters; not an SF or paper reproduction.

Training uses C=32 projected-ladder gate exponentials. C=24/40 are diagnostics,
never optimization choices. Unit norm is not evidence of cutoff convergence.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from ai4ct.qbench.common import seed_everything, sha256, validate_x, validate_xy, write_json
from ai4ct.qbench.variational import _data_digest, _rng_state, _restore_rng

MODEL_IDS=("cv_kerr_1mode_cut32", "cv_gaussian_1mode_cut32")
SCHEMA="qbench-cv-bounded-fock-v1"
BOUNDS=(math.pi, .08, math.pi, .10, .10, .20)
THRESHOLDS={"norm":1e-11, "initial_lost_mass":1e-13,
            "tail24":1e-8, "tail32":1e-11, "tail40":1e-14,
            "obs24_32":1e-7, "obs32_40":1e-9,
            "prob24_32":1e-7, "prob32_40":1e-9,
            "state24_32":1e-6, "state32_40":1e-8}


def resolved_config(config=None):
    defaults={"epochs":30,"lr":.005,"betas":[.9,.999],"eps":1e-8,
              "weight_decay":0.,"optimizer":"Adam","layers":3,"cutoff":32,
              "dtype":"float64","device":"cpu","threads":1,
              "max_wall_seconds":1800.,"resume_from":None}
    supplied=dict(config or {})
    if set(supplied)-set(defaults): raise ValueError("Unknown CV configuration keys")
    cfg={**defaults,**supplied}
    if isinstance(cfg['epochs'],bool) or not isinstance(cfg['epochs'],int) or not 1<=cfg['epochs']<=30:
        raise ValueError("epochs must be 1..30; registered comparison uses 30")
    for key in set(defaults)-{'epochs','resume_from','max_wall_seconds'}:
        if cfg[key]!=defaults[key]: raise ValueError(f"Fixed CV protocol field differs: {key}")
    if not np.isfinite(cfg['max_wall_seconds']) or cfg['max_wall_seconds']<=0:
        raise ValueError("Invalid wall budget")
    cfg['resume_from']=str(cfg['resume_from']) if cfg['resume_from'] is not None else None
    return cfg


class InputTransform:
    def __init__(self, values): self.values=values

    @classmethod
    def fit(cls,x):
        scaler=StandardScaler().fit(validate_x(x))
        return cls({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                    'variance':scaler.var_.tolist(),'n_train':int(scaler.n_samples_seen_),
                    'definition':'train-only StandardScaler; no PCA, clipping or IF'})

    def transform(self,x):
        z=(validate_x(x)-np.asarray(self.values['mean']))/np.asarray(self.values['scale'])
        if not np.isfinite(z).all():raise FloatingPointError("Nonfinite standardized inputs")
        return z


@lru_cache(maxsize=3)
def _operators(c):
    if c not in (24,32,40): raise ValueError("Only preregistered 24/32/40 cutoffs supported")
    a=torch.diag(torch.sqrt(torch.arange(1,c,dtype=torch.float64)),diagonal=1).to(torch.complex128)
    ad=a.conj().T
    n=torch.arange(c,dtype=torch.float64)
    return a,ad,n,(a+ad)/math.sqrt(2),(a-ad)/(1j*math.sqrt(2))


class CVNetwork(nn.Module):
    def __init__(self,model_id,seed=42):
        super().__init__()
        if model_id not in MODEL_IDS:raise ValueError("Unknown CV model")
        self.model_id=model_id
        self.gaussian=model_id==MODEL_IDS[1]
        generator=torch.Generator(device='cpu').manual_seed(int(seed))
        self.front_weight=nn.Parameter(torch.randn(2,17,generator=generator,dtype=torch.float64)/math.sqrt(17))
        self.front_bias=nn.Parameter(torch.zeros(2,dtype=torch.float64))
        # Always draw all six columns before discarding Kerr, preserving all shared initial values.
        all_raw=torch.empty(3,6,dtype=torch.float64).uniform_(-.1,.1,generator=generator)
        self.raw_gates=nn.Parameter(all_raw[:,:5].clone() if self.gaussian else all_raw)
        self.read_weight=nn.Parameter(torch.randn(2,3,generator=generator,dtype=torch.float64)/math.sqrt(3))
        self.read_bias=nn.Parameter(torch.zeros(2,dtype=torch.float64))

    def physical_gates(self):
        raw=torch.cat([self.raw_gates,torch.zeros(3,1,dtype=torch.float64)],dim=1) if self.gaussian else self.raw_gates
        return torch.tanh(raw)*torch.tensor(BOUNDS,dtype=torch.float64)

    def simulate(self,z,cutoff=32):
        if z.ndim!=2 or z.shape[1]!=17 or not len(z) or z.dtype!=torch.float64 or z.device.type!='cpu':
            raise ValueError("CV expects nonempty CPU float64 n by 17 tensor")
        if not torch.isfinite(z).all() or not all(torch.isfinite(p).all() for p in self.parameters()):
            raise FloatingPointError("Nonfinite CV input or parameters")
        a,ad,n,q,p=_operators(cutoff)
        u=.35*torch.tanh(z@self.front_weight.T+self.front_bias)
        alpha=torch.complex(u[:,0],u[:,1])
        terms=[torch.exp(-torch.abs(alpha)**2/2)]
        for level in range(1,cutoff):terms.append(terms[-1]*alpha/math.sqrt(level))
        psi=torch.stack(terms,dim=-1)
        initial_norm=psi.abs().square().sum(dim=-1)
        lost=float(torch.max(torch.abs(1-initial_norm)).detach())
        if lost>THRESHOLDS['initial_lost_mass']:raise FloatingPointError("Initial coherent projection mass exceeds budget")
        psi=psi/torch.sqrt(initial_norm)[:,None]
        max_norm=max_tail=0.

        def monitor(state):
            nonlocal max_norm,max_tail
            probability=state.detach().abs().square()
            norm=float(torch.max(torch.abs(probability.sum(dim=-1)-1)))
            tail=float(torch.max(probability[:,-4:].sum(dim=-1)))
            if not math.isfinite(norm) or not math.isfinite(tail):raise FloatingPointError("Nonfinite CV state diagnostics")
            max_norm=max(max_norm,norm);max_tail=max(max_tail,tail)
            if norm>THRESHOLDS['norm']:raise FloatingPointError("Finite-Fock state norm drift exceeds budget")
            if tail>THRESHOLDS[f'tail{cutoff}']:raise FloatingPointError(f"C{cutoff} boundary tail exceeds budget")

        monitor(psi)
        for phi1,r,phi2,br,bi,k in self.physical_gates():
            psi=psi*torch.exp(1j*phi1*n);monitor(psi)
            squeeze=torch.linalg.matrix_exp(.5*r*(a@a-ad@ad))
            psi=psi@squeeze.T;monitor(psi)
            psi=psi*torch.exp(1j*phi2*n);monitor(psi)
            beta=torch.complex(br,bi)
            displacement=torch.linalg.matrix_exp(beta*ad-beta.conj()*a)
            psi=psi@displacement.T;monitor(psi)
            if not self.gaussian:
                psi=psi*torch.exp(1j*k*n*n);monitor(psi)
        moments=torch.stack([torch.einsum('bi,ij,bj->b',psi.conj(),op,psi).real for op in (q,p)],dim=1)
        number=(psi.abs().square()*n).sum(dim=1,keepdim=True)
        moments=torch.cat([moments,number],dim=1)
        logits=moments@(2*torch.tanh(self.read_weight)).T+2*torch.tanh(self.read_bias)
        if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite CV output")
        return logits,moments,psi,{'cutoff':cutoff,'max_norm_error':max_norm,
                                   'max_tail_last4':max_tail,'initial_projection_mass_error':lost}

    def forward(self,z):return self.simulate(z,32)[0]


def cutoff_check(model,z,include24=False):
    """No labels; fixed-parameter comparisons only. Never adapts a cutoff."""
    with torch.no_grad():
        values={c:model.simulate(z,c) for c in ((24,32,40) if include24 else (32,40))}
        report={'cutoffs':{str(c):v[3] for c,v in values.items()},'comparisons':{}}
        for low,high in ((24,32),(32,40)) if include24 else ((32,40),):
            l,h=values[low],values[high]
            key=f'{low}_{high}'
            difference={'observable_abs':float(torch.max(torch.abs(l[1]-h[1]))),
                        'probability_abs':float(torch.max(torch.abs(l[0].softmax(-1)[:,1]-h[0].softmax(-1)[:,1]))),
                        'state_l2':float(torch.max(torch.linalg.vector_norm(l[2]-h[2][:,:low],dim=-1)))}
            for metric,field in [('obs','observable_abs'),('prob','probability_abs'),('state','state_l2')]:
                if difference[field]>THRESHOLDS[f'{metric}{key}']:
                    raise FloatingPointError(f"CV cutoff {key} {field} exceeds fixed convergence budget")
            report['comparisons'][key]=difference
    return values[32][0],report


class FittedCV:
    def __init__(self,model,preprocessor,metadata):
        self.model,self.preprocessor,self.metadata=model,preprocessor,metadata
        self.last_prediction_diagnostics=None

    def _logits(self,x):
        transformed=self.preprocessor.transform(x)
        self.model.eval();parts=[];diagnostics=[]
        for start in range(0,len(transformed),256):
            z=torch.as_tensor(transformed[start:start+256],dtype=torch.float64)
            logits,report=cutoff_check(self.model,z,include24=False)
            parts.append(logits.numpy());diagnostics.append(report)
        self.last_prediction_diagnostics=diagnostics
        return np.concatenate(parts)

    def predict_score(self,x):
        logits=self._logits(x);return logits[:,1]-logits[:,0]

    def predict_proba(self,x):
        logits=self._logits(x);exp=np.exp(logits-logits.max(axis=1,keepdims=True))
        return exp[:,1]/exp.sum(axis=1)


def _read_checkpoint(path):
    path=Path(path)
    index=json.loads((path.parent.parent/'checkpoint_index.json').read_text(encoding='utf-8'))
    if path.name not in index or sha256(path)!=index[path.name]['sha256']:
        raise ValueError("CV checkpoint hash mismatch")
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=SCHEMA or saved['source_sha256']!=sha256(__file__):
        raise ValueError("CV checkpoint schema/source differs")
    return saved,index


def fit_cv(model_id,x_train,y_train,*,seed,output_dir,config=None):
    cfg=resolved_config(config)
    if model_id not in MODEL_IDS:raise ValueError("Unknown CV model")
    x,y=validate_xy(x_train,y_train)
    fingerprint=_data_digest(x,y)
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError("CV output must be new/empty")
    seed_everything(seed,threads=1)
    model=CVNetwork(model_id,seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005,betas=(.9,.999),eps=1e-8,weight_decay=0)
    preprocessor=InputTransform.fit(x)
    history=[];start_epoch=0;best_loss=float('inf');best_epoch=None;lineage=None
    # Validate all lineage before creating output, then preserve exact snapshots.
    if cfg['resume_from'] is not None:
        source=Path(cfg['resume_from']);saved,old_index=_read_checkpoint(source)
        if saved['model_id']!=model_id or saved['seed']!=seed or saved['training_data_sha256']!=fingerprint:
            raise ValueError("CV resume model/seed/training data differs")
        for key in set(cfg)-{'epochs','resume_from','max_wall_seconds'}:
            if saved['config'][key]!=cfg[key]:raise ValueError("CV resume configuration differs")
        start_epoch=saved['epoch']
        if start_epoch>=cfg['epochs']:raise ValueError("CV resume target already reached")
        preprocessor=InputTransform(saved['preprocessor'])
        if not np.array_equal(preprocessor.transform(x),saved['training_standardized'].numpy()):
            raise ValueError("CV resume preprocessor/training transform differs")
        history=copy.deepcopy(saved['history']);best_loss=saved['best_loss'];best_epoch=saved['best_epoch']
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
        _restore_rng(torch,saved['rng_state'])
        names=['initial.pt']+[f'epoch_{i:04d}.pt' for i in range(1,start_epoch+1)]
        for name in names:
            if name not in old_index or sha256(source.parent/name)!=old_index[name]['sha256']:
                raise ValueError("CV resume ancestor checksum differs")
        lineage={'checkpoint_sha256':sha256(source),'epoch':start_epoch}
    output.mkdir(parents=True,exist_ok=True);checkpoints=output/'checkpoints';checkpoints.mkdir()
    index={}
    if lineage:
        for name in names:
            shutil.copy2(source.parent/name,checkpoints/name)
            index[name]={'sha256':sha256(checkpoints/name),'epoch':old_index[name]['epoch'],'copied_from_resume_lineage':True}
    z=torch.as_tensor(preprocessor.transform(x),dtype=torch.float64);target=torch.as_tensor(y,dtype=torch.long)
    metadata={'schema':SCHEMA,'model_id':model_id,'family':'cv','seed':int(seed),'config':cfg,'status':'running',
      'n_train':len(y),'training_data_sha256':fingerprint,'source_sha256':sha256(__file__),
      'checkpoint_epoch_snapshots':'saved','resume_mode':'epoch_boundary','resume_lineage':lineage,
      'backend':'torch_cpu_complex128_finite_fock_matrix_exponential','quantum_hardware_used':False,
      'adaptation':'one-mode bounded finite-Fock hybrid; not exact infinite-dimensional or original SF experiment',
      'parameter_counts':{'trainable':sum(p.numel() for p in model.parameters()),'front':36,'readout':8,
                          'cv':15 if model.gaussian else 18},
      'gaussian_control':'Kerr disabled; all common parameters and initial values matched; parameter count differs by 3',
      'preprocessor':preprocessor.values,'limits':{'alpha_component':.35,'gate_column_bounds':list(BOUNDS),
                                                 'readout_weight_and_bias':2.,'layers':3},
      'thresholds':THRESHOLDS,'feature_representation':'train_only_StandardScaler17_then_learned_bounded_complex_displacement',
      'probability_kind':'uncalibrated_classical_softmax_of_CV_moments',
      'threshold_rule':'Pass iff score>0 / probability>.5; ties Fail',
      'selection':'final epoch; best is diagnostic minimum training loss',
      'versions':{'torch':str(torch.__version__),'numpy':np.__version__}}
    write_json(output/'metadata.json',metadata)
    started=time.perf_counter()

    def save(name,epoch):
        path=checkpoints/name
        if path.exists():raise FileExistsError("CV checkpoint would overwrite prior bytes")
        payload={'schema':SCHEMA,'model_id':model_id,'seed':int(seed),'config':cfg,'epoch':epoch,
          'source_sha256':sha256(__file__),'training_data_sha256':fingerprint,
          'model_state':copy.deepcopy(model.state_dict()),'optimizer_state':copy.deepcopy(optimizer.state_dict()),
          'scheduler_state':None,'amp_scaler_state':None,'rng_state':_rng_state(torch),
          'sampler_state':{'kind':'full_batch_no_shuffle','order':list(range(len(y))),'cursor':0},
          'preprocessor':preprocessor.values,'training_standardized':z.clone(),
          'history':copy.deepcopy(history),'best_loss':best_loss,'best_epoch':best_epoch}
        torch.save(payload,path)
        index[name]={'epoch':epoch,'sha256':sha256(path),'bytes':path.stat().st_size}
        write_json(output/'checkpoint_index.json',index)

    try:
        _,initial_diagnostics=cutoff_check(model,z,include24=True)
        metadata['initial_or_resume_convergence']=initial_diagnostics
        if not lineage:save('initial.pt',0)
        for epoch in range(start_epoch+1,cfg['epochs']+1):
            if time.perf_counter()-started>cfg['max_wall_seconds']:raise TimeoutError("CV fit wall budget exhausted")
            tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(z),target)
            if not torch.isfinite(loss):raise FloatingPointError("CV loss is nonfinite")
            loss.backward()
            gradients=[p.grad for p in model.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):raise FloatingPointError("CV gradients missing/nonfinite")
            gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
            optimizer.step()
            with torch.no_grad():
                logits,_,_,diagnostics=model.simulate(z,32)
                after=float(torch.nn.functional.cross_entropy(logits,target))
            history.append({'epoch':epoch,'train_loss_pre_update':float(loss.detach()),'train_loss':after,
                            'gradient_norm':gradient_norm,'seconds':time.perf_counter()-tick,'state_diagnostics':diagnostics})
            if after<best_loss:best_loss,best_epoch=after,epoch
            save(f'epoch_{epoch:04d}.pt',epoch)
            write_json(checkpoints/'aliases.json',{'last':f'epoch_{epoch:04d}.pt','best':f'epoch_{best_epoch:04d}.pt',
                         'best_rule':'minimum post-update training loss; final used for inference'})
        _,final_diagnostics=cutoff_check(model,z,include24=True)
        save('final.pt',cfg['epochs'])
        metadata.update(status='completed',epochs_completed=cfg['epochs'],fit_seconds=time.perf_counter()-started,
                        history=history,training_cutoff_convergence=final_diagnostics)
        write_json(output/'metadata.json',metadata)
        return FittedCV(model,preprocessor,metadata)
    except Exception as error:
        metadata.update(status='failed',history=history,error_type=type(error).__name__,error=str(error))
        write_json(output/'metadata.json',metadata)
        raise


def load_cv(output_dir):
    output=Path(output_dir)
    metadata=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('schema')!=SCHEMA or metadata.get('status')!='completed':raise ValueError("Only completed CV fits can be loaded")
    saved,_=_read_checkpoint(output/'checkpoints/final.pt')
    for key in ('model_id','seed','config','source_sha256','training_data_sha256'):
        if saved[key]!=metadata[key]:raise ValueError("CV saved metadata differs")
    if saved['epoch']!=metadata['config']['epochs']:raise ValueError("CV final epoch differs")
    model=CVNetwork(saved['model_id'],saved['seed']);model.load_state_dict(saved['model_state']);model.eval()
    _,report=cutoff_check(model,saved['training_standardized'],include24=True)
    fitted=FittedCV(model,InputTransform(saved['preprocessor']),metadata)
    fitted.load_training_convergence=report
    return fitted
