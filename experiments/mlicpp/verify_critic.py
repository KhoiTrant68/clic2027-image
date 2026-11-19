"""Reload trained critic on CPU and independently score held-source groups."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def verify(out):
    torch.set_num_threads(2)
    data=np.load(out/'dataset.npz');mask=data['validation'];groups=data['groups']
    assert set(groups[~mask,0]).isdisjoint(set(groups[mask,0]))
    assert int((~mask).sum())==768 and int(mask.sum())==384
    results={}
    for name in ['critic_best.pt','critic_last.pt']:
        ck=torch.load(out/name,map_location='cpu',weights_only=False)
        model=torch.nn.Sequential(torch.nn.Linear(ck['input_dim'],ck['hidden']),torch.nn.GELU(),
            torch.nn.Linear(ck['hidden'],ck['hidden']),torch.nn.GELU(),torch.nn.Linear(ck['hidden'],1))
        model.load_state_dict(ck['state_dict'],strict=True);model.eval()
        x=((torch.tensor(data['x'])-ck['mean'])/ck['std']).clamp(-20,20)
        with torch.no_grad(): pred=(model(x).squeeze(1)*ck['target_scale']).numpy()
        assert np.isfinite(pred).all()
        y=data['y'];regret=[];selected=[];oracle=[]
        for key in np.unique(groups[mask],axis=0):
            ix=np.where((groups==key).all(1))[0]
            chosen=ix[np.argmin(pred[ix])];best=ix[np.argmin(y[ix])]
            selected.append(float(y[chosen]));oracle.append(float(y[best]));regret.append(float(y[chosen]-y[best]))
        mse=float(np.mean((pred[mask]-y[mask])**2))
        if 'metrics' in ck:
            assert np.isclose(mse,ck['metrics']['validation_mse'],rtol=1e-3,atol=1e-10)
        results[name]=dict(step=ck['step'],validation_mse=mse,regret=float(np.mean(regret)),
            chosen_delta_j=float(np.mean(selected)),teacher_delta_j=float(np.mean(oracle)),
            sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
    result=dict(passed=True,train_examples=768,validation_examples=384,source_disjoint=True,
                checkpoints=results,device='cpu',note='Offline teacher-state decisions, not sequential rollout')
    (out/'checkpoint_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify(Path(sys.argv[1]))
