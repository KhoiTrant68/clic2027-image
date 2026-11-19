"""Paired CPU ablation: regression versus decision cost on fixed teacher states."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def run(out,data_path):
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'source.py')
    torch.set_num_threads(2)
    cfg=dict(seeds=[9041,9042,9043],steps=2000,groups_per_batch=16,hidden=256,lr=3e-4,
        device='cpu',threads=2,data=str(data_path),temperature=1.,mse_weight=.1)
    save(out/'config.json',cfg)
    d=np.load(data_path);mask=d['validation'];groups=d['groups'];raw=torch.tensor(d['x']);target=torch.tensor(d['y'])
    assert set(groups[~mask,0]).isdisjoint(set(groups[mask,0]))
    tr=torch.tensor(~mask);va=torch.tensor(mask)
    mean=raw[tr].mean(0);std=raw[tr].std(0,unbiased=False).clamp_min(1e-4)
    x=((raw-mean)/std).clamp(-20,20);scale=target[tr].std(unbiased=False).clamp_min(1e-6);y=target/scale
    group_indices=torch.tensor(np.stack([np.where((groups==k).all(1))[0] for k in np.unique(groups,axis=0)]))
    assert group_indices.shape[1]==6
    train_groups=group_indices[tr[group_indices[:,0]]];val_groups=group_indices[va[group_indices[:,0]]]
    assert torch.all(target[group_indices[:,-1]]==0)
    all_results=[];started=time.monotonic()
    for seed in cfg['seeds']:
        for objective in ['mse','decision']:
            torch.manual_seed(seed)
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),
                torch.nn.Linear(256,256),torch.nn.GELU(),torch.nn.Linear(256,1))
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
            generator=torch.Generator().manual_seed(seed)
            def evaluate():
                with torch.no_grad(): pred=model(x).squeeze(1)*scale
                result={}
                for label,idx in [('train',train_groups),('validation',val_groups)]:
                    values=target[idx];chosen=values.gather(1,pred[idx].argmin(1,keepdim=True)).squeeze(1)
                    result[label]=dict(mse=float(((pred[idx]-values)**2).mean()),
                        chosen_delta_j=float(chosen.mean()),regret=float((chosen-values.min(1).values).mean()),
                        harmful_fraction=float((chosen>0).float().mean()),
                        no_change_regret=float(-values.min(1).values.mean()))
                return result
            name=f'{objective}-seed{seed}';save(out/'status.json',dict(state='running',run=name))
            with (out/(name+'.jsonl')).open('w',buffering=1) as log:
                for step in range(1,cfg['steps']+1):
                    idx=train_groups[torch.randint(len(train_groups),(cfg['groups_per_batch'],),generator=generator)]
                    p=model(x[idx]).squeeze(-1);truth=y[idx]
                    mse=(p-truth).square().mean()
                    # Fixed within-group baseline makes the no-change action exact.
                    advantage=p-p[:,-1:]
                    regret=truth-truth.min(1,keepdim=True).values
                    decision=(torch.softmax(-advantage/cfg['temperature'],1)*regret).sum(1).mean()
                    loss=mse if objective=='mse' else decision+cfg['mse_weight']*mse
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    if step==1 or step%200==0:
                        entry=dict(run=name,step=step,loss=float(loss),**evaluate())
                        log.write(json.dumps(entry)+'\n');save(out/'progress.json',entry)
                    if time.monotonic()-started>3600: raise TimeoutError('One-hour budget')
            result=dict(seed=seed,objective=objective,step=cfg['steps'],**evaluate());all_results.append(result)
            torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,scale=scale,config=cfg,
                seed=seed,objective=objective),out/(name+'.pt'))
            save(out/'results.json',all_results);print(json.dumps(result),flush=True)
    save(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    try:run(a.out,a.data)
    except Exception as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e)))
        raise
