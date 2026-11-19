"""Train an encoder-side rollout cost critic; split by source, no backbone update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from probe import settings, load, write_json
from joint_symbol_gpu import JointIntervention
from lookahead_pilot import encode


def memory_check():
    total,used=map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.total,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip().split(','))
    if total-used<1536: raise RuntimeError('GPU headroom below 1.5 GiB')
    return used


def state_features(trace):
    features=[]
    for row in trace:
        for key in ['symbols','indexes','residual']:
            value=np.asarray(row[key],dtype=np.float32).reshape(32,-1)
            features.extend([value.mean(1),value.std(1)])
    return np.concatenate(features)


def action_features(trace,action):
    positions=action['positions'];step=action['step'];rows=[]
    for p,d in zip(positions,action['deltas']):
        rows.append([p//128/31,(p%128)//8/15,p%8/7,d,
            trace[step]['symbols'][p],trace[step]['indexes'][p],trace[step]['residual'][p]])
    padded=np.zeros((8,7),dtype=np.float32);padded[:len(rows)]=rows
    return np.concatenate([[step/19,len(rows)/8],padded.flatten()]).astype(np.float32)


def prepare(out,a,torch):
    ref=Path(a.rollout_run)
    assert json.loads((ref/'status.json').read_text())['state']=='complete'
    c=SimpleNamespace(**json.loads((ref/'config.json').read_text()))
    inventory=json.loads((ref/'inventory.json').read_text())
    train_sources={inventory[i]['source'] for i in a.train_workers}
    val_sources={inventory[i]['source'] for i in a.validation_workers}
    assert train_sources.isdisjoint(val_sources)
    model,meta=load(SimpleNamespace(upstream=Path(c.upstream),weights=Path(c.weights),legacy=True),2)
    model.requires_grad_(False)
    hook=JointIntervention(model)
    features=[];labels=[];split=[];groups=[];identity_checks=0
    for worker in a.train_workers+a.validation_workers:
        record=inventory[worker];folder=ref/f'worker-{worker}'
        summary=json.loads((folder/'summary.json').read_text())
        log=[json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
        v=torch.load(Path(c.reference_run)/'cache'/(record['name']+'.pt'),map_location='cpu',weights_only=True)
        x=v['original'].float().cuda()/255
        actions=[];current=summary['baseline']['j']
        with torch.no_grad():
            blob,_,trace,_=encode(model,hook,x,[])
            assert hashlib.sha256(blob).hexdigest()==summary['baseline']['stream_sha256']
            identity_checks+=1
            for start in range(0,len(log),5):
                if time.monotonic()>a.deadline: raise TimeoutError('Training run wall-time limit')
                memory_check()
                batch=log[start:start+5]
                assert len(batch)==5 and len({(r['sweep'],r['step']) for r in batch})==1
                shared=state_features(trace)
                # Include no-change action (zero target) in each ranking group.
                for r in batch:
                    features.append(np.concatenate([shared,action_features(trace,r['action'])]))
                    labels.append(r['j']-current)
                    split.append(worker in a.validation_workers)
                    groups.append([worker,r['sweep'],r['step']])
                features.append(np.concatenate([shared,np.zeros(58,dtype=np.float32)]))
                labels.append(0.);split.append(worker in a.validation_workers);groups.append(groups[-1])
                winner=min(batch,key=lambda r:r['j'])
                if winner['j']<current:
                    actions.append(winner['action']);current=winner['j']
                    blob,_,trace,_=encode(model,hook,x,actions)
                    assert hashlib.sha256(blob).hexdigest()==winner['stream_sha256'], 'Replay mismatch'
                write_json(out/'prepare.progress.json',dict(worker=worker,groups=(start//5)+1,examples=len(labels)))
        print('prepared source',worker,'examples',len(labels),flush=True)
    hook.close()
    np.savez_compressed(out/'dataset.npz',x=np.stack(features),y=np.asarray(labels,dtype=np.float32),
        validation=np.asarray(split,dtype=bool),groups=np.asarray(groups))
    write_json(out/'dataset.json',dict(train_sources=sorted(train_sources),validation_sources=sorted(val_sources),
        identity_checks=identity_checks,examples=len(labels),backbone=meta,
        source_role='reused development; source-disjoint training and validation, not independent test'))
    del hook,model
    torch.cuda.empty_cache()


def train(out,a,torch):
    torch.manual_seed(a.seed)
    d=np.load(out/'dataset.npz');mask=d['validation'];device='cuda'
    raw=torch.tensor(d['x'],device=device);target=torch.tensor(d['y'],device=device)
    tr=torch.tensor(~mask,device=device);va=torch.tensor(mask,device=device)
    mean=raw[tr].mean(0);std=raw[tr].std(0,unbiased=False).clamp_min(1e-4)
    x=((raw-mean)/std).clamp(-20,20);scale=target[tr].std(unbiased=False).clamp_min(1e-6)
    y=target/scale
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],a.hidden),torch.nn.GELU(),
        torch.nn.Linear(a.hidden,a.hidden),torch.nn.GELU(),torch.nn.Linear(a.hidden,1)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4,betas=(0.9,0.999),eps=1e-8)
    initial=[p.detach().clone() for p in model.parameters()]
    train_idx=torch.where(tr)[0];best=float('inf');start=time.monotonic()
    def metrics():
        model.eval()
        with torch.no_grad():
            prediction=model(x).squeeze(1)*scale
        result={}
        for name,selection in [('train',tr),('validation',va)]:
            result[name+'_mse']=float(((prediction[selection]-target[selection])**2).mean())
            result[name+'_zero_mse']=float((target[selection]**2).mean())
        result['validation_mean_predictor_mse']=float(((target[va]-target[tr].mean())**2).mean())
        pred=prediction.cpu().numpy();truth=d['y'];group=d['groups'];regrets=[];zeros=[];agreements=[]
        for key in np.unique(group[mask],axis=0):
            idx=np.where((group==key).all(1))[0]
            chosen=idx[np.argmin(pred[idx])];oracle=idx[np.argmin(truth[idx])]
            regrets.append(float(truth[chosen]-truth[oracle]));zeros.append(float(-truth[oracle]))
            agreements.append(bool(truth[chosen]==truth[oracle]))
        result.update(validation_ranking_regret=float(np.mean(regrets)),
            validation_no_change_regret=float(np.mean(zeros)),validation_top1=float(np.mean(agreements)))
        return result
    with (out/'train.jsonl').open('w',buffering=1) as log:
        before=metrics();write_json(out/'initial_metrics.json',before)
        for step in range(1,a.steps+1):
            model.train();idx=train_idx[torch.randint(len(train_idx),(a.batch_size,),device=device)]
            loss=torch.nn.functional.mse_loss(model(x[idx]).squeeze(1),y[idx])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step==1 or step%100==0:
                if time.monotonic()>a.deadline: raise TimeoutError('Training run wall-time limit')
                used=memory_check();m=metrics();entry=dict(step=step,loss=float(loss),elapsed_s=time.monotonic()-start,gpu_used_mib=used,**m)
                log.write(json.dumps(entry)+'\n');write_json(out/'train.progress.json',entry)
                print('train',step,'loss',float(loss),'validation regret',m['validation_ranking_regret'],flush=True)
                if m['validation_ranking_regret']<best:
                    best=m['validation_ranking_regret']
                    torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),std=std.cpu(),target_scale=scale.cpu(),
                        input_dim=x.shape[1],hidden=a.hidden,step=step,metrics=m),out/'critic_best.pt')
        changed=float(sum((p-q).abs().sum() for p,q in zip(model.parameters(),initial)))
        assert changed>0
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean.cpu(),std=std.cpu(),
            target_scale=scale.cpu(),step=a.steps,input_dim=x.shape[1],hidden=a.hidden),out/'critic_last.pt')
        write_json(out/'training_summary.json',dict(steps=a.steps,parameters=sum(p.numel() for p in model.parameters()),
            weight_change_l1=changed,best_validation_regret=best,final=metrics(),
            caveat='Offline ranking on teacher states; not rollout policy evaluation or a codec RD result'))


def main(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False);shutil.copyfile(args.config,out/'config.json')
    a=SimpleNamespace(**json.loads((out/'config.json').read_text()))
    snapshot=out/'source_archive';snapshot.mkdir()
    for name in ['critic_train.py','joint_symbol_gpu.py','lookahead_pilot.py','probe.py','stream.py']:
        shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    start=time.monotonic()
    a.deadline=start+a.wall_time_limit_s
    try:
        memory_check();torch,_=settings();torch.cuda.set_per_process_memory_fraction(a.allocator_fraction)
        write_json(out/'status.json',dict(state='running',stage='prepare'))
        prepare(out,a,torch)
        write_json(out/'status.json',dict(state='running',stage='train'))
        train(out,a,torch)
        write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-start))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=repr(exc),elapsed_s=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    main(p.parse_args())
