"""Matched full-backbone RD finetuning and learned-step engineering ablation."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from probe import settings, load, write_json, sha
from geometry_codec import Quantizer
from stream import pack


class Steps:
    def __init__(self, model, torch, learn):
        import models.mlicpp as module
        self.model,self.torch,self.module=model,torch,module
        self.log_step=torch.nn.Parameter(torch.zeros(320,device=next(model.parameters()).device),requires_grad=learn)
        self.quantizer=Quantizer(model,None,None)
        self.original_round=module.ste_round
        self.original_gaussian=model.gaussian_conditional.forward
        self.hook=model.g_a.register_forward_pre_hook(lambda *_:self.reset())
        self.reset()
        module.ste_round=self.round
        model.gaussian_conditional.forward=self.gaussian

    def reset(self):
        self.ri=-1;self.gi=0
        self.delta=self.log_step.clamp(-math.log(2),math.log(2)).exp()[None]
        self.quantizer.select('oracle',1.,self.delta)

    def round(self,value):
        if self.ri==-1:
            self.ri=0
            return self.original_round(value)
        i=self.ri//2;self.ri+=1
        assert 0<=i<10
        d=self.delta[:,32*i:32*(i+1),None,None]
        return self.original_round(value/d)*d

    def gaussian(self,value,scale,means=None,training=None):
        i=self.gi;self.gi+=1
        assert 0<=i<10
        d=self.delta[:,32*i:32*(i+1),None,None]
        result,lik=self.original_gaussian(value/d,scale/d,None if means is None else means/d,training)
        return result*d,lik

    def close(self):
        self.hook.remove();self.quantizer.close()
        self.module.ste_round=self.original_round
        self.model.gaussian_conditional.forward=self.original_gaussian


def crop(torch,path,rng,size=256):
    with Image.open(path) as im:
        im=im.convert('RGB');w,h=im.size
        if min(w,h)<size: raise ValueError('Image smaller than crop')
        left=rng.randrange(w-size+1);top=rng.randrange(h-size+1)
        array=np.asarray(im.crop((left,top,left+size,top+size))).copy()
    return torch.from_numpy(array).permute(2,0,1).float()/255


def objective(torch,model,x,lmbda):
    output=model(x)
    bpp=sum(-value.clamp_min(1e-9).log2().sum()/(x.shape[0]*x.shape[2]*x.shape[3]) for value in output['likelihoods'].values())
    mse=(output['x_hat']-x).square().mean()*255**2
    return bpp+lmbda*mse,bpp,mse


def audit(config,out):
    a=json.loads(config.read_text());source=Path(a['data'])
    manifest=json.loads((source/'manifest.json').read_text())
    records=manifest['images'];seen=set();counts={}
    for r in records:
        assert r['name'] not in seen;seen.add(r['name'])
        # Read train/development only. Older test names remain quarantined.
        if r['split'] in ('train','validation'):
            assert sha(source/'images'/r['name'])==r['sha256']
        counts[r['split']]=counts.get(r['split'],0)+1
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config,out/'config.json')
    write_json(out/'data_audit.json',dict(counts=counts,source_manifest_sha256=sha(source/'manifest.json'),
        records=records,existing_test_quarantined=True,pretraining_overlap_unknown=True))
    write_json(out/'status.json',dict(state='prepared',training_started=False))


def execute(args):
    a=SimpleNamespace(**json.loads(args.config.read_text()))
    if args.stage=='audit':return audit(args.config,args.out)
    torch,_=settings();torch.manual_seed(args.seed)
    if args.device=='cpu':torch.cuda.synchronize=lambda *a,**k:None
    else:
        import subprocess
        free=int(subprocess.check_output(['nvidia-smi','-i',str(getattr(a,'gpu_index',2)),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<a.required_free_mib:raise RuntimeError(f'Need {a.required_free_mib} MiB free, found {free}; no training started')
        torch.cuda.set_per_process_memory_fraction(.70)
    m,meta=load(SimpleNamespace(upstream=Path(a.upstream),weights=Path(a.weights),legacy=True,device=args.device),args.quality)
    records=json.loads((Path(a.data)/'manifest.json').read_text())['images']
    train=[r for r in records if r['split']=='train'];rng=random.Random(args.seed)
    params=[p for n,p in m.named_parameters() if not n.endswith('.quantiles')]
    aux=[p for n,p in m.named_parameters() if n.endswith('.quantiles')]
    lmbda={1:.0018,2:.0035,3:.0067}[args.quality]
    folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'backbone.json',meta);shutil.copyfile(args.config,folder/'config.json')
    snapshot=folder/'source_archive';snapshot.mkdir()
    for name in ['end_to_end.py','geometry_codec.py','probe.py','stream.py']:
        shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    q=None
    if args.stage=='smoke':
        m.eval();x=crop(torch,Path(a.data)/'images'/train[0]['name'],rng)[None].to(args.device)
        with torch.no_grad():
            native=m(x);encoded=m.compress(x);decoded=m.decompress(encoded['strings'],encoded['shape'])['x_hat']
            q=Steps(m,torch,True);q.reset();adapt=m(x);enc=m.compress(x)
            assert torch.equal(native['x_hat'],adapt['x_hat'])
            assert encoded['strings']==enc['strings']
            assert torch.equal(decoded,m.decompress(enc['strings'],enc['shape'])['x_hat'])
        m.train();optimizer=torch.optim.Adam(params+[q.log_step],lr=a.learning_rate)
        loss,_,_=objective(torch,m,x,lmbda);loss.backward()
        grad=q.log_step.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        optimizer.step()
        write_json(folder/'smoke.json',dict(identity_forward=True,identity_stream=True,identity_decode=True,
            log_step_gradient_l1=float(grad.abs().sum()),updated_log_step_l1=float(q.log_step.abs().sum()),
            loss=float(loss),trainable_backbone=sum(p.numel() for p in params)))
        write_json(folder/'status.json',dict(state='complete',stage='smoke',device=args.device))
        q.close();return
    if args.arm=='joint':q=Steps(m,torch,True);params=params+[q.log_step]
    opt=torch.optim.Adam(params,lr=a.learning_rate,betas=(.9,.999),eps=1e-8)
    auxopt=torch.optim.Adam(aux,lr=a.aux_learning_rate)
    m.train();start=time.monotonic()
    write_json(folder/'status.json',dict(state='running',stage='train',arm=args.arm))
    with (folder/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            x=torch.stack([crop(torch,Path(a.data)/'images'/rng.choice(train)['name'],rng) for _ in range(a.batch_size)]).to(args.device)
            opt.zero_grad(set_to_none=True);auxopt.zero_grad(set_to_none=True)
            loss,bpp,mse=objective(torch,m,x,lmbda)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
            opt.step();auxopt.zero_grad(set_to_none=True)
            auxloss=m.aux_loss();assert torch.isfinite(auxloss);auxloss.backward();auxopt.step()
            if step==1 or step%25==0:
                entry=dict(step=step,loss=float(loss.detach()),estimated_bpp=float(bpp.detach()),mse=float(mse.detach()),seconds=time.monotonic()-start)
                log.write(json.dumps(entry)+'\n');write_json(folder/'progress.json',entry)
            if step%250==0 or step==a.steps:
                torch.save(dict(state_dict=m.state_dict(),log_step=None if q is None else q.log_step.detach(),
                    optimizer=opt.state_dict(),aux_optimizer=auxopt.state_dict(),step=step,arm=args.arm,seed=args.seed,
                    quality=args.quality),folder/'last.pt.tmp');(folder/'last.pt.tmp').replace(folder/'last.pt')
            if time.monotonic()-start>a.wall_time_limit_s:raise TimeoutError('Training budget')
    write_json(folder/'status.json',dict(state='complete',stage='train',seconds=time.monotonic()-start))
    if q:q.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['audit','smoke','train'],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--arm',choices=['baseline','joint'],default='baseline');p.add_argument('--seed',type=int,default=9041);p.add_argument('--quality',type=int,choices=[1,2,3],default=2)
    args=p.parse_args()
    try:execute(args)
    except Exception as e:
        if args.out.exists():
            write_json(args.out/'failure.json',dict(error=repr(e)))
            write_json(args.out/'status.json',dict(state='failed',error=repr(e)))
        raise
