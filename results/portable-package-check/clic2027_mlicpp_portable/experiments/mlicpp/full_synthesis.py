"""Development-only full synthesis perceptual pilot, with/without conditional GAN.

Frozen real bitstreams; every synthesis parameter is trainable. No test opened.
This is a strong engineering baseline, not a novel compression method.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from probe import ROOT, load, settings, sha, write_json
from stream import unpack


def cfg(out):
    return SimpleNamespace(**json.loads((out/'config.json').read_text()))


def setup(out,label):
    torch,install=settings();torch.manual_seed(cfg(out).seed)
    stop=install(out/(label+'.vram.jsonl'))
    write_json(out/(label+'.runtime.json'),dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),threads=torch.get_num_threads(),
        fp32=True,tf32=False,deterministic=True,cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED')))
    return torch,stop


def args(out):
    a=cfg(out)
    return SimpleNamespace(out=out,upstream=Path(a.upstream),weights=Path(a.weights),legacy=True,cache=Path(a.cache))


def records(out):
    return json.loads((out/'latents/manifest.json').read_text())['records']


def synthesis(out,torch,arm=None):
    sys.path.insert(0,str(Path(cfg(out).upstream)/'MLIC++'))
    from modules.transform import SynthesisTransformOld
    model=SynthesisTransformOld(N=192,M=320)
    path=out/'latents/initial_synthesis.pt' if arm is None else out/arm/'last.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(ck if arm is None else ck['synthesis'])
    return model.cuda()


def discriminator(torch):
    n=torch.nn; layers=[]; channels=6
    for width in [32,64,128,128]:
        layers += [n.utils.spectral_norm(n.Conv2d(channels,width,4,stride=2,padding=1)),n.LeakyReLU(.2)]
        channels=width
    return n.Sequential(*layers,n.utils.spectral_norm(n.Conv2d(channels,1,3,padding=1))).cuda()


def sample(out,row,torch):
    value=torch.load(out/'latents'/row['split']/(row['name']+'.pt'),weights_only=True)
    return value['latent'].cuda(),value['original'].cuda().float()/255,value['base'].cuda().float()/255


def prepare(out):
    torch,stop=setup(out,'prepare'); a=cfg(out); cache=Path(a.cache)
    manifest=json.loads((cache/'manifest.json').read_text())
    selected=[]
    for split,limit in [('train',a.train_limit),('validation',a.validation_limit)]:
        selected += [r for r in manifest['records'] if r['split']==split][:limit]
        (out/'latents'/split).mkdir(parents=True)
    assert all(r['split']!='test' for r in selected)
    model,meta=load(args(out),a.quality); model.requires_grad_(False)
    torch.save(model.g_s.cpu().state_dict(),out/'latents/initial_synthesis.pt');model.cuda()
    captured=[]
    hook=model.g_s.register_forward_pre_hook(lambda m,v:captured.append(v[0].detach().cpu().clone()))
    with torch.no_grad():
        for i,row in enumerate(selected):
            old=torch.load(cache/row['split']/(row['name']+'.pt'),weights_only=True)
            blob=(cache/row['split']/(row['name']+'.bin')).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==row['stream_sha256']
            q,legacy,size,shape,strings=unpack(blob)
            assert q==a.quality and legacy and size==(256,256)
            model.update_resolutions(16,16);captured.clear()
            image=model.decompress(strings,shape)['x_hat'].clamp(0,1).mul(255).round().byte().cpu()
            assert len(captured)==1 and torch.equal(image,old['base'])
            torch.save(dict(latent=captured[0],original=old['original'],base=old['base']),out/'latents'/row['split']/(row['name']+'.pt'))
            write_json(out/'prepare.progress.json',dict(images=i+1,total=len(selected)))
    hook.remove()
    meta.update(records=selected,edge_scale=manifest['edge_scale'],source_cache_manifest_sha256=sha(cache/'manifest.json'),
        initial_synthesis_sha256=sha(out/'latents/initial_synthesis.pt'),adapter_sha256=sha(Path(a.adapter)),
        initial_synthesis_parameters=sum(p.numel() for p in model.g_s.parameters()),development_only=True)
    write_json(out/'latents/manifest.json',meta);stop.set()


def train(out,arm):
    torch,stop=setup(out,arm);a=cfg(out)
    import lpips
    from pilot_model import edge_regions
    model=synthesis(out,torch).train().requires_grad_(True)
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate,betas=(.9,.99))
    d=discriminator(torch) if arm=='gan' else None
    dop=torch.optim.Adam(d.parameters(),lr=a.discriminator_learning_rate,betas=(0.,.99)) if d is not None else None
    rows=[r for r in records(out) if r['split']=='train'];rng=random.Random(a.seed)
    folder=out/arm;folder.mkdir()
    scale=json.loads((out/'latents/manifest.json').read_text())['edge_scale']
    started=time.monotonic()
    with (folder/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            batch=[sample(out,rng.choice(rows),torch) for _ in range(a.batch_size)]
            y,x,base=[torch.cat([v[j] for v in batch]) for j in range(3)]
            opt.zero_grad(set_to_none=True)
            reconstructed=model(y).clamp(0,1)
            mse=(reconstructed-x).square().mean()
            lp=perceptual(reconstructed*2-1,x*2-1).mean()
            edge=edge_regions(reconstructed,x,scale,'rational').mean()
            loss=a.pixel_weight*mse+a.lpips_weight*lp+a.edge_weight*edge
            dl=loss.new_tensor(0.);adversarial=loss.new_tensor(0.)
            if d is not None:
                d.train().requires_grad_(True);dop.zero_grad(set_to_none=True)
                real=d(torch.cat((base,x),1));fake=d(torch.cat((base,reconstructed.detach()),1))
                dl=(1-real).relu().mean()+(1+fake).relu().mean()
                if not torch.isfinite(dl):raise ValueError('Nonfinite discriminator')
                dl.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),1.);dop.step()
                d.eval().requires_grad_(False)
                adversarial=-d(torch.cat((base,reconstructed),1)).mean()
                weight=a.gan_weight*max(0.,min(1.,(step-a.gan_warmup)/max(1,a.gan_warmup)))
                loss=loss+weight*adversarial
            if not torch.isfinite(loss):raise ValueError('Nonfinite generator')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            opt.step()
            if step==1 or step%a.log_every==0 or step==a.steps:
                row=dict(step=step,loss=float(loss.detach()),mse=float(mse.detach()),lpips=float(lp.detach()),
                    edge=float(edge.detach()),adversarial=float(adversarial.detach()),discriminator=float(dl.detach()),
                    seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(row)+'\n');write_json(folder/'progress.json',row)
            if step%250==0 or step==a.steps:
                ck=dict(synthesis=model.state_dict(),step=step,arm=arm,trainable_parameters=sum(p.numel() for p in model.parameters()))
                torch.save(ck,folder/'last.pt.tmp');(folder/'last.pt.tmp').replace(folder/'last.pt')
    write_json(folder/'status.json',dict(state='complete',steps=a.steps,seconds=time.monotonic()-started));stop.set()


def evaluate(out):
    torch,stop=setup(out,'evaluate');a=cfg(out)
    import numpy as np
    from PIL import Image
    from pilot import make_tail
    from pilot_model import Adapter
    from projection_probe import uint8,edge_error
    sys.path.insert(0,str(ROOT/'src'))
    from clic27.eval import metrics as M
    metric=M.Perceptual('cuda');tail=make_tail(args(out),torch)
    adapter=Adapter(hidden=32,amplitude=.25).cuda().eval()
    adapter.load_state_dict(torch.load(a.adapter,map_location='cpu',weights_only=True)['adapter'])
    all_rows=[];summaries=[];validation=[r for r in records(out) if r['split']=='validation']
    for method in ['base','mean','reconstruction','gan']:
        model=synthesis(out,torch,method).eval() if method in ['reconstruction','gan'] else None
        folder=out/'recon'/method;folder.mkdir(parents=True)
        method_rows=[]
        for r in validation:
            latent,x,base=sample(out,r,torch)
            with torch.no_grad():
                if method=='base': image=base
                elif method=='mean':
                    f=torch.load(Path(a.cache)/'validation'/(r['name']+'.pt'),weights_only=True)['feature'].cuda()
                    image=tail(adapter(f)[0])
                else:image=model(latent)
                original=uint8(x);recon=uint8(image)
                values=metric.all(original,recon)
            Image.fromarray(recon).save(folder/(r['name']+'.png'))
            entry=dict(method=method,name=r['name'],bytes=r['bytes'],pixels=r['pixels'],
                pixel_sha256=hashlib.sha256(recon.tobytes()).hexdigest(),edge=edge_error(original,recon),**values)
            method_rows.append(entry);all_rows.append(entry)
        summaries.append(dict(method=method,n=len(method_rows),actual_bpp=8*sum(r['bytes'] for r in method_rows)/sum(r['pixels'] for r in method_rows),
            psnr_pooled=M.pooled_psnr([r['mse'] for r in method_rows],[r['pixels'] for r in method_rows]),
            **{k:float(np.mean([r[k] for r in method_rows])) for k in ['lpips','dists','msssim','edge']}))
        if model is not None:del model
    write_json(out/'validation_metrics.json',all_rows);write_json(out/'summary.json',summaries)
    M.write_csv(summaries,out/'summary.csv');stop.set()


def verify(out):
    torch,stop=setup(out,'fresh');a=cfg(out)
    from pilot_model import Adapter
    from projection_probe import uint8
    expected={(r['method'],r['name']):r['pixel_sha256'] for r in json.loads((out/'validation_metrics.json').read_text())}
    model,_=load(args(out),a.quality);original={k:v.cpu().clone() for k,v in model.g_s.state_dict().items()}
    adapter=Adapter(hidden=32,amplitude=.25).cuda().eval()
    adapter.load_state_dict(torch.load(a.adapter,map_location='cpu',weights_only=True)['adapter'])
    matches=0
    for method in ['base','mean','reconstruction','gan']:
        ck=original if method in ['base','mean'] else torch.load(out/method/'last.pt',map_location='cpu',weights_only=True)['synthesis']
        model.g_s.load_state_dict(ck)
        hook=None
        if method=='mean':hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,v:(adapter(v[0])[0],))
        for r in [r for r in records(out) if r['split']=='validation']:
            _,_,_,shape,strings=unpack((Path(a.cache)/'validation'/(r['name']+'.bin')).read_bytes())
            model.update_resolutions(16,16)
            with torch.no_grad():image=uint8(model.decompress(strings,shape)['x_hat'])
            assert hashlib.sha256(image.tobytes()).hexdigest()==expected[method,r['name']]
            matches+=1
        if hook is not None:hook.remove()
    assert matches==len(expected)
    write_json(out/'fresh.json',dict(matches=matches,source_images_not_used=True));stop.set()


def run(out,path):
    out.mkdir(parents=True,exist_ok=False);(out/'config.json').write_bytes(path.read_bytes())
    names=['full_synthesis.py','probe.py','stream.py','pilot.py','pilot_model.py','projection_probe.py']
    frozen={n:sha(Path(__file__).with_name(n)) for n in names}
    write_json(out/'sources.sha256.json',frozen);started=time.monotonic();budget=cfg(out).wall_time_limit_s
    stages=[('prepare',None),('train','reconstruction'),('train','gan'),('evaluate',None),('verify',None)]
    try:
        for stage,arm in stages:
            write_json(out/'status.json',dict(state='running',stage=stage,arm=arm))
            command=[sys.executable,__file__,'--out',str(out),'--stage',stage]+(['--arm',arm] if arm else [])
            with (out/(stage+('-'+arm if arm else '')+'.log')).open('w') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,budget-(time.monotonic()-started)))
            if proc.returncode:raise RuntimeError(f'{stage}/{arm} exit {proc.returncode}')
        assert all(sha(Path(__file__).with_name(n))==v for n,v in frozen.items())
    except Exception as error:
        write_json(out/'status.json',dict(state='failed',error=str(error),stage=stage,arm=arm));raise
    write_json(out/'status.json',dict(state='complete',development_only=True,seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--arm',choices=['reconstruction','gan'])
    p.add_argument('--stage',choices=['run','prepare','train','evaluate','verify'],required=True)
    a=p.parse_args()
    if a.stage=='run':
        if a.config is None:p.error('--config required')
        run(a.out,a.config)
    elif a.stage=='train':
        if a.arm is None:p.error('--arm required')
        train(a.out,a.arm)
    else:globals()[a.stage](a.out)
