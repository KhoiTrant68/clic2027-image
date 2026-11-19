"""Paired decoder-only adaptation: fixed rate versus mixed quantization factors."""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from probe import settings,load,sha,write_json
from geometry_codec import Quantizer
from end_to_end import crop
from spatial_synthesis import SpatialSynthesis
from entropy_synthesis import EntropySynthesis, observe_cells


def run(a):
    c=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.config,a.out/'config.json');shutil.copyfile(__file__,a.out/'source.py')
    try:
        if a.device == 'cpu' and not a.smoke:
            raise ValueError('Use CPU for the smoke test only')
        torch,_=settings()
        if a.device == 'cpu':
            # Upstream compress/decompress synchronize CUDA even for CPU tensors.
            torch.cuda.synchronize = lambda *args, **kwargs: None
        if a.device == 'cuda':
            free=int(subprocess.check_output(['nvidia-smi','-i',str(c['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<12288:raise RuntimeError('Need 12 GiB free before starting')
            torch.cuda.set_per_process_memory_fraction(8192*2**20/torch.cuda.get_device_properties(0).total_memory)
        model,meta=load(SimpleNamespace(upstream=Path(c['upstream']),weights=Path(c['weights']),legacy=True,device=a.device),2)
        initial=torch.load(c['synthesis'],map_location='cpu',weights_only=True)['synthesis']
        model.g_s.load_state_dict(initial);model.requires_grad_(False)
        write_json(a.out/'runtime.json',dict(backbone=meta,torch=torch.__version__,device=a.device,architecture=a.architecture,
            gpu=torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
            synthesis_sha256=sha(Path(c['synthesis'])),smoke=a.smoke,objective='10*MSE + LPIPS; same in both arms',factors=[1,2,4]))
        source=Path(c['data']);records=json.loads((source/'manifest.json').read_text())['images']
        train_count = a.train_images if a.train_images is not None else (2 if a.smoke else 160)
        dev_count = a.dev_images if a.dev_images is not None else (1 if a.smoke else 32)
        train=[r for r in records if r['split']=='train'][:train_count]
        dev=[r for r in records if r['split']=='validation'][:dev_count]
        if len(train) != train_count or len(dev) != dev_count:
            raise ValueError('Dataset has fewer images than requested')
        data=a.out/'development-center256';(data/'images').mkdir(parents=True)
        devrecords=[]
        for r in dev:
            path=source/'images'/r['name'];assert sha(path)==r['sha256']
            with Image.open(path) as im:
                im=im.convert('RGB');w,h=im.size;x=(w-256)//2;y=(h-256)//2
                im.crop((x,y,x+256,y+256)).save(data/'images'/r['name'])
            devrecords.append(dict(name=r['name'],sha256=sha(data/'images'/r['name'])))
        write_json(data/'manifest.json',dict(images=devrecords,role='reused development crops, not test'))
        quant=Quantizer(model,None,None);captured=[]
        hook=model.g_s.register_forward_pre_hook(lambda m,v:captured.append(v[0].detach().cpu().clone()))
        cached=[];rng=random.Random(1984)
        with torch.no_grad():
            for i,r in enumerate(train):
                path=source/'images'/r['name'];assert sha(path)==r['sha256']
                x=crop(torch,path,rng)[None].to(a.device);latents={}; observations={}
                for factor in [1,2,4]:
                    quant.select('uniform',factor);encoded=model.compress(x);captured.clear()
                    if a.entropy:
                        with observe_cells(model, factor) as evidence:
                            model.decompress(encoded['strings'], encoded['shape'])
                            observations[factor] = evidence().cpu()
                    else:
                        model.decompress(encoded['strings'], encoded['shape'])
                    assert len(captured)==1
                    latents[factor]=captured[0]
                cached.append((x.cpu(),latents,observations))
                write_json(a.out/'status.json',dict(state='running',stage='cache_real_entropy_latents',images=i+1,total=len(train)))
        hook.remove();quant.close()
        import lpips
        perceptual=lpips.LPIPS(net='alex',verbose=False).to(a.device).eval().requires_grad_(False)
        steps = a.steps if a.steps is not None else (2 if a.smoke else 1000)
        if a.entropy:
            arms = ['plain', 'entropy-blind', 'entropy']
        elif a.architecture:
            arms = ['plain', 'global', 'local']
        else:
            arms = ['single', 'mixed']
        synthesis = model.g_s
        for arm in arms:
            folder=a.out/arm;folder.mkdir()
            torch.manual_seed(9041);rng=random.Random(9041);frng=random.Random(77)
            synthesis.load_state_dict(initial)
            architecture = None
            model.g_s = synthesis
            if arm in {'global', 'local'}:
                architecture = dict(channels=cached[0][1][1].shape[1], mode=arm)
                model.g_s = SpatialSynthesis(synthesis, **architecture).to(a.device)
                latent = cached[0][1][1].to(a.device)
                with torch.no_grad():
                    torch.testing.assert_close(model.g_s(latent), synthesis(latent), rtol=0, atol=0)
            elif arm in {'entropy', 'entropy-blind'}:
                architecture = dict(channels=cached[0][1][1].shape[1], mode=arm)
                model.g_s = EntropySynthesis(synthesis, **architecture).to(a.device)
                latent = cached[0][1][1].to(a.device)
                evidence = cached[0][2][1].to(a.device)
                with torch.no_grad():
                    torch.testing.assert_close(model.g_s(latent, evidence, 1), synthesis(latent), rtol=0, atol=0)
            model.g_s.train().requires_grad_(True)
            opt=torch.optim.Adam(model.g_s.parameters(),lr=1e-5,betas=(.9,.99));started=time.monotonic()
            with (folder/'train.jsonl').open('w',buffering=1) as log:
                for step in range(1,steps+1):
                    samples=[cached[rng.randrange(len(cached))] for _ in range(2)]
                    factors=[1 if arm=='single' else frng.choice([1,2,4]) for _ in samples]
                    x=torch.cat([r[0] for r in samples]).to(a.device);y=torch.cat([r[1][f] for r,f in zip(samples,factors)]).to(a.device)
                    opt.zero_grad(set_to_none=True)
                    if arm in {'entropy', 'entropy-blind'}:
                        evidence = torch.cat([r[2][f] for r, f in zip(samples, factors)]).to(a.device)
                        factor_batch = torch.as_tensor(factors, dtype=y.dtype, device=y.device).reshape(-1, 1, 1, 1)
                        recon = model.g_s(y, evidence, factor_batch).clamp(0, 1)
                    else:
                        recon = model.g_s(y).clamp(0, 1)
                    loss=10*(recon-x).square().mean()+perceptual(recon*2-1,x*2-1).mean()
                    assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.g_s.parameters(),1.);assert torch.isfinite(norm);opt.step()
                    if step==1 or step%25==0 or step==steps:
                        entry=dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started)
                        log.write(json.dumps(entry)+'\n');write_json(folder/'progress.json',entry)
                        write_json(a.out/'status.json',dict(state='running',stage='train',arm=arm,**entry))
                    if step%250==0 or step==steps:
                        torch.save(dict(synthesis=model.g_s.state_dict(),step=step,arm=arm,architecture=architecture),folder/'last.pt.tmp')
                        (folder/'last.pt.tmp').replace(folder/'last.pt')
                    if time.monotonic()-started>7200:raise TimeoutError('Two-hour training budget per arm')
        del opt,model,synthesis,perceptual,cached,x,y,recon,loss
        if a.device == 'cuda':
            torch.cuda.empty_cache()
        summaries=[]
        for factor in [1,2,4]:
            for arm in ['initial', *arms]:
                folder=a.out/f'eval-{arm}-factor{factor}'
                ck=Path(c['synthesis']) if arm=='initial' else a.out/arm/'last.pt'
                args=[sys.executable,str(Path(__file__).with_name('end_to_end_eval.py')),'--config',str(a.config),
                    '--data',str(data),'--out',str(folder),'--quality','2','--synthesis',str(ck),'--quantizer-factor',str(factor),
                    '--device',a.device]
                write_json(a.out/'status.json',dict(state='running',stage='eval',arm=arm,factor=factor))
                with (a.out/'eval.log').open('a',buffering=1) as log:
                    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run(args+['--fresh'],stdout=log,stderr=subprocess.STDOUT,check=True)
                rows=json.loads((folder/'metrics.json').read_text())
                if arm!='initial':
                    base=json.loads((a.out/f'eval-initial-factor{factor}'/'metrics.json').read_text())
                    assert [r['stream_sha256'] for r in base]==[r['stream_sha256'] for r in rows]
                summaries.append(dict(arm=arm,factor=factor,**{k:float(np.mean([r[k] for r in rows])) for k in ['bpp','psnr','lpips','dists']}))
                write_json(a.out/'comparison.json',summaries)
        write_json(a.out/'status.json',dict(state='complete',smoke=a.smoke,training_steps_per_arm=steps,fresh_matches=3*(1+len(arms))*len(dev)))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    ablation = p.add_mutually_exclusive_group()
    ablation.add_argument('--architecture',action='store_true')
    ablation.add_argument('--entropy',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--steps', type=int)
    p.add_argument('--train-images', type=int)
    p.add_argument('--dev-images', type=int)
    args = p.parse_args()
    for name in ('steps', 'train_images', 'dev_images'):
        value = getattr(args, name)
        if value is not None and (not args.smoke or value < 1):
            p.error(f'--{name.replace("_", "-")} requires --smoke and a positive value')
    run(args)
