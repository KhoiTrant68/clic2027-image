"""Train, serialize, fresh-decode and evaluate conditional innovation on new images.

This tests conditional correction, not a validated probabilistic prior-conflict
model. All three correction arms use identical 46-byte packet overhead.
"""
import argparse
import hashlib
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

from probe import ROOT,load,settings,sha,write_json
from stream import pack as base_pack,unpack as base_unpack
from innovation_stream import pack,unpack


def cfg(out):return SimpleNamespace(**json.loads((out/'config.json').read_text()))


def setup(out,label):
    torch,install=settings();torch.manual_seed(cfg(out).seed)
    return torch,install(out/(label+'.vram.jsonl'))


def arguments(out):
    a=cfg(out)
    return SimpleNamespace(out=out,upstream=Path(a.upstream),weights=Path(a.weights),legacy=True)


def generator(out,torch):
    a=cfg(out);sys.path.insert(0,str(Path(a.upstream)/'MLIC++'))
    from modules.transform import SynthesisTransformOld
    g=SynthesisTransformOld(N=192,M=320)
    g.load_state_dict(torch.load(Path(a.synthesis_run)/'reconstruction/last.pt',map_location='cpu',weights_only=True)['synthesis'])
    return g.cuda().eval().requires_grad_(False)


def rows(out):return json.loads((out/'cache/manifest.json').read_text())['records']


def data(out,row,torch):
    v=torch.load(out/'cache'/row['split']/(row['name']+'.pt'),weights_only=True)
    return v['feature'].cuda(),v['original'].cuda().float()/255,v['base'].cuda().float()/255


def uint8(value):return value[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()


def build(out,torch,mode,trained=False):
    from innovation_model import Innovation
    a=cfg(out);template=torch.load(out/'cache/template.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(a.seed)
    model=Innovation(template,a.rank,a.grid,mode,a.coefficient_step,a.amplitude).cuda()
    if trained:
        ck=torch.load(out/mode/'last.pt',map_location='cpu',weights_only=True)
        assert ck['step']==a.steps
        model.load_state_dict(ck['innovation'])
    return model


def objective(torch,lp,value,target,scale,a):
    from pilot_model import edge_regions
    mse=(value-target).square().mean()
    perceptual=lp(value*2-1,target*2-1).mean()
    edge=edge_regions(value,target,scale,'rational').mean()
    return a.pixel_weight*mse+a.lpips_weight*perceptual+a.edge_weight*edge


def prepare(out):
    torch,stop=setup(out,'prepare');a=cfg(out)
    old=Path(a.old_cache);prior=Path(a.synthesis_run)
    manifest=json.loads((prior/'latents/manifest.json').read_text())
    selected=[]
    for split,limit in [('train',a.train_limit),('validation',a.validation_limit)]:
        selected += [r for r in manifest['records'] if r['split']==split][:limit]
    for split in ['train','validation','test']:(out/'cache'/split).mkdir(parents=True)
    g=generator(out,torch);captured=[]
    hook=g.synthesis_transform[5].register_forward_pre_hook(lambda m,v:captured.append(v[0].detach().cpu().clone()))
    template=None;count=0;records=[]
    with torch.no_grad():
        for row in selected:
            value=torch.load(prior/'latents'/row['split']/(row['name']+'.pt'),weights_only=True)
            captured.clear();recon=g(value['latent'].cuda()).clamp(0,1).mul(255).round().byte().cpu()
            assert len(captured)==1
            blob=(old/row['split']/(row['name']+'.bin')).read_bytes()
            assert sha(old/row['split']/(row['name']+'.bin'))==row['stream_sha256']
            torch.save(dict(feature=captured[0],original=value['original'],base=recon),out/'cache'/row['split']/(row['name']+'.pt'))
            (out/'cache'/row['split']/(row['name']+'.bin')).write_bytes(blob)
            records.append(row)
            if row['split']=='train':
                template=captured[0].clone() if template is None else template+captured[0]
                count+=1
    hook.remove();torch.save(template/count,out/'cache/template.pt');del g
    test_manifest=json.loads((Path(a.test_data)/'manifest.json').read_text())
    previous=[]
    for p in [Path('/home/jovyan/clic2027-artifacts/data/div2k/pilot224/manifest.json'),Path('/home/jovyan/clic2027-artifacts/data/div2k/cvpr-holdout64/manifest.json')]:
        previous += [r['sha256'] for r in json.loads(p.read_text())['images']]
    assert not {r['sha256'] for r in test_manifest['images']} & set(previous)
    model,meta=load(arguments(out),a.quality)
    model.g_s.load_state_dict(torch.load(prior/'reconstruction/last.pt',map_location='cpu',weights_only=True)['synthesis'])
    hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,v:captured.append(v[0].detach().cpu().clone()))
    with torch.no_grad():
        for i,row in enumerate(test_manifest['images'][:a.test_limit]):
            p=Path(a.test_data)/'images'/row['name'];assert sha(p)==row['sha256']
            with Image.open(p) as im:
                im=im.convert('RGB');rng=random.Random('9041/'+row['name'])
                left,top=rng.randrange(im.width-255),rng.randrange(im.height-255)
                source=np.array(im.crop((left,top,left+256,top+256)))
            x=torch.from_numpy(source.copy()).permute(2,0,1)[None].float().cuda()/255
            model.update_resolutions(16,16);encoded=model.compress(x)
            blob=base_pack(a.quality,True,(256,256),tuple(encoded['shape']),encoded['strings'])
            model.update_resolutions(16,16);captured.clear()
            recon=model.decompress(encoded['strings'],encoded['shape'])['x_hat'].clamp(0,1).mul(255).round().byte().cpu()
            assert len(captured)==1
            name=p.stem+'-innovation'
            torch.save(dict(feature=captured[0],original=torch.from_numpy(source.copy()).permute(2,0,1)[None],base=recon),out/'cache/test'/(name+'.pt'))
            (out/'cache/test'/(name+'.bin')).write_bytes(blob)
            records.append(dict(name=name,source=p.name,source_sha256=row['sha256'],split='test',bytes=len(blob),pixels=65536,
                stream_sha256=hashlib.sha256(blob).hexdigest(),crop=[left,top]))
            write_json(out/'prepare.progress.json',dict(test_images=i+1,total=a.test_limit))
    hook.remove()
    meta.update(records=records,edge_scale=manifest['edge_scale'],synthesis_sha256=sha(prior/'reconstruction/last.pt'),
        source_cache_sha256=sha(prior/'latents/manifest.json'),data_sha256=sha(Path(a.test_data)/'manifest.json'),
        template_sha256=sha(out/'cache/template.pt'),test_source_disjoint_previous_288=True)
    write_json(out/'cache/manifest.json',meta);stop.set()


def train(out,mode):
    torch,stop=setup(out,mode);a=cfg(out)
    import lpips
    g=generator(out,torch);tail=g.synthesis_transform[5:]
    model=build(out,torch,mode).train()
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate,betas=(.9,.99))
    selected=[r for r in rows(out) if r['split']=='train'];rng=random.Random(a.seed)
    scale=json.loads((out/'cache/manifest.json').read_text())['edge_scale']
    folder=out/mode;folder.mkdir();started=time.monotonic()
    write_json(folder/'model.json',dict(mode=mode,parameters=sum(p.numel() for p in model.parameters()),
        decoder='fixed trained full-synthesis; decoder basis input is sole difference in primary contrast'))
    with (folder/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            batch=[data(out,rng.choice(selected),torch) for _ in range(a.batch_size)]
            f,x,base=[torch.cat([v[j] for v in batch]) for j in range(3)]
            opt.zero_grad(set_to_none=True)
            c=model.propose(f,x,base);output=model.render(f,base,tail,model.quantize(c,ste=True))
            loss=objective(torch,lp,output,x,scale,a)
            if not torch.isfinite(loss):raise ValueError('Nonfinite innovation loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            opt.step()
            if step==1 or step%25==0 or step==a.steps:
                entry=dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(entry)+'\n');write_json(folder/'progress.json',entry)
        torch.save(dict(innovation=model.state_dict(),step=a.steps,mode=mode),folder/'last.pt')
    write_json(folder/'status.json',dict(state='complete',steps=a.steps));stop.set()


def choose(torch,model,tail,lp,f,x,base,scale,a):
    with torch.no_grad():
        initial=model.propose(f,x,base)
        choices=[initial,torch.zeros_like(initial)]
        scores=[float(objective(torch,lp,model.render(f,base,tail,model.quantize(c)),x,scale,a)) for c in choices]
        best=choices[int(np.argmin(scores))].clone();best_score=min(scores)
    c=torch.nn.Parameter(initial.detach().clone());opt=torch.optim.Adam([c],lr=a.search_lr)
    for _ in range(a.search_steps):
        opt.zero_grad(set_to_none=True)
        value=model.render(f,base,tail,model.quantize(c,ste=True))
        loss=objective(torch,lp,value,x,scale,a);loss.backward();opt.step()
        with torch.no_grad():
            c.clamp_(-127*a.coefficient_step,127*a.coefficient_step)
            score=float(objective(torch,lp,model.render(f,base,tail,model.quantize(c)),x,scale,a))
            if score<best_score:best,best_score=c.detach().clone(),score
    return (best/a.coefficient_step).round().clamp(-127,127).to(torch.int8).cpu().numpy()[0]


def infer(out):
    torch,stop=setup(out,'infer');a=cfg(out)
    import lpips
    sys.path.insert(0,str(ROOT/'src'))
    from clic27.eval import metrics as M
    from projection_probe import edge_error
    metric=M.Perceptual('cuda');g=generator(out,torch);tail=g.synthesis_transform[5:]
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    scale=json.loads((out/'cache/manifest.json').read_text())['edge_scale']
    selected=[r for r in rows(out) if r['split']=='test'];results=[]
    for mode in ['base']+a.modes:
        model=build(out,torch,mode,True).eval().requires_grad_(False) if mode!='base' else None
        folder=out/'streams'/mode;folder.mkdir(parents=True)
        reconfolder=out/'recon'/mode;reconfolder.mkdir(parents=True)
        for i,row in enumerate(selected):
            f,x,base=data(out,row,torch);blob=(out/'cache/test'/(row['name']+'.bin')).read_bytes()
            if model is None:packet=blob;image=uint8(base)
            else:
                q=choose(torch,model,tail,lp,f,x,base,scale,a)
                packet=pack(blob,mode,q,a.rank,a.grid)
                _,parsed,decoded_q=unpack(packet);assert parsed==mode
                with torch.no_grad():image=uint8(model.render(f,base,tail,torch.from_numpy(decoded_q.copy())[None].float().cuda()*a.coefficient_step))
            (folder/(row['name']+'.bin')).write_bytes(packet);Image.fromarray(image).save(reconfolder/(row['name']+'.png'))
            original=uint8(x)
            with torch.no_grad():values=metric.all(original,image)
            results.append(dict(method=mode,name=row['name'],bytes=len(packet),base_bytes=len(blob),pixels=65536,
                packet_sha256=hashlib.sha256(packet).hexdigest(),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                edge=edge_error(original,image),**values))
            write_json(out/'infer.progress.json',dict(mode=mode,images=i+1,total=len(selected)))
        if model is not None:del model
    write_json(out/'test_metrics.json',results);stop.set()


def latent_search(out):
    # Isolate native entropy coding per image and retain the last legal candidate.
    from innovation_tto_recovery import complete_tto
    complete_tto(out)


def verify(out):
    torch,stop=setup(out,'fresh');a=cfg(out)
    model,_=load(arguments(out),a.quality)
    model.g_s.load_state_dict(torch.load(Path(a.synthesis_run)/'reconstruction/last.pt',map_location='cpu',weights_only=True)['synthesis'])
    tail=model.g_s.synthesis_transform[5:];captured=[]
    hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,v:captured.append(v[0].detach()))
    heads={mode:build(out,torch,mode,True).eval().requires_grad_(False) for mode in a.modes}
    matches=0
    for row in json.loads((out/'test_metrics.json').read_text()):
        method=row['method'];packet=(out/'streams'/method/(row['name']+'.bin')).read_bytes()
        assert hashlib.sha256(packet).hexdigest()==row['packet_sha256']
        if method in ['base','latent_tto']:blob=packet;q=None
        else:
            blob,parsed,q=unpack(packet);assert parsed==method
        quality,legacy,size,shape,strings=base_unpack(blob)
        assert quality==a.quality and legacy and size==(256,256)
        model.update_resolutions(16,16);captured.clear()
        with torch.no_grad():
            base=model.decompress(strings,shape)['x_hat'].clamp(0,1)
            feature=captured[0]
            if q is None:image=uint8(base)
            else:
                # Pixel residual uses exactly the uint8 base used by the encoder.
                rounded=base.mul(255).round()/255
                coefficient=torch.from_numpy(q.copy())[None].float().cuda()*a.coefficient_step
                image=uint8(heads[method].render(feature,rounded,tail,coefficient))
        assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256'],(method,row['name'])
        matches+=1
    hook.remove();write_json(out/'fresh.json',dict(matches=matches,source_images_not_used=True));stop.set()


def summarize(out):
    sys.path.insert(0,str(ROOT/'src'));from clic27.eval import metrics as M
    data_rows=json.loads((out/'test_metrics.json').read_text());summary=[]
    for method in ['base']+cfg(out).modes+['latent_tto']:
        selected=[r for r in data_rows if r['method']==method]
        summary.append(dict(method=method,n=len(selected),actual_bpp=8*sum(r['bytes'] for r in selected)/sum(r['pixels'] for r in selected),
            psnr_pooled=M.pooled_psnr([r['mse'] for r in selected],[r['pixels'] for r in selected]),
            **{key:float(np.mean([r[key] for r in selected])) for key in ['lpips','dists','msssim','edge']}))
    write_json(out/'summary.json',summary);M.write_csv(summary,out/'summary.csv')


def run(out,path):
    out.mkdir(parents=True,exist_ok=False);(out/'config.json').write_bytes(path.read_bytes())
    names=['innovation_pilot.py','innovation_model.py','innovation_stream.py','probe.py','stream.py','pilot_model.py','innovation_tto_recovery.py']
    frozen={name:sha(Path(__file__).with_name(name)) for name in names}
    write_json(out/'sources.sha256.json',frozen);started=time.monotonic();a=cfg(out)
    stages=[('prepare',None)]+[('train',m) for m in a.modes]+[('infer',None),('latent_search',None),('verify',None),('summarize',None)]
    try:
        for stage,mode in stages:
            write_json(out/'status.json',dict(state='running',stage=stage,mode=mode))
            command=[sys.executable,__file__,'--out',str(out),'--stage',stage]+(['--mode',mode] if mode else [])
            with (out/(stage+('-'+mode if mode else '')+'.log')).open('w') as log:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,a.wall_time_limit_s-(time.monotonic()-started)))
            if p.returncode:raise RuntimeError(f'{stage}/{mode}: {p.returncode}')
        assert all(sha(Path(__file__).with_name(n))==v for n,v in frozen.items())
    except Exception as e:
        write_json(out/'status.json',dict(state='failed',stage=stage,mode=mode,error=str(e)));raise
    write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-started,test_images=a.test_limit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--mode',choices=['conditional','fixed','pixel'])
    p.add_argument('--stage',choices=['run','prepare','train','infer','latent_search','verify','summarize'],required=True)
    a=p.parse_args()
    if a.stage=='run':
        if a.config is None:p.error('--config required')
        run(a.out,a.config)
    elif a.stage=='train':
        if a.mode is None:p.error('--mode required')
        train(a.out,a.mode)
    else:globals()[a.stage](a.out)
