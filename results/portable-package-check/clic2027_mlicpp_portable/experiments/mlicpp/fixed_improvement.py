"""Bounded 2x2 pilot: fixed-basis training duration x encoder search duration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np
from PIL import Image

import innovation_pilot as I
from innovation_stream import pack,unpack
from probe import sha,write_json
from stream import unpack as base_unpack


def arms(out):
    a=I.cfg(out)
    return [(f'{key}_s{steps}',key,steps) for key in ['reference','long'] for steps in a.search_budgets]


def prepare(out):
    I.prepare(out)
    import torch
    a=I.cfg(out);prior=Path(a.reference_run)
    template=torch.load(out/'cache/template.pt',weights_only=True)
    reference=torch.load(out/'reference_fixed/last.pt',map_location='cpu',weights_only=True)['innovation']['template']
    assert torch.equal(template,reference)
    previous_records=[r for r in I.rows(prior) if r['split'] in ['train','validation']]
    current_records=[r for r in I.rows(out) if r['split'] in ['train','validation']]
    assert previous_records==current_records
    test=[r for r in I.rows(out) if r['split']=='test'];hashes={r['source_sha256'] for r in test}
    previous=set()
    if not a.smoke:
        for path in a.excluded_manifests:
            previous.update(r['sha256'] for r in json.loads(Path(path).read_text())['images'])
        assert len(previous)==322 and not hashes&previous
    source={r['name']:r for r in json.loads((Path(a.test_data)/'manifest.json').read_text())['images']}
    for row in test:
        path=Path(a.test_data)/'images'/row['source'];raw=path.read_bytes()
        assert sha(path)==row['source_sha256']==source[row['source']]['sha256']
        assert zlib.crc32(raw)==source[row['source']]['crc32']
        left,top=row['crop']
        with Image.open(path) as im:actual=np.array(im.convert('RGB').crop((left,top,left+256,top+256)))
        target=torch.load(out/'cache/test'/(row['name']+'.pt'),weights_only=True)['original'][0].permute(1,2,0).numpy()
        assert np.array_equal(actual,target)
    write_json(out/'source_audit.json',dict(images=len(test),previous_sources=len(previous),disjoint_previous=not bool(hashes&previous),
        source_crc_and_sha_verified=True,cached_target_matches_source_crop=True,template_equal_reference=True,
        train_validation_manifest_equal_reference=True,author_pretraining_overlap_audited=False,smoke=a.smoke))


def head(out,torch,key):
    from innovation_model import Innovation
    a=I.cfg(out)
    path=out/'reference_fixed/last.pt' if key=='reference' else out/'fixed/last.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    assert ck['step']==(a.reference_steps if key=='reference' else a.steps)
    model=Innovation(ck['innovation']['template'],a.rank,a.grid,'fixed',a.coefficient_step,a.amplitude).cuda()
    model.load_state_dict(ck['innovation'])
    return model.eval().requires_grad_(False)


def search(torch,model,tail,lp,f,x,base,scale,a):
    """Keep the original choose() trajectory and snapshot two fixed budgets."""
    with torch.no_grad():
        initial=model.propose(f,x,base);choices=[initial,torch.zeros_like(initial)]
        scores=[float(I.objective(torch,lp,model.render(f,base,tail,model.quantize(c)),x,scale,a)) for c in choices]
        best=choices[int(np.argmin(scores))].clone();best_score=min(scores)
    c=torch.nn.Parameter(initial.detach().clone());opt=torch.optim.Adam([c],lr=a.search_lr)
    torch.cuda.synchronize();started=time.monotonic();snapshots={}
    for step in range(1,max(a.search_budgets)+1):
        opt.zero_grad(set_to_none=True)
        value=model.render(f,base,tail,model.quantize(c,ste=True))
        loss=I.objective(torch,lp,value,x,scale,a)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite coefficient search loss')
        loss.backward()
        if c.grad is None or not torch.isfinite(c.grad).all():raise FloatingPointError('Nonfinite coefficient gradient')
        opt.step()
        with torch.no_grad():
            c.clamp_(-127*a.coefficient_step,127*a.coefficient_step)
            score=float(I.objective(torch,lp,model.render(f,base,tail,model.quantize(c)),x,scale,a))
            if not np.isfinite(score):raise FloatingPointError('Nonfinite quantized score')
            if score<best_score:best,best_score=c.detach().clone(),score
        if step in a.search_budgets:
            torch.cuda.synchronize()
            snapshots[step]=dict(q=(best/a.coefficient_step).round().clamp(-127,127).to(torch.int8).cpu().numpy()[0],
                selection_objective=best_score,search_seconds=time.monotonic()-started)
    ordered=[snapshots[s]['selection_objective'] for s in sorted(snapshots)]
    assert all(y<=x for x,y in zip(ordered,ordered[1:]))
    return snapshots


def infer(out):
    torch,stop=I.setup(out,'infer');a=I.cfg(out)
    import lpips
    sys.path.insert(0,str(I.ROOT/'src'))
    from clic27.eval import metrics as M
    from projection_probe import edge_error
    metric=M.Perceptual('cuda');g=I.generator(out,torch);tail=g.synthesis_transform[5:]
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    scale=json.loads((out/'cache/manifest.json').read_text())['edge_scale']
    selected=[r for r in I.rows(out) if r['split']=='test'];results=[]
    for key in ['reference','long']:
        model=head(out,torch,key)
        for i,row in enumerate(selected):
            f,x,base=I.data(out,row,torch);blob=(out/'cache/test'/(row['name']+'.bin')).read_bytes()
            snapshots=search(torch,model,tail,lp,f,x,base,scale,a)
            if a.parity_check and i==0:
                baseline=I.choose(torch,model,tail,lp,f,x,base,scale,SimpleNamespace(**{**vars(a),'search_steps':min(a.search_budgets)}))
                assert np.array_equal(baseline,snapshots[min(a.search_budgets)]['q'])
            pairs=[('base',blob,I.uint8(base),None)] if key=='reference' else []
            for budget,record in snapshots.items():
                packet=pack(blob,'fixed',record['q'],a.rank,a.grid)
                _,mode,q=unpack(packet);assert mode=='fixed' and len(packet)==len(blob)+46
                with torch.no_grad():image=I.uint8(model.render(f,base,tail,torch.from_numpy(q.copy())[None].float().cuda()*a.coefficient_step))
                pairs.append((f'{key}_s{budget}',packet,image,{k:v for k,v in record.items() if k!='q'}))
            for method,packet,image,search_record in pairs:
                for folder in ['streams','recon']:(out/folder/method).mkdir(parents=True,exist_ok=True)
                (out/'streams'/method/(row['name']+'.bin')).write_bytes(packet)
                Image.fromarray(image).save(out/'recon'/method/(row['name']+'.png'))
                original=I.uint8(x)
                with torch.no_grad():values=metric.all(original,image)
                results.append(dict(method=method,name=row['name'],bytes=len(packet),base_bytes=len(blob),pixels=65536,
                    packet_sha256=hashlib.sha256(packet).hexdigest(),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    edge=edge_error(original,image),**(search_record or {}),**values))
            write_json(out/'infer.progress.json',dict(model=key,images=i+1,total=len(selected)))
        del model
    write_json(out/'test_metrics.json',results);stop.set()


def verify(out):
    # Only streams and model assets are read here; no target or feature cache.
    torch,stop=I.setup(out,'fresh');a=I.cfg(out)
    model,_=I.load(I.arguments(out),a.quality)
    model.g_s.load_state_dict(torch.load(Path(a.synthesis_run)/'reconstruction/last.pt',map_location='cpu',weights_only=True)['synthesis'])
    tail=model.g_s.synthesis_transform[5:];captured=[]
    hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,v:captured.append(v[0].detach()))
    heads={key:head(out,torch,key) for key in ['reference','long']};matches=0
    for row in json.loads((out/'test_metrics.json').read_text()):
        method=row['method'];packet=(out/'streams'/method/(row['name']+'.bin')).read_bytes()
        assert sha(out/'streams'/method/(row['name']+'.bin'))==row['packet_sha256']
        if method=='base':blob=packet;q=None
        else:
            blob,mode,q=unpack(packet);assert mode=='fixed'
        quality,legacy,size,shape,strings=base_unpack(blob)
        assert quality==a.quality and legacy and size==(256,256)
        model.update_resolutions(16,16);captured.clear()
        with torch.no_grad():
            base=model.decompress(strings,shape)['x_hat'].clamp(0,1)
            if q is None:image=I.uint8(base)
            else:
                key=method.split('_s')[0];coefficient=torch.from_numpy(q.copy())[None].float().cuda()*a.coefficient_step
                image=I.uint8(heads[key].render(captured[0],base.mul(255).round()/255,tail,coefficient))
        assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256'],(method,row['name'])
        matches+=1
    hook.remove();write_json(out/'fresh.json',dict(matches=matches,source_images_not_used=True));stop.set()


def run(out,config):
    out.mkdir(parents=True,exist_ok=False);shutil.copyfile(config,out/'config.json');a=I.cfg(out)
    if not a.smoke:assert a.steps==5000 and a.search_budgets==[40,160] and a.test_limit==32
    names=['fixed_improvement.py','analyze_fixed_improvement.py','innovation_pilot.py','innovation_model.py','innovation_stream.py','probe.py','stream.py','pilot_model.py']
    (out/'source_archive').mkdir()
    for name in names:shutil.copyfile(Path(__file__).with_name(name),out/'source_archive'/name)
    frozen={name:sha(out/'source_archive'/name) for name in names};write_json(out/'sources.sha256.json',frozen)
    (out/'reference_fixed').mkdir();shutil.copyfile(Path(a.reference_run)/'fixed/last.pt',out/'reference_fixed/last.pt')
    shutil.copyfile(Path(a.reference_run)/'fixed/train.jsonl',out/'reference_fixed/train.jsonl')
    write_json(out/'reference.json',dict(run=a.reference_run,checkpoint_sha256=sha(out/'reference_fixed/last.pt')))
    started=time.monotonic()
    try:
        for stage in ['prepare','train','infer','verify','analyze']:
            write_json(out/'status.json',dict(state='running',stage=stage))
            with (out/(stage+'.log')).open('w') as log:
                command=[sys.executable,__file__,'--stage',stage,'--out',str(out)]
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,a.wall_time_limit_s-(time.monotonic()-started)))
            if p.returncode:raise RuntimeError(f'{stage}: exit {p.returncode}')
        assert all(sha(Path(__file__).with_name(name))==h for name,h in frozen.items())
    except Exception as error:
        write_json(out/'status.json',dict(state='failed',stage=stage,error=str(error)));raise
    write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-started,test_images=a.test_limit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--stage',choices=['run','prepare','train','infer','verify','analyze'],required=True);a=p.parse_args()
    if a.stage=='run':run(a.out,a.config)
    elif a.stage=='prepare':prepare(a.out)
    elif a.stage=='train':I.train(a.out,'fixed')
    elif a.stage=='analyze':
        from analyze_fixed_improvement import main
        main(a.out)
    else:globals()[a.stage](a.out)
