"""Recover completed TTO packets; isolate remaining cases with numerical guards.

Training, coefficients, objective, LR and search steps remain unchanged. Failed
TTO cases retain the legal original stream and stay in the evaluation denominator.
"""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import innovation_pilot as I
from probe import ROOT,sha,write_json
from stream import pack as base_pack


def worker(out,name):
    torch,stop=I.setup(out,'worker-'+name);a=I.cfg(out)
    import lpips
    model,_=I.load(I.arguments(out),a.quality)
    model.g_s.load_state_dict(torch.load(Path(a.synthesis_run)/'reconstruction/last.pt',map_location='cpu',weights_only=True)['synthesis'])
    model.requires_grad_(False)
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    scale=json.loads((out/'cache/manifest.json').read_text())['edge_scale']
    row=next(r for r in I.rows(out) if r['name']==name and r['split']=='test')
    _,x,base=I.data(out,row,torch)
    original_blob=(out/'cache/test'/(name+'.bin')).read_bytes();budget=len(original_blob)+46
    best_blob=original_blob;best_image=I.uint8(base)
    with torch.no_grad():
        best_score=float(I.objective(torch,lp,base,x,scale,a));initial=model.g_a(x).detach()
    parameter=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([parameter],lr=.02)
    hook=model.g_a.register_forward_hook(lambda m,args,result:parameter)
    attempted=feasible=0;reason=None;completed_steps=0;rejected=[]
    progress=out/('worker-'+name+'.json')
    import models.mlicpp as implementation
    native_encoder=implementation.BufferedRansEncoder
    class CheckedEncoder:
        def __init__(self):self.inner=native_encoder()
        def encode_with_indexes(self,symbols,indexes,cdfs,lengths,offsets):
            symbol=np.asarray(symbols,dtype=np.int64);index=np.asarray(indexes,dtype=np.int64)
            cdf=np.asarray(cdfs,dtype=np.int64);length=np.asarray(lengths,dtype=np.int64);offset=np.asarray(offsets,dtype=np.int64)
            if len(symbol)!=len(index) or np.any(index<0) or np.any(index>=len(length)):
                raise FloatingPointError('Invalid entropy indexes')
            value=symbol-offset[index];maximum=length[index]-2
            chosen=np.where((value<0)|(value>=maximum),maximum,value)
            frequency=cdf[index,chosen+1]-cdf[index,chosen]
            raw=np.where(value<0,-2*value-1,np.where(value>=maximum,2*(value-maximum),0))
            # The native bypass loop operates on uint32 values with 4-bit shifts.
            # Reject 28-bit overflow before it can reach an undefined 32-bit shift.
            bad=np.where((frequency<=0)|(raw>=2**28))[0]
            if len(bad):
                diagnostic=dict(step=completed_steps,symbol_min=int(symbol.min()),symbol_max=int(symbol.max()),
                    minimum_frequency=int(frequency.min()),maximum_bypass=int(raw.max()),
                    invalid_symbols=len(bad),examples=[dict(symbol=int(symbol[j]),index=int(index[j]),
                        frequency=int(frequency[j]),bypass=int(raw[j])) for j in bad[:4]])
                write_json(out/(name+'-entropy-reject-'+str(completed_steps)+'.json'),diagnostic)
                raise FloatingPointError('Unsafe entropy candidate: zero frequency or bypass overflow')
            self.inner.encode_with_indexes(symbols,indexes,cdfs,lengths,offsets)
        def flush(self):return self.inner.flush()
    implementation.BufferedRansEncoder=CheckedEncoder
    stream_path=out/'streams/latent_tto'/(name+'.bin');image_path=out/'recon/latent_tto'/(name+'.png')
    stream_path.write_bytes(best_blob);Image.fromarray(best_image).save(image_path)
    original_quantizers=[]
    for module in [model.gaussian_conditional,model.entropy_bottleneck]:
        original=module.quantize;original_quantizers.append((module,original))
        def guarded(value,mode,means=None,original=original):
            if not torch.isfinite(value).all() or (means is not None and not torch.isfinite(means).all()):
                raise FloatingPointError('Nonfinite entropy quantizer input')
            result=original(value,mode,means)
            if mode=='symbols' and ((result==torch.iinfo(torch.int32).min).any() or (result==torch.iinfo(torch.int32).max).any()):
                raise FloatingPointError('Extreme int32 entropy symbol')
            return result
        module.quantize=guarded
    faulthandler.enable();faulthandler.dump_traceback_later(20,repeat=True)
    try:
        for step in range(1,a.search_steps+1):
            write_json(progress,dict(step=step,phase='forward'))
            model.update_resolutions(16,16);optimizer.zero_grad(set_to_none=True)
            value=model(x);rate=-sum(v.clamp_min(1e-9).log2().sum() for v in value['likelihoods'].values())/65536
            loss=I.objective(torch,lp,value['x_hat'].clamp(0,1),x,scale,a)+.5*(rate-budget*8/65536).relu()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite latent gradient')
            norm=torch.nn.utils.clip_grad_norm_([parameter],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite latent gradient norm')
            optimizer.step();completed_steps=step
            if not torch.isfinite(parameter).all():raise FloatingPointError('Nonfinite optimized latent')
            if step%5==0 or step==a.search_steps:
                with torch.no_grad():
                    write_json(progress,dict(step=step,phase='compress'))
                    model.update_resolutions(16,16);attempted+=1
                    try:encoded=model.compress(x)
                    except FloatingPointError as error:
                        rejected.append(dict(step=step,reason=str(error)));continue
                    candidate=base_pack(a.quality,True,(256,256),tuple(encoded['shape']),encoded['strings'])
                    if len(candidate)<=budget:
                        feasible+=1;write_json(progress,dict(step=step,phase='decompress'))
                        model.update_resolutions(16,16)
                        decoded=model.decompress(encoded['strings'],encoded['shape'])['x_hat'].clamp(0,1)
                        if not torch.isfinite(decoded).all():raise FloatingPointError('Nonfinite candidate decode')
                        score=float(I.objective(torch,lp,decoded,x,scale,a))
                        if score<best_score:
                            best_blob,best_image,best_score=candidate,I.uint8(decoded),score
                            stream_path.write_bytes(best_blob);Image.fromarray(best_image).save(image_path)
    except FloatingPointError as error:
        reason=str(error)
    finally:
        hook.remove();faulthandler.cancel_dump_traceback_later()
        implementation.BufferedRansEncoder=native_encoder
        for module,original in original_quantizers:module.quantize=original
    assert len(best_blob)<=budget
    (out/'streams/latent_tto'/(name+'.bin')).write_bytes(best_blob)
    Image.fromarray(best_image).save(out/'recon/latent_tto'/(name+'.png'))
    write_json(out/'worker-results'/(name+'.json'),dict(name=name,candidates_attempted=attempted,candidates_feasible=feasible,
        completed_steps=completed_steps,numerical_stop=reason,rejected_candidates=rejected,reused_v1=False))
    stop.set()


def complete_tto(out,source=None):
    source=out if source is None else source
    for folder in ['worker-results','streams/latent_tto','recon/latent_tto']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    test=[r for r in I.rows(out) if r['split']=='test']
    for i,row in enumerate(test):
        name=row['name'];stream=out/'streams/latent_tto'/(name+'.bin');image=out/'recon/latent_tto'/(name+'.png')
        if stream.exists() and image.exists() and (out/'worker-results'/(name+'.json')).exists():
            continue
        if stream.exists() and image.exists():
            write_json(out/'worker-results'/(name+'.json'),dict(name=name,candidates_attempted=None,candidates_feasible=None,
                completed_steps=I.cfg(out).search_steps,numerical_stop=None,reused_v1=True,
                note='Completed output preserved; candidate-count history not persisted by v1.'))
        else:
            write_json(out/'status.json',dict(state='running',stage='recover_tto',name=name,images=i,total=len(test)))
            try:
                with (out/('worker-'+name+'.log')).open('w') as log:
                    p=subprocess.run([sys.executable,__file__,'--out',str(out),'--name',name],stdout=log,stderr=subprocess.STDOUT,timeout=120)
                if p.returncode:raise RuntimeError('worker exit '+str(p.returncode))
            except (subprocess.TimeoutExpired,RuntimeError) as error:
                # Keep every source image. Original legal base is an explicit fallback.
                if not (stream.exists() and image.exists()):
                    shutil.copyfile(out/'cache/test'/(name+'.bin'),stream)
                    import torch
                    v=torch.load(out/'cache/test'/(name+'.pt'),weights_only=True)['base'][0].permute(1,2,0).numpy()
                    Image.fromarray(v).save(image)
                write_json(out/'worker-results'/(name+'.json'),dict(name=name,candidates_attempted=None,candidates_feasible=None,
                    completed_steps=None,numerical_stop=str(error),reused_v1=False,
                    fallback_original=stream.read_bytes()==(out/'cache/test'/(name+'.bin')).read_bytes(),
                    note='Preserve last valid candidate if worker wrote one; otherwise original stream.'))
    # Recompute all TTO metrics from preserved output files; do not invent v1 history.
    torch,stop=I.setup(out,'recovered-metrics');sys.path.insert(0,str(ROOT/'src'))
    from clic27.eval import metrics as M
    from projection_probe import edge_error
    metric=M.Perceptual('cuda');results=[r for r in json.loads((source/'test_metrics.json').read_text()) if r['method']!='latent_tto']
    assert {r['method'] for r in results}=={'base','conditional','fixed','pixel'}
    for row in test:
        name=row['name'];v=torch.load(out/'cache/test'/(name+'.pt'),weights_only=True)
        original=v['original'][0].permute(1,2,0).numpy()
        image=np.array(Image.open(out/'recon/latent_tto'/(name+'.png')).convert('RGB'))
        packet=(out/'streams/latent_tto'/(name+'.bin')).read_bytes();budget=row['bytes']+46
        assert len(packet)<=budget
        with torch.no_grad():values=metric.all(original,image)
        record=json.loads((out/'worker-results'/(name+'.json')).read_text())
        record.pop('name')
        results.append(dict(method='latent_tto',name=name,bytes=len(packet),base_bytes=row['bytes'],budget_bytes=budget,pixels=65536,
            packet_sha256=hashlib.sha256(packet).hexdigest(),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            edge=edge_error(original,image),**record,**values))
    write_json(out/'test_metrics.json',results);stop.set()


def recover(source,out):
    assert json.loads((source/'status.json').read_text())['state']=='failed'
    shutil.copytree(source,out)  # independent files; never mutate the failed run.
    (out/'worker-results').mkdir(exist_ok=True)
    write_json(out/'recovery_lineage.json',dict(source=str(source),source_status=json.loads((source/'status.json').read_text()),
        source_elapsed_until_stop_s=(source/'status.json').stat().st_mtime-(source/'config.json').stat().st_mtime,
        change='Numerical guards and per-image process/timeout; reuse completed streams. No model/data/objective/LR/step changes.'))
    frozen=json.loads((source/'sources.sha256.json').read_text())
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in frozen.items())
    frozen[Path(__file__).name]=sha(Path(__file__))
    write_json(out/'sources.sha256.json',frozen)
    started=time.monotonic()
    complete_tto(out,source)
    for stage in ['verify','summarize']:
        write_json(out/'status.json',dict(state='running',stage=stage))
        with (out/(stage+'-recovery.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).with_name('innovation_pilot.py')),'--stage',stage,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if p.returncode:
            write_json(out/'status.json',dict(state='failed',stage=stage,exit_code=p.returncode));raise RuntimeError(stage)
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in frozen.items())
    write_json(out/'status.json',dict(state='complete',recovered=True,recovery_elapsed_s=time.monotonic()-started,
        test_images=sum(r['split']=='test' for r in I.rows(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--name')
    a=p.parse_args()
    if a.name:worker(a.out,a.name)
    elif a.source:recover(a.source,a.out)
    else:p.error('--source or --name required')
