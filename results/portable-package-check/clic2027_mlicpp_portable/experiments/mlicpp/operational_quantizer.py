"""Development-only hard RD optimization; no gradients or likelihood proxy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from probe import settings, load, write_json, sha
from stream import pack, unpack

HEADER = struct.Struct('>4sBBf')
MODES = ['identity','scalar','slices']


def cfg(out): return SimpleNamespace(**json.loads((out/'config.json').read_text()))


def setup(out, stage):
    torch, install = settings(); stop = install(out/(stage+'.vram.jsonl'))
    a=cfg(out)
    m, meta=load(SimpleNamespace(upstream=Path(a.upstream),weights=Path(a.weights),legacy=True),2)
    m.requires_grad_(False)
    from geometry_codec import Quantizer
    q=Quantizer(m,None,None)
    return torch,stop,m,q,meta


def array(t): return t[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()


def encode(torch,m,q,x,steps,mode,factor):
    factor=float(np.float32(factor))
    delta=torch.tensor(steps,dtype=torch.float32,device='cuda').repeat_interleave(32)[None]
    # Reuse scaling implementation; delta is a shared model asset, not an oracle.
    q.select('oracle',factor,delta)
    encoded=m.compress(x)
    raw=pack(2,True,(256,256),tuple(encoded['shape']),encoded['strings'])
    blob=HEADER.pack(b'OPQ1',1,MODES.index(mode),factor)+raw
    image=array(m.decompress(encoded['strings'],encoded['shape'])['x_hat'])
    return blob,image


def inventory(out):
    a=cfg(out); ref=Path(a.reference_run)
    records=json.loads((ref/'manifest.json').read_text())['records']
    selected=[]
    for split,limit in [('train',a.train_sources),('validation',a.validation_sources)]:
        seen=set()
        for r in records:
            if r['split']==split and r['source'] not in seen and len(seen)<limit:
                selected.append(r); seen.add(r['source'])
        assert len(seen)==limit
    assert not {r['source'] for r in selected if r['split']=='train'} & {r['source'] for r in selected if r['split']=='validation'}
    write_json(out/'inventory.json',dict(records=selected,role='development only',test_cache_opened=False))
    return selected


def data(torch,out,rows):
    result=[]
    for r in rows:
        v=torch.load(Path(cfg(out).reference_run)/'cache'/(r['name']+'.pt'),weights_only=True)
        x=v['original'].cuda().float()/255
        result.append((r,x,v['original'][0].permute(1,2,0).numpy()))
    return result


def search(out):
    torch,stop,m,q,meta=setup(out,'search'); a=cfg(out); rows=inventory(out)
    train=data(torch,out,[r for r in rows if r['split']=='train'])
    calls=0; log=(out/'search.jsonl').open('w',buffering=1)
    def score(steps,mode):
        nonlocal calls
        values=[]
        with torch.no_grad():
            for r,x,target in train:
                blob,image=encode(torch,m,q,x,steps,mode,1.)
                bpp=len(blob)*8/65536; mse=float(((image.astype(np.float64)-target)**2).mean())
                values.append(dict(name=r['name'],bpp=bpp,mse=mse,j=bpp+a.lambda_mse*mse))
        calls+=1; loss=float(np.mean([r['j'] for r in values]))
        entry=dict(call=calls,mode=mode,steps=np.asarray(steps).tolist(),j=loss,images=values)
        log.write(json.dumps(entry)+'\n'); write_json(out/'search.progress.json',entry)
        print('candidate',calls,mode,loss,flush=True)
        return loss
    # Exact identity check against original upstream before any optimization.
    r,x,target=train[0]
    with torch.no_grad():
        q.close(); orig=m.compress(x); original=array(m.decompress(orig['strings'],orig['shape'])['x_hat'])
        from geometry_codec import Quantizer
        q=Quantizer(m,None,None)
        blob,image=encode(torch,m,q,x,np.ones(10),'identity',1.)
        assert blob[HEADER.size:]==pack(2,True,(256,256),tuple(orig['shape']),orig['strings'])
        assert np.array_equal(image,original)
    write_json(out/'smoke.json',dict(identity_payload_matches=True,identity_pixels_match=True))
    baseline=score(np.ones(10),'identity')
    scalar=np.ones(10); scalar_loss=baseline
    for value in np.geomspace(.7,1.5,a.scalar_candidates):
        proposal=np.full(10,value); loss=score(proposal,'scalar')
        if loss<scalar_loss: scalar,scalar_loss=proposal.copy(),loss
    theta=np.zeros(10); current=baseline
    for sweep in range(a.sweeps):
        for i in range(10):
            winner=theta.copy(); winloss=current
            for direction in [-1,1]:
                proposal=theta.copy(); proposal[i]=np.clip(proposal[i]+direction*a.log_step,np.log(.5),np.log(2))
                loss=score(np.exp(proposal),'slices')
                if loss<winloss: winner,winloss=proposal.copy(),loss
            theta,current=winner,winloss
        write_json(out/'model_asset.partial.json',dict(sweep=sweep+1,steps=np.exp(theta).tolist(),j=current))
    asset=dict(identity=np.ones(10).tolist(),scalar=scalar.tolist(),slices=np.exp(theta).tolist())
    write_json(out/'model_asset.json',asset)
    write_json(out/'search.summary.json',dict(identity_j=baseline,scalar_j=scalar_loss,slices_j=current,
        calls=calls,train_sources=len(train),parameters=dict(scalar=1,slices=10),backbone=meta))
    log.close(); q.close(); stop.set()


def evaluate(out):
    torch,stop,m,q,meta=setup(out,'evaluate'); a=cfg(out)
    asset=json.loads((out/'model_asset.json').read_text())
    selected=[r for r in json.loads((out/'inventory.json').read_text())['records'] if r['split']=='validation']
    validation=data(torch,out,selected); (out/'streams').mkdir(); results=[]
    with torch.no_grad():
        for r,x,target in validation:
            for mode in MODES:
                for factor in a.factors:
                    blob,image=encode(torch,m,q,x,asset[mode],mode,factor)
                    name=f'{r["name"]}-{mode}-{factor:g}.bin'; (out/'streams'/name).write_bytes(blob)
                    bpp=len(blob)*8/65536; mse=float(((image.astype(np.float64)-target)**2).mean())
                    results.append(dict(name=r['name'],mode=mode,factor=factor,stream=name,bpp=bpp,mse=mse,
                        psnr=float(10*np.log10(255**2/max(mse,1e-10))),j=bpp+a.lambda_mse*mse,
                        pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),stream_sha256=hashlib.sha256(blob).hexdigest()))
            write_json(out/'metrics.json',results); print('validation',r['name'],len(results),flush=True)
    q.close(); stop.set()


def decode(out):
    torch,stop,m,q,meta=setup(out,'decode')
    asset=json.loads((out/'model_asset.json').read_text()); rows=json.loads((out/'metrics.json').read_text())
    with torch.no_grad():
        for row in rows:
            blob=(out/'streams'/row['stream']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==row['stream_sha256']
            magic,version,mode,factor=HEADER.unpack_from(blob)
            assert magic==b'OPQ1' and version==1 and mode in range(3) and 0<factor<10
            quality,legacy,size,shape,strings=unpack(blob[HEADER.size:])
            assert quality==2 and legacy and size==(256,256)
            delta=torch.tensor(asset[MODES[mode]],dtype=torch.float32,device='cuda').repeat_interleave(32)[None]
            q.select('oracle',factor,delta)
            image=array(m.decompress(strings,shape)['x_hat'])
            assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256']
    write_json(out/'fresh_decode.json',dict(passed=len(rows),expected=len(rows),source_cache_read=False))
    q.close(); stop.set()


def analyze(out):
    rows=json.loads((out/'metrics.json').read_text()); a=cfg(out)
    means=[dict(mode=mode,factor=factor,**{k:float(np.mean([r[k] for r in rows if r['mode']==mode and r['factor']==factor]))
        for k in ['bpp','mse','psnr','j']}) for mode in MODES for factor in a.factors]
    values={mode:{r['name']:r['j'] for r in rows if r['mode']==mode and r['factor']==1.} for mode in MODES}
    names=sorted(values['slices']); delta=np.array([values['slices'][n]-values['scalar'][n] for n in names])
    rng=np.random.default_rng(a.seed); samples=rng.integers(0,len(names),(2000,len(names)))
    relative=float(delta.mean()/np.mean(list(values['scalar'].values()))*100)
    write_json(out/'analysis.json',dict(means=means,primary=dict(comparison='slices minus scalar at factor 1',
        delta_j=float(delta.mean()),relative_j_percent=relative,ci95=np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
        exploratory_gate_passed=relative<=-.5),caveat='development, small sample, no held-out confirmation'))


def run(args):
    out=args.out; out.mkdir(parents=True,exist_ok=False); shutil.copyfile(args.config,out/'config.json')
    snap=out/'source_archive';snap.mkdir()
    for name in ['operational_quantizer.py','geometry_codec.py','probe.py','stream.py','run_operational_quantizer.sh']:
        shutil.copyfile(Path(__file__).parent/name,snap/name)
    start=time.monotonic()
    try:
        for stage in ['search','evaluate','decode']:
            write_json(out/'status.json',dict(state='running',stage=stage))
            remaining=cfg(out).wall_time_limit_s-(time.monotonic()-start)
            if remaining<=0: raise TimeoutError('Wall-time budget exhausted')
            subprocess.run([sys.executable,__file__,'--stage',stage,'--out',str(out)],check=True,timeout=remaining)
        analyze(out); write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-start))
    except BaseException as e:
        write_json(out/'status.json',dict(state='failed',error=repr(e),elapsed_s=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--stage',choices=['run','search','evaluate','decode'],default='run');args=p.parse_args()
    if args.stage=='run':run(args)
    else:globals()[args.stage](args.out)
