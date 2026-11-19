"""CPU diagnostic on development crops, never accesses the reserved test set."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from PIL import Image
from probe import sha,write_json


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());source=Path(cfg['data'])
    records=[r for r in json.loads((source/'manifest.json').read_text())['images'] if r['split']=='validation'][:8]
    assert len(records)==8
    data=a.out/'development-center256';(data/'images').mkdir(parents=True)
    manifest=[]
    for r in records:
        path=source/'images'/r['name'];assert sha(path)==r['sha256']
        with Image.open(path) as im:
            im=im.convert('RGB');w,h=im.size;assert min(w,h)>=256
            x=(w-256)//2;y=(h-256)//2
            im.crop((x,y,x+256,y+256)).save(data/'images'/r['name'])
        manifest.append(dict(name=r['name'],sha256=sha(data/'images'/r['name']),source_sha256=r['sha256'],crop=[x,y,256,256]))
    write_json(data/'manifest.json',dict(images=manifest,role='development only; 8 center crops, not full-image benchmark'))
    variants=[('original',None)]+[(f'{arm}-seed{seed}',a.campaign/f'{arm}-q1-seed{seed}'/'last.pt') for seed in [9041,9042] for arm in ['baseline','joint']]
    results=[]
    try:
        for name,checkpoint in variants:
            write_json(a.out/'status.json',dict(state='running',stage=name,device='cpu',completed=len(results)))
            args=[sys.executable,str(Path(__file__).with_name('end_to_end_eval.py')),'--config',str(a.config),
                '--data',str(data),'--out',str(a.out/name),'--quality','1','--device','cpu']
            if checkpoint:args+=['--checkpoint',str(checkpoint)]
            with (a.out/'pipeline.log').open('a',buffering=1) as log:
                subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run(args+['--fresh'],check=True,stdout=log,stderr=subprocess.STDOUT)
            rows=json.loads((a.out/name/'metrics.json').read_text())
            results.append(dict(name=name,images=len(rows),bpp=sum(r['bytes']*8 for r in rows)/sum(r['pixels'] for r in rows),
                **{k:sum(r[k] for r in rows)/len(rows) for k in ['psnr','lpips','dists','encode_s','decode_s']}))
            write_json(a.out/'comparison.json',results)
        write_json(a.out/'status.json',dict(state='complete',completed=len(results),caveat='Development crops; differing rates; no win claim'))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
