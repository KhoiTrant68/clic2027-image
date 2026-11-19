"""Fixed-grid, full-image development evaluation with matched-bitstream controls."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from probe import write_json,sha


def run(a):
    c=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.config,a.out/'config.json')
    archive=a.out/'source_archive';archive.mkdir()
    for name in ['gpu_perceptual_full.py','end_to_end_eval.py','end_to_end.py','geometry_codec.py','probe.py','stream.py']:
        shutil.copyfile(Path(__file__).parent/name,archive/name)
    started=time.monotonic();rows_out=[]
    try:
        free=int(subprocess.check_output(['nvidia-smi','-i',str(c['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        required=c.get('required_free_mib',32768)
        if free<required:raise RuntimeError(f'Need {required} MiB free on selected GPU, found {free}')
        source=Path(c['data']);data=a.out/'development32-full';(data/'images').mkdir(parents=True)
        records=[r for r in json.loads((source/'manifest.json').read_text())['images'] if r['split']=='validation']
        count=c.get('expected_images',32);assert len(records)==count
        for r in records:
            path=source/'images'/r['name'];assert sha(path)==r['sha256']
            shutil.copyfile(path,data/'images'/r['name'])
        write_json(data/'manifest.json',dict(images=records,role='reused development sources, full resolution'))
        write_json(a.out/'protocol.json',dict(factors=[.5,1,2,4],quality=2,images=count,
            synthesis_sha256=sha(Path(c['synthesis'])),gpu_index=c['gpu_index'],same_stream_required=True))
        for factor in [.5,1.,2.,4.]:
            for arm in ['original','perceptual']:
                if time.monotonic()-started>4*3600:raise TimeoutError('Four-hour budget before next configuration')
                free=int(subprocess.check_output(['nvidia-smi','-i',str(c['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<required:raise RuntimeError('GPU headroom reduced; stop before next configuration')
                folder=a.out/f'{arm}-factor{factor:g}'
                write_json(a.out/'status.json',dict(state='running',arm=arm,factor=factor,completed=len(rows_out),total=8))
                args=[sys.executable,str(Path(__file__).with_name('end_to_end_eval.py')),'--config',str(a.config),
                    '--data',str(data),'--out',str(folder),'--quality','2','--quantizer-factor',str(factor)]
                if arm=='perceptual':args+=['--synthesis',c['synthesis']]
                with (a.out/'pipeline.log').open('a',buffering=1) as log:
                    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run(args+['--fresh'],stdout=log,stderr=subprocess.STDOUT,check=True)
                rows=json.loads((folder/'metrics.json').read_text());assert len(rows)==count
                if arm=='perceptual':
                    base=json.loads((a.out/f'original-factor{factor:g}'/'metrics.json').read_text())
                    assert [(r['name'],r['stream_sha256']) for r in base]==[(r['name'],r['stream_sha256']) for r in rows]
                pixels=sum(r['pixels'] for r in rows)
                rows_out.append(dict(arm=arm,factor=factor,bpp=8*sum(r['bytes'] for r in rows)/pixels,
                    pooled_psnr=float(10*np.log10(255**2/(sum(r['mse']*r['pixels'] for r in rows)/pixels))),
                    **{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','lpips','dists','encode_s','decode_s']}))
                write_json(a.out/'comparison.json',rows_out)
        write_json(a.out/'status.json',dict(state='complete',configurations=8,images_per_configuration=count,
            fresh_decode_matches=8*count,same_stream_verified=True,seconds=time.monotonic()-started))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
