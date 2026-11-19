"""Same-bitstream perceptual decoder check, serialized after the CPU rate probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from probe import write_json


def run(a):
    a.out.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+10800
    try:
        while True:
            status=json.loads((a.rate/'status.json').read_text())
            if status['state']=='complete':break
            if status['state']=='failed':raise RuntimeError('Rate probe failed; inspect first')
            if time.monotonic()>deadline:raise TimeoutError('Previous CPU job did not finish in 3h')
            write_json(a.out/'status.json',dict(state='waiting_for_cpu_rate_probe'));time.sleep(20)
        results=[]
        for factor in [1.,2.]:
            for arm in ['original','perceptual']:
                folder=a.out/f'{arm}-factor{factor:g}'
                write_json(a.out/'status.json',dict(state='running',factor=factor,arm=arm))
                args=[sys.executable,str(Path(__file__).with_name('end_to_end_eval.py')),'--config',str(a.config),
                    '--data',str(a.development/'development-center256'),'--out',str(folder),'--quality','2',
                    '--device','cpu','--quantizer-factor',str(factor)]
                if arm=='perceptual':args+=['--synthesis',str(a.synthesis)]
                with (a.out/'pipeline.log').open('a',buffering=1) as log:
                    subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
                    subprocess.run(args+['--fresh'],check=True,stdout=log,stderr=subprocess.STDOUT)
                rows=json.loads((folder/'metrics.json').read_text())
                if arm=='perceptual':
                    reference=json.loads((a.out/f'original-factor{factor:g}'/'metrics.json').read_text())
                    assert [(r['name'],r['stream_sha256']) for r in reference]==[(r['name'],r['stream_sha256']) for r in rows]
                results.append(dict(arm=arm,factor=factor,bpp=sum(r['bytes']*8 for r in rows)/sum(r['pixels'] for r in rows),
                    **{k:sum(r[k] for r in rows)/len(rows) for k in ['psnr','lpips','dists']}))
                write_json(a.out/'comparison.json',results)
        write_json(a.out/'status.json',dict(state='complete',same_stream_verified=True,caveat='Reused development crops; not human preference'))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','development','rate','synthesis','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
