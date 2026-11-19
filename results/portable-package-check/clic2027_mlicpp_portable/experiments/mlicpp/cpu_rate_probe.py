"""Wait for CPU development diagnostics, then test low-rate coverage on the same crops."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from probe import write_json


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+7200
    try:
        while True:
            status=json.loads((a.development/'status.json').read_text())
            if status['state']=='complete':break
            if status['state']=='failed':raise RuntimeError('CPU development failed; inspect before proceeding')
            if time.monotonic()>deadline:raise TimeoutError('Development did not finish within 2h')
            write_json(a.out/'status.json',dict(state='waiting_for_cpu_development'))
            time.sleep(20)
        results=[]
        for factor in [2.,4.,8.]:
            folder=a.out/f'factor-{factor:g}'
            write_json(a.out/'status.json',dict(state='running',factor=factor))
            args=[sys.executable,str(Path(__file__).with_name('end_to_end_eval.py')),'--config',str(a.config),
                '--data',str(a.development/'development-center256'),'--out',str(folder),'--quality','1',
                '--device','cpu','--quantizer-factor',str(factor)]
            with (a.out/'pipeline.log').open('a',buffering=1) as log:
                subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run(args+['--fresh'],check=True,stdout=log,stderr=subprocess.STDOUT)
            rows=json.loads((folder/'metrics.json').read_text())
            results.append(dict(factor=factor,bpp=sum(r['bytes']*8 for r in rows)/sum(r['pixels'] for r in rows),
                **{k:sum(r[k] for r in rows)/len(rows) for k in ['psnr','lpips','dists']}))
            write_json(a.out/'comparison.json',results)
        write_json(a.out/'status.json',dict(state='complete',caveat='Rate coverage diagnostic on crops; not CLIC compliance'))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--development',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
