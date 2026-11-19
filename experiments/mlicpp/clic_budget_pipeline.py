"""CLIC validation grid followed by conservative package allocation and blind prep."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from PIL import Image
from probe import sha,write_json


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        files=sorted(a.images.glob('*.png'));assert len(files)==30
        pixels=0;records=[]
        for path in files:
            with Image.open(path) as im:pixels+=im.width*im.height
            records.append(dict(name=path.name,split='validation',sha256=sha(path)))
        assert pixels==86245376
        data=a.out/'input';data.mkdir();(data/'images').symlink_to(a.images,target_is_directory=True)
        write_json(data/'manifest.json',dict(images=records,dataset='CLIC 2025 test / 2027 validation; reused benchmark'))
        cfg=json.loads(a.config.read_text());cfg.update(data=str(data),expected_images=30,required_free_mib=10240,eval_memory_mib=8192)
        config=a.out/'config.json';write_json(config,cfg)
        deadline=time.monotonic()+12*3600
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=10240:break
            if time.monotonic()>deadline:raise TimeoutError('12h resource wait exhausted')
            write_json(a.out/'status.json',dict(state='waiting_for_gpu',free_mib=free,required_free_mib=10240));time.sleep(30)
        with (a.out/'pipeline.log').open('a',buffering=1) as log:
            write_json(a.out/'status.json',dict(state='running',stage='full30_grid'))
            subprocess.run([sys.executable,str(Path(__file__).with_name('gpu_perceptual_full.py')),'--config',str(config),'--out',str(a.out/'grid')],stdout=log,stderr=subprocess.STDOUT,check=True)
            write_json(a.out/'status.json',dict(state='running',stage='budget_and_blind'))
            subprocess.run([sys.executable,str(Path(__file__).with_name('budget_blind.py')),'--source',str(a.out/'grid'),'--out',str(a.out/'budgets')],stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(a.out/'status.json',dict(state='complete'))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','images','out']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
