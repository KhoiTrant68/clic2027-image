"""Bounded resource-aware sequential campaign; never stops unrelated GPU jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from probe import write_json,ROOT


def run(a):
    if a.resume:
        assert json.loads((a.out/'config.json').read_text())==json.loads(a.config.read_text()), 'Resume config differs'
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(a.config,a.out/'config.json')
    source=a.out/('source_resume_'+str(time.time_ns()) if a.resume else 'source_archive');source.mkdir()
    for name in ['end_to_end_campaign.py','end_to_end.py','end_to_end_eval.py','end_to_end_compare.py','end_to_end.json','geometry_codec.py','probe.py','stream.py']:
        shutil.copyfile(Path(__file__).parent/name,source/name)
    deadline=time.monotonic()+a.campaign_hours*3600
    cfg=json.loads(a.config.read_text());data=Path(cfg.get('test_data','/home/jovyan/clic2027-artifacts/data/div2k/end-to-end-test32'))
    def call(script,args,timeout=14400):
        with (a.out/'pipeline.log').open('a',buffering=1) as log:
            if time.monotonic()>=deadline:raise TimeoutError('Campaign budget exhausted before next stage')
            subprocess.run([sys.executable,str(Path(__file__).parent/script)]+list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,check=True)
    def available(stage):
        wait_until=min(deadline,time.monotonic()+a.gpu_wait_hours*3600)
        while time.monotonic()<wait_until:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(cfg.get('gpu_index',2)),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=cfg['required_free_mib']:return
            write_json(a.out/'status.json',dict(state='waiting_for_gpu',next_stage=stage,free_mib=free,
                required_free_mib=cfg['required_free_mib'],wait_remaining_s=round(wait_until-time.monotonic())))
            time.sleep(30)
        raise TimeoutError('GPU did not become available within configured wait / campaign budget')
    try:
        write_json(a.out/'status.json',dict(state='running',stage='prepare_new_test_sources'))
        if not (data/'manifest.json').exists():
            call('fetch_pilot_data.py',['--out',data,'--holdout-only','--holdout-start',370,'--holdout-count',32],3600)
        new=json.loads((data/'manifest.json').read_text())
        assert new['holdout_start']==370 and new['holdout_count']==32
        old=json.loads((Path(cfg['data'])/'manifest.json').read_text())
        assert not {r['name'] for r in new['images']} & {r['name'] for r in old['images']}
        write_json(a.out/'test_split.json',dict(manifest=str(data/'manifest.json'),names=[r['name'] for r in new['images']],
            role='not used for optimization; prior backbone pretraining overlap unknown'))
        evaluations=[]
        for quality in [1,2,3]:
            for seed in [9041,9042]:
                for arm in ['baseline','joint']:
                    name=f'{arm}-q{quality}-seed{seed}';folder=a.out/name
                    if a.resume and folder.exists():
                        status=json.loads((folder/'status.json').read_text())
                        assert status['state']=='complete' and (folder/'last.pt').is_file(), 'Refuse to overwrite incomplete run'
                        continue
                    available('train-'+name)
                    write_json(a.out/'status.json',dict(state='running',stage='train',run=name))
                    call('end_to_end.py',['--config',a.config,'--out',folder,'--stage','train','--quality',quality,'--seed',seed,'--arm',arm])
                    # No test evaluation influences any subsequent training schedule.
        for quality in [1,2,3]:
            variants=[('original',None)]+[(f'{arm}-seed{seed}',a.out/f'{arm}-q{quality}-seed{seed}'/'last.pt')
                for seed in [9041,9042] for arm in ['baseline','joint']]
            for name,checkpoint in variants:
                available(f'eval-{name}-q{quality}');out=a.out/f'eval-{name}-q{quality}'
                args=['--config',a.config,'--data',data,'--out',out,'--quality',quality]
                if checkpoint:args+=['--checkpoint',checkpoint]
                write_json(a.out/'status.json',dict(state='running',stage='evaluate',run=out.name))
                call('end_to_end_eval.py',args);call('end_to_end_eval.py',args+['--fresh'])
                rows=json.loads((out/'metrics.json').read_text())
                evaluations.append(dict(name=name,quality=quality,recon=str(out/'recon'),
                    bpp=8*sum(r['bytes'] for r in rows)/sum(r['pixels'] for r in rows),
                    pooled_psnr=float(10*np.log10(255**2/(sum(r['mse']*r['pixels'] for r in rows)/sum(r['pixels'] for r in rows)))),
                    **{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','lpips','dists','encode_s','decode_s']}))
                write_json(a.out/'comparison.json',evaluations)
        call('end_to_end_compare.py',['--out',a.out],300)
        # Predeclared approximate-rate blind preview for seed 9041 only, never pick by quality.
        base=next(r for r in evaluations if r['name']=='baseline-seed9041' and r['quality']==2)
        joint=min((r for r in evaluations if r['name']=='joint-seed9041'),key=lambda r:abs(r['bpp']-base['bpp']))
        mismatch=abs(joint['bpp']/base['bpp']-1)
        write_json(a.out/'blind_rate_check.json',dict(relative_mismatch=mismatch,threshold=.05,accepted=mismatch<=.05))
        if mismatch<=.05:
            config=a.out/'blind_config.json';write_json(config,dict(reference=str(data/'images'),
                codecs={'baseline':base['recon'],'joint':joint['recon']}))
            sys.path.insert(0,str(ROOT/'src'));from clic27.blind_score import prepare
            prepare(config,a.out/'blind_study')
        write_json(a.out/'status.json',dict(state='complete',blind_study_prepared=mismatch<=.05,
            caveat='Three measured rate points; inspect overlapping RD range. Human ratings not yet collected.'))
    except BaseException as exc:
        write_json(a.out/'status.json',dict(state='stopped',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--gpu-wait-hours',type=float,default=6)
    p.add_argument('--campaign-hours',type=float,default=24)
    run(p.parse_args())
