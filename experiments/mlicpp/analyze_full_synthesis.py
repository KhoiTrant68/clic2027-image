"""Audit full-synthesis development pilot, never promote validation to test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(root):
    status=read(root/'status.json');assert status['state']=='complete' and status['development_only']
    cfg=read(root/'config.json'); rows=read(root/'validation_metrics.json')
    methods=['base','mean','reconstruction','gan']
    names=sorted(r['name'] for r in rows if r['method']=='base')
    assert len(names)==len(set(names))==cfg['validation_limit']
    by={(r['method'],r['name']):r for r in rows}
    assert len(rows)==len(by)==len(names)*len(methods)
    for name in names:
        assert len({by[m,name]['bytes'] for m in methods})==1
    assert read(root/'fresh.json')['matches']==len(rows)
    for name,digest in read(root/'sources.sha256.json').items():
        assert hashlib.sha256((root/'source_archive'/name).read_bytes()).hexdigest()==digest,name
    ix=np.random.default_rng(cfg['seed']).integers(0,len(names),(10000,len(names)))
    comparisons=[]
    for treatment,control in [('gan','reconstruction'),('reconstruction','mean'),('gan','mean'),('reconstruction','base')]:
        for metric in ['lpips','dists','edge','mse']:
            delta=np.asarray([by[treatment,n][metric]-by[control,n][metric] for n in names])
            comparisons.append(dict(treatment=treatment,control=control,metric=metric,delta=float(delta.mean()),
                ci95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()))
    result=dict(development_only=True,no_test_accessed=True,bootstrap_replicates=10000,seed=cfg['seed'],
        note='Exploratory paired-by-image intervals on reused development data; not confirmatory, no human preference inference.',
        comparisons=comparisons,fresh_pixel_matches=len(rows),source_hashes_verified=True)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
