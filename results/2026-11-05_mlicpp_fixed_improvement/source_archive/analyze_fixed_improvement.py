"""Paired same-byte factorial comparisons; no torch required for analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main(root):
    c=read(root/'config.json');rows=read(root/'test_metrics.json');budgets=c['search_budgets']
    methods=['base']+[f'{key}_s{s}' for key in ['reference','long'] for s in budgets]
    names=sorted(r['name'] for r in rows if r['method']=='base');by={(r['method'],r['name']):r for r in rows}
    assert len(names)==len(set(names))==c['test_limit'] and len(by)==len(rows)==len(methods)*len(names)
    assert read(root/'fresh.json')['matches']==len(rows)
    assert read(root/'source_audit.json')['images']==len(names)
    for name,h in read(root/'sources.sha256.json').items():assert hashlib.sha256((root/'source_archive'/name).read_bytes()).hexdigest()==h
    for name in names:
        b=by['base',name]['bytes']
        assert all(by[m,name]['bytes']==b+46 for m in methods[1:])
        for key in ['reference','long']:
            assert by[f'{key}_s{budgets[-1]}',name]['selection_objective']<=by[f'{key}_s{budgets[0]}',name]['selection_objective']
    summary=[]
    for method in methods:
        group=[by[method,n] for n in names];mse=np.mean([r['mse'] for r in group])
        summary.append(dict(method=method,n=len(group),actual_bpp=8*sum(r['bytes'] for r in group)/(65536*len(group)),
            psnr_pooled=float(10*math.log10(255**2/mse)),
            **{k:float(np.mean([r[k] for r in group])) for k in ['lpips','dists','msssim','edge']},
            mean_search_seconds=float(np.mean([r.get('search_seconds',0) for r in group]))))
    small,large=budgets
    pairs=[(f'long_s{large}',f'reference_s{small}'),(f'long_s{small}',f'reference_s{small}'),
           (f'reference_s{large}',f'reference_s{small}'),(f'long_s{large}',f'long_s{small}'),
           (f'long_s{large}',f'reference_s{large}'),(f'reference_s{small}','base'),(f'long_s{large}','base')]
    ix=np.random.default_rng(c['seed']).integers(0,len(names),(10000,len(names)));comparisons=[]
    for treatment,control in pairs:
        for metric in ['lpips','dists','mse','edge']:
            delta=np.array([by[treatment,n][metric]-by[control,n][metric] for n in names])
            comparisons.append(dict(treatment=treatment,control=control,metric=metric,delta=float(delta.mean()),
                ci95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()))
    old={r['step']:r for r in map(json.loads,(root/'reference_fixed/train.jsonl').read_text().splitlines())}
    new={r['step']:r for r in map(json.loads,(root/'fixed/train.jsonl').read_text().splitlines())}
    common=sorted(set(old)&set(new));prefix_equal=all(old[s]['loss']==new[s]['loss'] for s in common)
    assert common and prefix_equal,'Training prefix differs: do not treat duration as sole factor'
    analysis=dict(test_sources=len(names),fresh_pixel_matches=len(rows),bootstrap_replicates=10000,
        prefix_log_checkpoints=common,reference_training_prefix_exact=prefix_equal,comparisons=comparisons,
        limitations='One seed/rate, 32 new crops. Encoder compute increases with search budget. Pointwise exploratory CIs; no novelty/human/CLIC eligibility claim.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (root/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from clic27.eval.metrics import write_csv
    write_csv(summary,root/'summary.csv')
    print(json.dumps(dict(summary=summary,primary=comparisons[:2]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
