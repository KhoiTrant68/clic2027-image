"""Descriptive interpolation on overlapping measured rates; no extrapolation."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(rows):
    results=[]
    for seed in [9041,9042]:
        curves={name:sorted((r for r in rows if r['name']==name),key=lambda r:r['bpp'])
                for name in ['original',f'baseline-seed{seed}',f'joint-seed{seed}']}
        if any(len(v)!=3 for v in curves.values()):raise ValueError('Need three rates per curve')
        rates={k:np.array([r['bpp'] for r in v]) for k,v in curves.items()}
        if any(np.any(np.diff(v)<=0) for v in rates.values()):raise ValueError('Duplicate/nonpositive rate interval')
        lo=max(v[0] for v in rates.values());hi=min(v[-1] for v in rates.values())
        record=dict(seed=seed,overlap=[float(lo),float(hi)],has_overlap=bool(hi>lo))
        if hi>lo:
            grid=np.geomspace(lo,hi,5);record['bpp']=grid.tolist();record['curves']={}
            for name,curve in curves.items():
                record['curves'][name]={metric:np.interp(np.log(grid),np.log(rates[name]),[r[metric] for r in curve]).tolist()
                    for metric in ['pooled_psnr','psnr','lpips','dists']}
            record['joint_minus_baseline']={metric:(np.array(record['curves'][f'joint-seed{seed}'][metric])-
                record['curves'][f'baseline-seed{seed}'][metric]).tolist() for metric in ['pooled_psnr','psnr','lpips','dists']}
        results.append(record)
    return dict(method='Piecewise linear in log(bpp), common range only; descriptive, not significance testing',seeds=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(json.loads((a.out/'comparison.json').read_text()))
    (a.out/'matched_rate.json').write_text(json.dumps(result,indent=2))
