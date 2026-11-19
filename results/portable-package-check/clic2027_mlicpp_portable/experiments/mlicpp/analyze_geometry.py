"""Predeclared descriptive RD and paired bootstrap analysis; no extrapolation."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(out):
    rows = json.loads((out/'metrics.json').read_text())
    factors = sorted({r['factor'] for r in rows})
    names = sorted({r['name'] for r in rows})
    modes = ['uniform', 'global', 'predicted', 'oracle']
    assert len(rows) == len(names)*len(factors)*len(modes)
    lookup = {(r['name'], r['mode'], r['factor']): r for r in rows}
    assert len(lookup) == len(rows)
    means = []
    for mode in modes:
        for factor in factors:
            subset = [lookup[n, mode, factor] for n in names]
            row = dict(mode=mode, factor=factor, images=len(names))
            for metric in ['bpp','payload_only_bpp','mse','psnr','lpips','dists']:
                row[metric] = float(np.mean([r[metric] for r in subset]))
            row['pooled_psnr'] = float(10*np.log10(255**2/row['mse']))
            means.append(row)
    contrasts = []
    rng = np.random.default_rng(9041)
    bootstrap = rng.integers(0, len(names), (2000, len(names)))
    for method, reference in [('global','uniform'),('predicted','uniform'),('predicted','global'),('oracle','uniform')]:
        for metric in ['bpp','mse','psnr','lpips','dists']:
            delta = np.array([lookup[n,method,1.0][metric]-lookup[n,reference,1.0][metric] for n in names])
            contrasts.append(dict(method=method, reference=reference, metric=metric, factor=1.,
                mean_delta=float(delta.mean()), ci95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist()))
    rd = []
    for method, reference in [('global','uniform'),('predicted','uniform'),('predicted','global'),('oracle','uniform')]:
        for rate_key in (['bpp','payload_only_bpp'] if method=='oracle' else ['bpp']):
            c = sorted([r for r in means if r['mode']==method], key=lambda r:r[rate_key])
            b = sorted([r for r in means if r['mode']==reference], key=lambda r:r[rate_key])
            lower = max(c[0][rate_key], b[0][rate_key]); upper = min(c[-1][rate_key], b[-1][rate_key])
            if lower >= upper:
                rd.append(dict(method=method, reference=reference, rate=rate_key, status='no_common_rate_interval')); continue
            grid = np.linspace(lower, upper, 21)
            cm = np.interp(grid, [r[rate_key] for r in c], [r['mse'] for r in c])
            bm = np.interp(grid, [r[rate_key] for r in b], [r['mse'] for r in b])
            rd.append(dict(method=method, reference=reference, rate=rate_key, interval=[lower,upper],
                mean_relative_mse_percent=float(np.mean(cm/bm-1)*100),
                min_relative_mse_percent=float(np.min(cm/bm-1)*100),
                max_relative_mse_percent=float(np.max(cm/bm-1)*100),
                caveat='descriptive linear interpolation of average RD curves; no matched-image rate claim or BD-rate'))
    result = dict(means=means, contrasts_same_scale_not_same_rate=contrasts, rd_interpolation=rd,
                  fresh_decode=json.loads((out/'fresh_decode.json').read_text()),
                  train=json.loads((out/'train.summary.json').read_text()))
    (out/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(means=means, rd_interpolation=rd), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('out',type=Path); analyze(p.parse_args().out)
