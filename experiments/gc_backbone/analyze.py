"""GC backbone ablation summary: ELIC vs MLIC++, MSE decoder vs GC-recipe GAN decoder, same streams.

    python analyze.py --run elic=... --run elic_gc=... --run mlicpp=... --run mlicpp_gc=... --out DIR

Each run dir holds metrics.csv (experiments/perceptual/score.py or the runner's own score stage, same columns).
Writes DIR/summary.json and DIR/summary.md: per-point pooled bpp / mean Q-hat / LPIPS / DISTS / pooled PSNR, the
gain of the GAN decoder on identical streams, and Bjontegaard deltas MLIC++ vs ELIC for both decoders.
Budget allocation (alone / with the frozen mix) is done by experiments/perceptual/compare.py.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/elic'))
from compare import bd  # noqa: E402  (same cubic Bjontegaard helper as the ELIC/MLIC++ report)

KEYS = ['qhat_v0', 'lpips_alex', 'dists', 'msssim']


def points(spec):
    """'path' or 'path:2,3,4,5' (keep only these qualities)."""
    path, _, keep = spec.rpartition(':') if ':' in spec.split('/')[-1] else (spec, '', '')
    keep = {k for k in keep.split(',') if k}
    rows = [r for r in csv.DictReader((Path(path) / 'metrics.csv').open()) if not keep or r['quality'] in keep]
    out = {}
    for q in sorted({r['quality'] for r in rows}, key=int):
        rs = [r for r in rows if r['quality'] == q]
        px = sum(int(r['pixels']) for r in rs)
        p = dict(quality=int(q), images=len(rs), bpp=8 * sum(int(r['bytes']) for r in rs) / px,
                 psnr=10 * math.log10(255 ** 2 / (sum(float(r['mse']) * int(r['pixels']) for r in rs) / px)))
        p.update({k: float(np.mean([float(r[k]) for r in rs])) for k in KEYS})
        out[int(q)] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='tag=path[:q,q,...] (tags: elic, elic_gc, mlicpp, mlicpp_gc)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    runs = {t: points(p) for t, p in (s.split('=', 1) for s in a.run)}
    for t, pts in runs.items():
        bad = [q for q, p in pts.items() if p['images'] != 30]
        if bad:
            raise SystemExit(f'{t}: incomplete qualities {bad}')
    a.out.mkdir(parents=True, exist_ok=True)
    res = dict(points=runs, gan_gain={}, bd={})
    lines = ['# GC backbone ablation (CLIC30, Q-hat v0)', '', '## Points', '',
             '| Run | q | bpp | Q-hat | LPIPS | DISTS | MS-SSIM | PSNR |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t, pts in runs.items():
        for q, p in pts.items():
            lines.append(f"| {t} | {q} | {p['bpp']:.4f} | {p['qhat_v0']:.3f} | {p['lpips_alex']:.4f} | {p['dists']:.4f} "
                         f"| {p['msssim']:.4f} | {p['psnr']:.2f} |")
    lines += ['', '## GAN decoder gain on identical streams (GC recipe minus MSE decoder)', '',
              '| Backbone | q | bpp | ΔQ-hat | ΔLPIPS | ΔDISTS | ΔPSNR |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('elic', 'mlicpp'):
        if arm in runs and arm + '_gc' in runs:
            for q, g in runs[arm + '_gc'].items():
                b = runs[arm].get(q)
                if b is None:
                    continue
                if abs(b['bpp'] - g['bpp']) > 1e-9:
                    raise SystemExit(f'{arm} q{q}: streams differ between decoders')
                d = {k: g[k] - b[k] for k in KEYS + ['psnr']}
                res['gan_gain'][f'{arm}/q{q}'] = d
                lines.append(f"| {arm} | {q} | {g['bpp']:.4f} | {d['qhat_v0']:+.3f} | {d['lpips_alex']:+.4f} "
                             f"| {d['dists']:+.4f} | {d['psnr']:+.2f} |")
    lines += ['', '## MLIC++ vs ELIC (Bjontegaard, cubic, overlap only; negative rate = MLIC++ saves bits)', '',
              '| Decoder | Metric | BD-rate MLIC++ (%) | BD-quality MLIC++ − ELIC |', '|---|---|---:|---:|']
    for dec, (e, m) in {'MSE': ('elic', 'mlicpp'), 'GC GAN': ('elic_gc', 'mlicpp_gc')}.items():
        if e not in runs or m not in runs:
            continue
        for k, sign in (('qhat_v0', 1), ('lpips_alex', -1), ('dists', -1), ('psnr', 1)):
            pe, pm = list(runs[e].values()), list(runs[m].values())
            r1, q1 = np.array([p['bpp'] for p in pe]), np.array([sign * p[k] for p in pe])
            r2, q2 = np.array([p['bpp'] for p in pm]), np.array([sign * p[k] for p in pm])
            rate, qual = bd(r1, q1, r2, q2)
            qual = None if qual is None else sign * qual
            res['bd'][f'{dec}/{k}'] = dict(rate=rate, quality=qual)
            fr = '—' if rate is None else f'{rate:+.2f}'
            fq = '—' if qual is None else f'{qual:+.4f}'
            lines.append(f'| {dec} | {k} | {fr} | {fq} |')
    (a.out / 'summary.json').write_text(json.dumps(res, indent=1))
    (a.out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
