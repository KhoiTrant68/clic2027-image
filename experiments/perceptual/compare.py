"""Compare scored codec runs with the frozen PULSE mix at 0.075/0.15/0.3 bpp, same allocator and Qhat.

Usage: compare.py --out DIR --run tag=PATH [--run tag=PATH ...]
Each PATH holds metrics.csv from experiments/perceptual/score.py over the full30 images.
Per budget it reports: the codec alone (per-image allocation over its points), the reoptimised reference
pool, the pool plus each codec, and the pool plus every codec.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
from bakeoff import allocate_one, qhat_score
from clic27.eval import metrics as M

BASE = ROOT / 'results/2026-10-15_pulse_integration/integrated_bakeoff'
METRICS = ['msssim', 'lpips_alex', 'lpips_vgg', 'dists', 'clipiqa']
PIXELS = 86245376


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', action='append', required=True, help='tag=path')
    p.add_argument('--qhat', type=Path, default=ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json')
    a = p.parse_args()
    qhat = json.loads(a.qhat.read_text())
    baseline = [r for r in M.read_csv(BASE / 'allocation.csv') if r['cand'] == 'mix']
    names = sorted({r['name'] for r in baseline})
    base_metrics = {(r['cand'], r['key'], r['name']): r for r in M.read_csv(BASE / 'metrics.csv')}
    catalog, pool = {}, {n: [] for n in names}
    for path in sorted((BASE / 'points').glob('*.csv')):
        for pt in M.read_csv(path):
            key = pt['cand'], pt['key'], pt['name']
            if pt['name'] in pool and key in base_metrics:
                s = qhat_score(qhat, base_metrics[key])
                if math.isfinite(s):
                    tag = 'reference/' + pt['cand'] + '/' + pt['key']
                    pool[pt['name']].append((int(pt['bytes']), s, tag))
                    catalog[pt['name'], tag] = base_metrics[key]
    manifest = json.loads((ROOT / 'results/2026-11-02_mlicpp/full30/manifest.json').read_text())
    npx = {r['name']: r['width'] * r['height'] for r in manifest['images']}
    if set(npx) != set(names) or sum(npx.values()) != PIXELS:
        raise ValueError('full30 inventory differs from the baseline mix')
    runs = {}
    for spec in a.run:
        tag, path = spec.split('=', 1)
        rows = M.read_csv(Path(path) / 'metrics.csv')
        qs = sorted({r['quality'] for r in rows})
        if {(r['name'], r['quality']) for r in rows} != {(n, q) for n in names for q in qs} or len(rows) != 30 * len(qs):
            raise ValueError(f'{tag}: incomplete or duplicate full30 metric grid')
        if sum(int(r['pixels']) for r in rows if r['quality'] == qs[0]) != PIXELS:
            raise ValueError(f'{tag}: wrong pixel total')
        opts = {n: [] for n in names}
        for r in rows:
            t = f'{tag}/q{r["quality"]}'
            catalog[r['name'], t] = r
            opts[r['name']].append((int(r['bytes']), float(r['qhat_v0']), t))
        runs[tag] = (qs, opts)
    results = []

    def stats(method, sel, budget=None):
        met = [catalog[n, o[2]] for n, o in sel.items()]
        total = sum(o[0] for o in sel.values())
        if budget is not None and total > math.floor(budget * PIXELS / 8):
            raise ValueError('Allocation exceeds byte budget')
        px = {n: npx[n] for n in sel}
        r = dict(method=method, budget_bpp=budget, bytes=total, actual_bpp=8 * total / PIXELS,
                 qhat_v0=sum(o[1] for o in sel.values()) / len(sel),
                 psnr_pooled=M.pooled_psnr([float(m['mse']) for m in met], list(px.values())),
                 used={})
        for o in sel.values():
            fam = o[2].split('/')[0] if not o[2].startswith('reference/') else 'reference/' + o[2].split('/')[1]
            r['used'][fam] = r['used'].get(fam, 0) + 1
        for k in METRICS:
            r[k] = sum(float(m[k]) for m in met) / len(met)
        return r

    for tag, (qs, opts) in runs.items():
        for q in qs:
            results.append(stats(f'{tag}_q{q}', {n: next(o for o in opts[n] if o[2] == f'{tag}/q{q}') for n in names}))
    for rate in (.075, .15, .3):
        budget = math.floor(rate * PIXELS / 8)
        fixed = {}
        for row in (r for r in baseline if float(r['rate']) == rate):
            catalog[row['name'], 'baseline'] = base_metrics[row['point_cand'], row['key'], row['name']]
            fixed[row['name']] = int(row['bytes']), float(row['score']), 'baseline'
        results.append(stats('frozen_pulse_mix', fixed, rate))
        ref = allocate_one(pool, budget)
        if ref is None or sum(v[1] for v in ref.values()) < sum(v[1] for v in fixed.values()):
            ref = fixed
        ref_stats = stats('reoptimized_reference_pool', ref, rate)
        results.append(ref_stats)
        extras = list(runs.items()) + ([('ALL', (None, {n: sum((o[n] for _, o in runs.values()), []) for n in names}))] if len(runs) > 1 else [])
        for tag, (_, opts) in extras:
            if tag != 'ALL':
                alone = allocate_one(opts, budget)
                results.append(stats(f'{tag}_alone', alone, rate) if alone else dict(method=f'{tag}_alone', budget_bpp=rate, infeasible=True))
            sel = allocate_one({n: pool[n] + opts[n] for n in names}, budget)
            if sel is None or sum(v[1] for v in sel.values()) < sum(v[1] for v in ref.values()):
                sel = ref
            r = stats(f'mix_plus_{tag}', sel, rate)
            r['delta_qhat'] = r['qhat_v0'] - ref_stats['qhat_v0']
            results.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'comparison.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = ['| Phương án | Trần | bpp | Q̂ ↑ | ΔQ̂ | LPIPS-A ↓ | DISTS ↓ | PSNR | Ảnh dùng |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        if r.get('infeasible'):
            lines.append(f"| {r['method']} | {r['budget_bpp']} | không đủ thấp | | | | | | |")
            continue
        used = ', '.join(f'{k}:{v}' for k, v in sorted(r['used'].items()) if not k.startswith('reference/')) or '-'
        psnr = f"{r['psnr_pooled']:.2f}"
        lines.append(f"| {r['method']} | {r['budget_bpp'] or '-'} | {r['actual_bpp']:.4f} | {r['qhat_v0']:.4f} | "
                     f"{r.get('delta_qhat', 0):+.4f} | {r['lpips_alex']:.4f} | {r['dists']:.4f} | {psnr} | {used} |")
    (a.out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
