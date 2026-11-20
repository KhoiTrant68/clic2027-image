"""Compare full30 ELIC points with MLIC++ (2026-11-02 run) and the frozen PULSE mix, same allocator and Qhat."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
from bakeoff import allocate_one, qhat_score
from clic27.eval import metrics as M

MLIC = ROOT / 'results/2026-11-02_mlicpp/full30'
BASE = ROOT / 'results/2026-10-15_pulse_integration/integrated_bakeoff'
METRICS = ['msssim', 'lpips_alex', 'lpips_vgg', 'dists', 'clipiqa']


def bd(r1, q1, r2, q2):
    """Bjontegaard deltas of curve 2 vs curve 1: (rate %, quality). Cubic fit, overlap interval only."""
    lr1, lr2 = np.log(r1), np.log(r2)
    lo, hi = max(min(q1), min(q2)), min(max(q1), max(q2))
    rate = None
    if lo < hi:
        p1, p2 = np.polyint(np.polyfit(q1, lr1, 3)), np.polyint(np.polyfit(q2, lr2, 3))
        d = ((np.polyval(p2, hi) - np.polyval(p2, lo)) - (np.polyval(p1, hi) - np.polyval(p1, lo))) / (hi - lo)
        rate = (math.exp(d) - 1) * 100
    lo, hi = max(lr1.min(), lr2.min()), min(lr1.max(), lr2.max())
    qual = None
    if lo < hi:
        p1, p2 = np.polyint(np.polyfit(lr1, q1, 3)), np.polyint(np.polyfit(lr2, q2, 3))
        qual = ((np.polyval(p2, hi) - np.polyval(p2, lo)) - (np.polyval(p1, hi) - np.polyval(p1, lo))) / (hi - lo)
    return rate, qual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.out / 'status.json').read_text())
    if status.get('state') != 'complete' or not status.get('fresh_pixel_parity') or not status.get('metrics'):
        raise ValueError('Comparison requires completed metrics and fresh-process parity')
    manifest = json.loads((a.out / 'manifest.json').read_text())
    mlic_manifest = json.loads((MLIC / 'manifest.json').read_text())
    if {r['name']: r['sha256'] for r in manifest['images']} != {r['name']: r['sha256'] for r in mlic_manifest['images']}:
        raise ValueError('ELIC and MLIC++ runs used different images')
    names = sorted(r['name'] for r in manifest['images'])
    pixels = {r['name']: r['height'] * r['width'] for r in manifest['images']}
    total_px = sum(pixels.values())
    if len(names) != 30 or total_px != 86245376:
        raise ValueError('Budget comparison requires the established full30 set')
    qhat_path = ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json'
    qsha = hashlib.sha256(qhat_path.read_bytes()).hexdigest()
    if qsha != manifest['qhat_sha256'] or qsha != mlic_manifest['qhat_sha256']:
        raise ValueError('Qhat differs between runs')
    qhat = json.loads(qhat_path.read_text())

    def grid(rows, tag, qualities):
        if {(r['name'], int(r['quality'])) for r in rows} != {(n, q) for n in names for q in qualities} or len(rows) != 30 * len(qualities):
            raise ValueError(f'Incomplete/duplicate {tag} metric grid')
        return {(r['name'], f'{tag}/q{r["quality"]}'): r for r in rows}

    elic = grid(M.read_csv(a.out / 'metrics.csv'), 'elic', manifest['qualities'])
    mlic = grid(M.read_csv(MLIC / 'metrics.csv'), 'mlicpp', mlic_manifest['qualities'])
    baseline = [r for r in M.read_csv(BASE / 'allocation.csv') if r['cand'] == 'mix']
    baseline_metrics = {(r['cand'], r['key'], r['name']): r for r in M.read_csv(BASE / 'metrics.csv')}
    baseline_pool, pool_metrics = {n: [] for n in names}, {}
    for path in sorted((BASE / 'points').glob('*.csv')):
        for point in M.read_csv(path):
            key = point['cand'], point['key'], point['name']
            if point['name'] not in baseline_pool or key not in baseline_metrics:
                continue
            score = qhat_score(qhat, baseline_metrics[key])
            if not math.isfinite(score):
                continue
            tag = 'reference/' + point['cand'] + '/' + point['key']
            baseline_pool[point['name']].append((int(point['bytes']), score, tag))
            pool_metrics[point['name'], tag] = baseline_metrics[key]
    catalog = elic | mlic | pool_metrics
    results, selections = [], []

    def stats(method, selected, budget=None):
        met = [catalog[n, o[2]] for n, o in selected.items()]
        total = sum(o[0] for o in selected.values())
        if budget is not None and total > math.floor(budget * total_px / 8):
            raise ValueError('Allocation exceeds actual byte budget')
        r = dict(method=method, budget_bpp=budget, n=len(met), bytes=total, actual_bpp=8 * total / total_px,
                 psnr_pooled=M.pooled_psnr([float(m['mse']) for m in met], [pixels[n] for n in selected]),
                 qhat_v0=sum(o[1] for o in selected.values()) / len(selected),
                 elic_images=sum(o[2].startswith('elic/') for o in selected.values()),
                 mlicpp_images=sum(o[2].startswith('mlicpp/') for o in selected.values()))
        for key in METRICS:
            r[key] = sum(float(m[key]) for m in met) / len(met)
        return r

    def options(table, tag, qualities):
        return {n: [(int(table[n, f'{tag}/q{q}']['bytes']), float(table[n, f'{tag}/q{q}']['qhat_v0']), f'{tag}/q{q}')
                    for q in qualities] for n in names}

    curves = {}
    for table, tag, qualities in ((elic, 'elic', manifest['qualities']), (mlic, 'mlicpp', mlic_manifest['qualities'])):
        pts = []
        for q in qualities:
            row = stats(f'{tag}_q{q}', {n: options(table, tag, [q])[n][0] for n in names})
            results.append(row)
            pts.append(row)
        curves[tag] = pts
    elic_opt = options(elic, 'elic', manifest['qualities'])
    mlic_opt = options(mlic, 'mlicpp', mlic_manifest['qualities'])
    feasibility = []
    for rate in (.075, .15, .3):
        budget = math.floor(rate * total_px / 8)
        fixed = {}
        for row in (r for r in baseline if float(r['rate']) == rate):
            metric = baseline_metrics[row['point_cand'], row['key'], row['name']]
            if not math.isclose(qhat_score(qhat, metric), float(row['score']), abs_tol=1e-6):
                raise ValueError('Baseline scores use a different Qhat/protocol')
            catalog[row['name'], 'baseline'] = metric
            fixed[row['name']] = int(row['bytes']), float(row['score']), 'baseline'
        if set(fixed) != set(names):
            raise ValueError('Baseline does not cover the same 30 images')
        frozen = stats('frozen_pulse_mix', fixed, rate)
        results.append(frozen)
        feas = dict(budget_bpp=rate)
        for tag, opt in (('elic', elic_opt), ('mlicpp', mlic_opt)):
            sel = allocate_one(opt, budget)
            feas[tag] = sel is not None
            if sel is not None:
                results.append(stats(f'{tag}_allocated', sel, rate))
        feasibility.append(feas)
        reference = allocate_one(baseline_pool, budget)
        if reference is None or sum(v[1] for v in reference.values()) < sum(v[1] for v in fixed.values()):
            reference = fixed
        ref_stats = stats('reoptimized_reference_pool', reference, rate)
        results.append(ref_stats)
        for label, extra in (('mix_plus_elic', [elic_opt]), ('mix_plus_elic_mlicpp', [elic_opt, mlic_opt])):
            pool = {n: baseline_pool[n] + sum((o[n] for o in extra), []) for n in names}
            sel = allocate_one(pool, budget)
            if sel is None or sum(v[1] for v in sel.values()) < sum(v[1] for v in reference.values()):
                sel = reference
            r = stats(label, sel, rate)
            r['delta_qhat_vs_reoptimized'] = r['qhat_v0'] - ref_stats['qhat_v0']
            results.append(r)
            if label == 'mix_plus_elic':
                selections.extend(dict(budget_bpp=rate, name=n, bytes=v[0], score=v[1], selected=v[2]) for n, v in sel.items())
    bdr = {}
    for key, sign in (('psnr_pooled', 1), ('msssim', 1), ('lpips_alex', -1), ('dists', -1), ('qhat_v0', 1)):
        r1 = np.array([p['actual_bpp'] for p in curves['mlicpp']])
        r2 = np.array([p['actual_bpp'] for p in curves['elic']])
        q1 = sign * np.array([p[key] for p in curves['mlicpp']])
        q2 = sign * np.array([p[key] for p in curves['elic']])
        rate, qual = bd(r1, q1, r2, q2)
        bdr[key] = dict(bd_rate_pct_elic_vs_mlicpp=rate, bd_quality_elic_minus_mlicpp=None if qual is None else sign * qual)
    timing = {}
    for tag, base in (('elic', a.out), ('mlicpp', MLIC)):
        qs = manifest['qualities'] if tag == 'elic' else mlic_manifest['qualities']
        dec = {q: json.loads((base / f'q{q}-decode.json').read_text()) for q in qs}
        timing[tag] = {f'q{q}': dict(api_decode_s_30=sum(i['decode_s'] for i in d['images']),
                                     peak_reserved_mib=max(i['peak_reserved_mib'] for i in d['images']),
                                     gpu=d['gpu'], parameters=d['parameters']) for q, d in dec.items()}
    M.write_csv(results, a.out / 'comparison.csv')
    M.write_csv(selections, a.out / 'expanded_allocation.csv')
    report = dict(results=results, feasibility=feasibility, bd=bdr, timing=timing, qhat_sha256=qsha,
                  scope='same 30 images; MLIC++ numbers from the 2026-11-02 A30 run; baseline mix is historical')
    (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# ELIC so với MLIC++ trên 30 ảnh', '',
             '| Phương án | Trần bpp | bpp thực | PSNR gộp | MS-SSIM | LPIPS-Alex ↓ | DISTS ↓ | Q̂ ↑ | ảnh ELIC | ảnh MLIC++ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        cap = '-' if r['budget_bpp'] is None else str(r['budget_bpp'])
        lines.append(f"| {r['method']} | {cap} | {r['actual_bpp']:.5f} | {r['psnr_pooled']:.2f} | {r['msssim']:.4f} | "
                     f"{r['lpips_alex']:.4f} | {r['dists']:.4f} | {r['qhat_v0']:.4f} | {r['elic_images']} | {r['mlicpp_images']} |")
    lines += ['', '## BD (ELIC so với MLIC++, theo đường các checkpoint, khoảng chồng lấn)', '',
              '| Metric | BD-rate ELIC (%) | BD-quality ELIC − MLIC++ |', '| --- | ---: | ---: |']
    for k, v in bdr.items():
        fr = '-' if v['bd_rate_pct_elic_vs_mlicpp'] is None else f"{v['bd_rate_pct_elic_vs_mlicpp']:+.2f}"
        fq = '-' if v['bd_quality_elic_minus_mlicpp'] is None else f"{v['bd_quality_elic_minus_mlicpp']:+.4f}"
        lines.append(f'| {k} | {fr} | {fq} |')
    lines += ['', '## Độ phủ ngân sách', '']
    for f in feasibility:
        lines.append(f"- {f['budget_bpp']} bpp: ELIC {'khả thi' if f['elic'] else 'không đủ thấp'}; MLIC++ {'khả thi' if f['mlicpp'] else 'không đủ thấp'}.")
    (a.out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
