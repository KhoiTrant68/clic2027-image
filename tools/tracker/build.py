"""Build the experiment tracker data (results/tracker/tracker.json) for the /results page.

Two parts:
  * results/experiments.json: hand-written registry (date, title, dataset, status, decision, reports);
  * every CLIC30 codec run that has per-image metrics with real byte counts. Q̂ v0 is recomputed from the
    six stored features with one formula, so all codecs are comparable. For each codec the builder lists its
    operating points, allocates bytes per image at 0.075 / 0.15 / 0.3 bpp (same knapsack as the bake-off),
    and measures the gain of adding the codec to the reference mix pool.
Run from the repo root:  python tools/tracker/build.py
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
from bakeoff import allocate_one, qhat_score  # noqa: E402

QHAT = ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json'
POOL = ROOT / 'results/2026-10-15_pulse_integration/integrated_bakeoff'
MANIFEST = ROOT / 'results/2026-11-02_mlicpp/full30/manifest.json'
BUDGETS = (0.075, 0.15, 0.3)
SCREEN = ('ebfd571f', 'bb7344a2', '86127fbd', '2a760bf1')  # bakeoff.LABELS: text/screen content
FEATURES = ['psnr', 'msssim', 'lpips_alex', 'lpips_vgg', 'dists', 'clipiqa']

# One entry per codec: where its per-image rows live, which experiment it belongs to, and its current status
# (chosen = in the reference mix pool, rejected, running).
#   kind 'bakeoff': points/<cand>.csv (bytes, t_dec) + metrics.csv keyed by (cand, key, name)
#   kind 'scores' : one metrics.csv with cand,key,name,bytes,t_dec + features
#   kind 'quality': one metrics.csv with quality,name,bytes,decode_s + features (experiments/perceptual layout)
CODECS = [
    dict(id='ditic', label='DiT-IC', family='Generative (diffusion)', kind='bakeoff', path=POOL, cand='ditic', experiment='ditic-crdr', status='chosen', in_pool=True),
    dict(id='ditics', label='DiT-IC chưng cất', family='Generative (diffusion)', kind='bakeoff', path=POOL, cand='ditics', experiment='ditic-speed', status='chosen', in_pool=True),
    dict(id='msillm', label='MS-ILLM', family='GAN', kind='bakeoff', path=POOL, cand='msillm', experiment='bakeoff-prelim', status='chosen', in_pool=True),
    dict(id='pulse', label='PULSE-L perceptual', family='GAN', kind='bakeoff', path=POOL, cand='pulse', experiment='pulse-integration', status='chosen', in_pool=True),
    dict(id='codlite', label='CoD-Lite', family='Generative (diffusion)', kind='bakeoff', path=POOL, cand='codlite', experiment='sd-turbo-refiner', status='chosen', in_pool=True),
    dict(id='vtmscc', label='VTM-SCC', family='Cổ điển', kind='bakeoff', path=POOL, cand='vtmscc', experiment='clic-b', status='chosen', in_pool=True),
    dict(id='vtm420', label='VTM 4:2:0', family='Cổ điển', kind='bakeoff', path=ROOT / 'results/2026-10-13_bakeoff_v0', cand='vtm420', experiment='bakeoff-l40', status='rejected'),
    dict(id='mbt', label='mbt2018-mean', family='MSE', kind='bakeoff', path=ROOT / 'results/2026-10-13_bakeoff_v0', cand='mbt', experiment='bakeoff-l40', status='rejected'),
    dict(id='crdr', label='CRDR', family='GAN', kind='bakeoff', path=ROOT / 'results/2026-10-10_bakeoff_v0', cand='crdr', experiment='ditic-crdr', status='rejected'),
    dict(id='turbo', label='MS-ILLM + SD-Turbo refiner', family='Generative (diffusion)', kind='bakeoff', path=ROOT / 'results/2026-10-08_bakeoff_merged', cand='turbo', experiment='sd-turbo-refiner', status='rejected'),
    dict(id='s1res', label='S1 + residual VTM', family='Codec tự xây', kind='bakeoff', path=ROOT / 'results/2026-10-08_bakeoff_merged', cand='s1res', experiment='s1-kodak', status='rejected',
         note='Hai ảnh f67f11de, ff32adfa chưa tính xong metric khi phiên 06/10 hết giờ; checkpoint S1 (train trên Kaggle) không còn nên không chạy bù được.'),
    dict(id='aeic_me', label='AEIC-ME', family='Generative (diffusion)', kind='scores', path=ROOT / 'results/2026-10-20_aeic_full30/full30_20261020/scores/metrics.csv', cand='aeic_me', experiment='aeic', status='rejected'),
    dict(id='aeic_se', label='AEIC-SE', family='Generative (diffusion)', kind='scores', path=ROOT / 'results/2026-10-20_aeic_full30/full30_20261020/scores/metrics.csv', cand='aeic_se', experiment='aeic', status='rejected'),
    dict(id='mlicpp', label='MLIC++ (MSE)', family='MSE', kind='quality', path=ROOT / 'results/2026-11-02_mlicpp/full30/metrics.csv', experiment='mlicpp-full30', status='rejected'),
    dict(id='elic', label='ELIC (MSE, tái lập)', family='MSE', kind='quality', path=ROOT / 'results/2026-11-19_elic/full30/metrics.csv', experiment='elic-full30', status='rejected'),
    dict(id='oscar', label='OSCAR', family='Generative (diffusion)', kind='quality', path=ROOT / 'results/2026-11-20_perceptual/oscar/metrics.csv', experiment='perceptual-scan-1120', status='rejected'),
    dict(id='osdiff', label='OSDiff', family='Generative (diffusion)', kind='quality', path=ROOT / 'results/2026-11-20_perceptual/osdiff/metrics.csv', experiment='perceptual-scan-1120', status='rejected'),
    dict(id='elic_gc', label='ELIC + decoder GAN (công thức GC)', family='GAN', kind='quality', path=ROOT / 'results/2026-11-20_gc_backbone/eval/elic/metrics.csv', experiment='gc-backbone', status='rejected'),
    dict(id='mlicpp_gc', label='MLIC++ + decoder GAN (công thức GC)', family='GAN', kind='quality', path=ROOT / 'results/2026-11-20_gc_backbone/eval/mlicpp/metrics.csv', experiment='gc-backbone', status='rejected'),
]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_rows(c):
    """Per-image rows {key, name, bytes, decode_s, features...} for one codec, or None if missing."""
    if c['kind'] == 'bakeoff':
        points, metrics = c['path'] / 'points' / f"{c['cand']}.csv", c['path'] / 'metrics.csv'
        if not points.exists():
            return None
        met = {(r['key'], r['name']): r for r in read_csv(metrics) if r['cand'] == c['cand']}
        out = []
        for p in read_csv(points):
            m = met.get((p['key'], p['name']))
            if m:
                out.append(dict(m, key=p['key'], name=p['name'], bytes=int(p['bytes']), decode_s=fnum(p.get('t_dec'))))
        return out
    if not c['path'].exists():
        return None
    rows = read_csv(c['path'])
    if c['kind'] == 'scores':
        return [dict(r, key=r['key'], bytes=int(r['bytes']), decode_s=fnum(r.get('t_dec'))) for r in rows if r['cand'] == c['cand']]
    return [dict(r, key=f"q{r['quality']}", bytes=int(r['bytes']), decode_s=fnum(r.get('decode_s'))) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/tracker/tracker.json')
    a = ap.parse_args()
    qhat = json.loads(QHAT.read_text())
    registry = json.loads((ROOT / 'results/experiments.json').read_text(encoding='utf-8'))
    images = {r['name']: r for r in json.loads(MANIFEST.read_text())['images']}
    names = sorted(images)
    pixels = {n: images[n]['width'] * images[n]['height'] for n in names}
    total_px = sum(pixels.values())
    assert len(names) == 30 and total_px == 86245376

    codecs, per_image, options, warnings, partial = [], {}, {}, [], {}
    for c in CODECS:
        rows = load_rows(c)
        if rows is None:
            warnings.append(f"{c['label']}: chưa có số liệu ({c['path'].relative_to(ROOT)})")
            continue
        by_key = {}
        for r in rows:
            if r['name'] not in pixels:
                continue
            feats = {k: fnum(r.get(k)) for k in FEATURES}
            if any(v is None for v in feats.values()):
                continue
            q = qhat_score(qhat, feats)
            by_key.setdefault(r['key'], {})[r['name']] = dict(
                name=r['name'], bytes=r['bytes'], bpp=8 * r['bytes'] / pixels[r['name']], qhat=q,
                mse=fnum(r.get('mse')), decode_s=r['decode_s'], **feats)
        points, opts = [], {n: [] for n in names}
        for key, imgs in sorted(by_key.items(), key=lambda kv: sum(v['bytes'] for v in kv[1].values())):
            complete = set(imgs) == set(names)
            if not complete:
                partial.setdefault(c['label'], []).append(f"{key} ({len(imgs)}/30)")
                if len(imgs) < 24:  # too few images to say anything about the set
                    continue
            vals = list(imgs.values())
            n_img, px = len(vals), sum(pixels[n] for n in imgs)
            mses = [v['mse'] for v in vals]
            pid = f"{c['id']}/{key}"
            total = sum(v['bytes'] for v in vals)
            dec = [v['decode_s'] for v in vals if v['decode_s'] is not None]
            points.append(dict(
                id=pid, key=key, bpp=8 * total / px, bytes=total, images=n_img, partial=not complete,
                qhat=sum(v['qhat'] for v in vals) / n_img,
                lpips=sum(v['lpips_alex'] for v in vals) / n_img, dists=sum(v['dists'] for v in vals) / n_img,
                msssim=sum(v['msssim'] for v in vals) / n_img, clipiqa=sum(v['clipiqa'] for v in vals) / n_img,
                # mse is on the 0-255 scale (clic27.eval.metrics.mse); pool it over pixels, then convert.
                psnr=(10 * math.log10(255 ** 2 / (sum(m * pixels[n] for m, n in zip(mses, imgs)) / px))
                      if all(m is not None for m in mses) else sum(v['psnr'] for v in vals) / n_img),
                decode_s=sum(dec) / len(dec) if len(dec) == n_img else None))
            per_image[pid] = [dict(name=v['name'], bpp=round(v['bpp'], 5), qhat=round(v['qhat'], 4), lpips=round(v['lpips_alex'], 4),
                                   dists=round(v['dists'], 4), psnr=round(v['psnr'], 2)) for v in vals]
            if complete:  # only full-set points can take part in a whole-set byte budget
                for n, v in imgs.items():
                    opts[n].append((v['bytes'], v['qhat'], pid))
        if not points:
            continue
        full = all(not p['partial'] for p in points)
        if full:
            options[c['id']] = opts
        codecs.append(dict(id=c['id'], label=c['label'], family=c['family'], experiment=c['experiment'], status=c['status'],
                           in_pool=bool(c.get('in_pool')), source=c['path'].relative_to(ROOT).as_posix(), points=points,
                           partial=not full, note=c.get('note')))

    def summarize(sel):
        imgs = {pid: {r['name']: r for r in per_image[pid]} for pid in {o[2] for o in sel.values()}}
        rows = {n: imgs[o[2]][n] for n, o in sel.items()}
        used = {}
        for o in sel.values():
            cid = o[2].split('/')[0]
            used[cid] = used.get(cid, 0) + 1
        # Pooled PSNR over the whole set, from each image's PSNR and pixel count.
        mse = sum(10 ** (-r['psnr'] / 10) * pixels[n] for n, r in rows.items()) / total_px
        return dict(bpp=8 * sum(o[0] for o in sel.values()) / total_px, qhat=sum(o[1] for o in sel.values()) / 30,
                    lpips=sum(r['lpips'] for r in rows.values()) / 30, dists=sum(r['dists'] for r in rows.values()) / 30,
                    psnr=-10 * math.log10(mse), used=used, choice={n: o[2] for n, o in sel.items()})

    for label, keys in partial.items():
        note = next((c['note'] for c in CODECS if c['label'] == label and c.get('note')), '')
        warnings.append(f"{label}: {len(keys)} điểm chỉ đo trên {keys[0].split(' ')[1].strip('()')} ảnh; hiện trên biểu đồ "
                        f"bằng điểm rỗng, không tham gia phân bổ ngân sách. {note}".strip())
    pool_ids = [c['id'] for c in codecs if c['in_pool']]
    budgets = []
    for rate in BUDGETS:
        cap = math.floor(rate * total_px / 8)
        pool = {n: sum((options[i][n] for i in pool_ids), []) for n in names}
        ref = allocate_one(pool, cap)
        entry = dict(budget=rate, mix=summarize(ref) if ref else None, codecs={})
        for c in codecs:
            if c['id'] not in options:  # measured on a subset of the images: shown on the chart only
                entry['codecs'][c['id']] = dict(alone=None, partial=True)
                continue
            alone = allocate_one(options[c['id']], cap)
            res = dict(alone=summarize(alone) if alone else None)
            if not c['in_pool'] and ref:
                plus = allocate_one({n: pool[n] + options[c['id']][n] for n in names}, cap)
                if plus and sum(o[1] for o in plus.values()) >= sum(o[1] for o in ref.values()):
                    s = summarize(plus)
                    res['with_mix'] = dict(delta_qhat=s['qhat'] - entry['mix']['qhat'], images=s['used'].get(c['id'], 0))
                else:
                    res['with_mix'] = dict(delta_qhat=0.0, images=0)
            entry['codecs'][c['id']] = res
        budgets.append(entry)

    out = dict(version=2, generated_from=[QHAT.relative_to(ROOT).as_posix(), 'results/experiments.json'],
               qhat='Q̂ v0 tính lại từ psnr, msssim, lpips_alex, lpips_vgg, dists, clipiqa của từng ảnh',
               dataset=dict(name='CLIC30', images=[dict(name=n, short=n[:8], width=images[n]['width'], height=images[n]['height'],
                                                        kind='screen' if n.startswith(SCREEN) else 'natural') for n in names],
                            pixels=total_px),
               registry=registry, codecs=codecs, budgets=budgets, per_image=per_image, warnings=warnings)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    mix = {b['budget']: round(b['mix']['qhat'], 4) for b in budgets}
    print(f"{len(codecs)} codecs, {sum(len(c['points']) for c in codecs)} points, mix Q̂ {mix}, {len(warnings)} warnings")
    for w in warnings:
        print('  -', w)


if __name__ == '__main__':
    main()
