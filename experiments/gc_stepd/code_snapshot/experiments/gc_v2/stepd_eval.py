"""Step D evaluation (plans/GC_v2_architecture_2026-11-23.md section 5): real streams of a GC checkpoint on the 30
CLIC validation images at every rate level, Qhat v0 + LPIPS, and BD-rate against a reference run.

    python stepd_eval.py encode --ckpt RUN/rd/model.pt --out EVAL/D1 [--beta 0]
    python ../perceptual/score.py --run EVAL/D1 --data VALID --qualities 0 1 2 3 4 5 6 7     # Qhat v0 + LPIPS
    python stepd_eval.py bd --ref EVAL/D0 --test EVAL/D1 [EVAL/D2 ...]

`encode` writes the layout experiments/perceptual/score.py reads: q<level>-decode.json and q<level>/recon/*.png,
decoded from the saved stream (same process; the stream is the real one). Quality map all zeros, one beta.
`bd` reports BD-rate (pooled bpp vs mean Qhat v0, and vs mean LPIPS) over all levels and over levels 0-3 only
(lambda 16-2, about 0.04-0.13 bpp on CLIC after the 24/11 calibration: the 0.075 region, measured separately as the plan requires). Gate: keep a change if
BD-rate <= -2 % (Qhat v0 and LPIPS); total <= -5 %.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
VALID = Path(os.environ.get('STEPD_VALID', '/mnt/data/khoitm8/clic2027/data/valid30'))  # set by run_stepd.sh off lab09


def bd(r1, q1, r2, q2):
    """BD-rate (%) of curve 2 vs curve 1 (cubic fit of log-rate over quality, overlap interval). Same formula as
    experiments/elic/compare.py. Returns None without overlap or with fewer than 4 points."""
    if len(r1) < 4 or len(r2) < 4:
        return None
    lr1, lr2 = np.log(r1), np.log(r2)
    lo, hi = max(min(q1), min(q2)), min(max(q1), max(q2))
    if lo >= hi:
        return None
    p1, p2 = np.polyint(np.polyfit(q1, lr1, 3)), np.polyint(np.polyfit(q2, lr2, 3))
    d = ((np.polyval(p2, hi) - np.polyval(p2, lo)) - (np.polyval(p1, hi) - np.polyval(p1, lo))) / (hi - lo)
    return (math.exp(d) - 1) * 100


def encode(a):
    import torch
    from PIL import Image
    from clic27.gc import codec
    from clic27.gc.model import GCodec
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    m = GCodec(sd['cfg']['model'])
    m.load_state_dict(sd['state_dict'], strict=False)
    m = m.cuda().eval()
    m.update_tables()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'meta.json').write_text(json.dumps(dict(ckpt=str(a.ckpt), stage=sd.get('stage'), step=sd.get('step'),
                                                     model=sd['cfg']['model'], beta=a.beta), indent=1))
    paths = sorted(VALID.glob('*.png'))
    paths = paths[: int(os.environ.get('STEPD_MAX_IMAGES', 0)) or None]  # smoke tests only
    for level in range(m.L):
        js = a.out / f'q{level}-decode.json'
        if js.exists():
            continue
        rec = a.out / f'q{level}' / 'recon'
        rec.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in paths:
            x = np.asarray(Image.open(p).convert('RGB'))
            xt = torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda() / 255
            buf = codec.compress(m, xt, level, a.beta, None, ent_dev='cuda')
            t = time.perf_counter()
            img = codec.decompress(m, buf, dev='cuda', ent_dev='cuda')
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            Image.fromarray(img).save(rec / p.name, compress_level=1)
            H, W = x.shape[:2]
            rows.append(dict(name=p.stem, width=W, height=H, bytes=len(buf), bpp=len(buf) * 8 / (H * W), decode_s=dt,
                             pixel_sha256=hashlib.sha256(img.tobytes()).hexdigest()))
        js.write_text(json.dumps(dict(level=level, images=rows), indent=1))
        bpp = sum(r['bytes'] for r in rows) * 8 / sum(r['width'] * r['height'] for r in rows)
        print(f'level {level}: {bpp:.4f} bpp', flush=True)


def curve(run: Path):
    import csv
    rows = list(csv.DictReader((run / 'metrics.csv').open()))
    out = {}
    for q in sorted({r['quality'] for r in rows}, key=int):
        rr = [r for r in rows if r['quality'] == q]
        if len(rr) != 30:
            raise ValueError(f'{run}: level {q} has {len(rr)} images, expected 30')
        bpp = sum(float(r['bytes']) for r in rr) * 8 / sum(float(r['pixels']) for r in rr)
        lp = 'lpips_alex' if 'lpips_alex' in rr[0] else 'lpips'
        out[int(q)] = (bpp, np.mean([float(r['qhat_v0']) for r in rr]), np.mean([float(r[lp]) for r in rr]))
    return out


def report(a):
    ref = curve(a.ref)
    lines = ['| run | BD-rate Qhat v0, all | BD-rate LPIPS, all | BD-rate Qhat v0, levels 0-3 | BD-rate LPIPS, levels 0-3 |',
             '|---|---:|---:|---:|---:|']
    out = {}
    for t in a.test:
        c = curve(t)
        row = {}
        for tag, lv in (('all', sorted(set(ref) & set(c))), ('low', [l for l in range(4) if l in ref and l in c])):
            r1, r2 = [ref[l][0] for l in lv], [c[l][0] for l in lv]
            row[f'qhat_{tag}'] = bd(r1, [ref[l][1] for l in lv], r2, [c[l][1] for l in lv])
            row[f'lpips_{tag}'] = bd(r1, [-ref[l][2] for l in lv], r2, [-c[l][2] for l in lv])
        out[str(t)] = row
        f = lambda v: '-' if v is None else f'{v:+.2f}%'  # noqa: E731
        lines.append(f"| {t.name} | {f(row['qhat_all'])} | {f(row['lpips_all'])} | {f(row['qhat_low'])} | {f(row['lpips_low'])} |")
    print('\n'.join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(ref=str(a.ref), runs=out), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('encode')
    e.add_argument('--ckpt', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    e.add_argument('--beta', type=int, default=0)
    b = sub.add_parser('bd')
    b.add_argument('--ref', type=Path, required=True)
    b.add_argument('--test', type=Path, nargs='+', required=True)
    b.add_argument('--json')
    a = p.parse_args()
    {'encode': encode, 'bd': report}[a.cmd](a)


if __name__ == '__main__':
    main()
