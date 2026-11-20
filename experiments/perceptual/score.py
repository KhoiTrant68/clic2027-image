"""Score any codec run laid out like experiments/elic (qN-decode.json + qN/recon/*.png) with the repo Qhat metrics.

Model-specific runners only have to produce, for each operating point N:
  <run>/qN-decode.json   {"images": [{"name", "width", "height", "bytes", "bpp", "decode_s", "pixel_sha256"}, ...], ...}
  <run>/qN/recon/<name>.png   decoded in a fresh process from the saved stream only
Resumable: rows already in <run>/metrics.csv are kept.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--qualities', nargs='+', required=True)
    p.add_argument('--qhat', type=Path, default=ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json')
    a = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for sub in ('src', 'experiments/qhat', 'experiments/bakeoff', 'experiments/new_codecs'):
        sys.path.insert(0, str(ROOT / sub))
    from half_vram import install
    stop = install(a.run / 'metrics.vram.jsonl')
    from qhat import Features
    from bakeoff import qhat_score
    from clic27.eval import metrics as M
    model = json.loads(a.qhat.read_text())
    feat = Features('cuda')
    if set(model['features']) - set(feat.names()):
        raise ValueError('Qhat features unavailable; refusing to substitute a different objective')
    out = a.run / 'metrics.csv'
    rows = M.read_csv(out) if out.exists() else []
    done = {(r['quality'], r['name']) for r in rows}
    for q in a.qualities:
        for img in json.loads((a.run / f'q{q}-decode.json').read_text())['images']:
            if (str(q), img['name']) in done:
                continue
            x = M.load(a.data / (img['name'] + '.png'))
            y = M.load(a.run / f'q{q}/recon' / (img['name'] + '.png'))
            if x.shape != y.shape:
                raise ValueError(f'Shape mismatch for {img["name"]} q{q}')
            values = feat(x, y, only=model['features'])
            values['mse'] = M.mse(x, y)
            values['qhat_v0'] = qhat_score(model, values)
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Non-finite metric')
            rows.append(dict(quality=q, name=img['name'], bytes=img['bytes'], pixels=img['width'] * img['height'],
                             bpp=img['bpp'], decode_s=img['decode_s'], **values))
            tmp = out.with_suffix('.csv.tmp')
            with tmp.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[-1]))
                w.writeheader()
                w.writerows(rows)
            tmp.replace(out)
            print(json.dumps(rows[-1]), flush=True)
    (a.run / 'metrics.meta.json').write_text(json.dumps(dict(
        qhat_sha256=hashlib.sha256(a.qhat.read_bytes()).hexdigest(), qualities=a.qualities, rows=len(rows)), indent=2))
    stop.set()


if __name__ == '__main__':
    main()
