"""Image-clustered paired bootstrap and lightweight evidence for a completed pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def tail(values, alpha=.9):
    values = np.sort(np.asarray(values).ravel())[::-1]
    mass = (1-alpha)*len(values)
    whole = int(mass)
    fraction = mass-whole
    return (values[:whole].sum() + (fraction*values[min(whole, len(values)-1)] if fraction > 1e-8 else 0))/mass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.run / 'status.json').read_text())
    if status.get('state') != 'complete' or not status.get('fresh_pixel_parity'):
        raise ValueError('Require all stages, metrics and fresh parity complete')
    cfg = json.loads((a.run / 'planned_config.json').read_text())
    cache = Path(cfg['cache'])
    manifest = json.loads((cache / 'manifest.json').read_text())
    rows = json.loads((a.run / 'test_metrics.json').read_text())
    methods = ['base', 'mean', 'cvar', 'risk', 'risk_hide_change']
    names = sorted(r['name'] for r in rows if r['method'] == 'base')
    if len(names) != len(set(names)) or len(names) != 32:
        raise ValueError('Require the planned 32 held-out images')
    by = {(r['method'], r['name']): r for r in rows}
    if len(by) != len(rows) or set(by) != {(m, n) for m in methods for n in names}:
        raise ValueError('Incomplete/duplicate test metric grid')
    for name in names:
        if len({by[m, name]['bytes'] for m in methods}) != 1:
            raise ValueError('Arms use different byte budgets')
    for mode in cfg['modes']:
        fresh = json.loads((a.run / f'fresh-{mode}.json').read_text())
        if fresh['matches'] != 32 or {r['name'] for r in fresh['images']} != set(names):
            raise ValueError('Incomplete fresh decode inventory')
        if any(by[mode, r['name']]['pixel_sha256'] != r['pixel_sha256'] for r in fresh['images']):
            raise ValueError('Fresh decode hashes differ')
    arrays = {m: {key: np.asarray([by[m, n][key] for n in names]) for key in
                     ['lpips', 'dists', 'edge_mean', 'edge_regions']} for m in methods}
    indices = np.random.default_rng(1984).integers(0, len(names), size=(2000, len(names)))
    comparisons = []
    for treatment, control in [('mean', 'base'), ('cvar', 'mean'), ('risk', 'mean'),
                               ('risk', 'cvar'), ('risk_hide_change', 'risk')]:
        for key in ['lpips', 'dists', 'edge_mean', 'edge_cvar']:
            def stat(method, sample):
                values = arrays[method]['edge_regions' if key == 'edge_cvar' else key][sample]
                return tail(values, cfg['alpha']) if key == 'edge_cvar' else float(values.mean())
            delta = stat(treatment, np.arange(len(names)))-stat(control, np.arange(len(names)))
            samples = [stat(treatment, i)-stat(control, i) for i in indices]
            lo, hi = np.quantile(samples, [.025, .975])
            comparisons.append(dict(treatment=treatment, control=control, metric=key,
                delta=delta, ci95_low=float(lo), ci95_high=float(hi),
                interpretation='negative is lower error; CI exploratory, no multiplicity adjustment'))
    a.out.mkdir(parents=True, exist_ok=False)
    for path in a.run.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.log'):
            shutil.copyfile(path, a.out / path.name)
    for mode in cfg['modes']:
        dest = a.out / mode
        dest.mkdir()
        for filename in ['config.json', 'progress.json', 'status.json', 'train.jsonl', 'validation.jsonl']:
            shutil.copyfile(a.run / mode / filename, dest / filename)
    if (a.run / 'source_archive').is_dir():
        shutil.copytree(a.run / 'source_archive', a.out / 'source_archive')
    shutil.copyfile(__file__, a.out / 'analysis_source.py')
    shutil.copyfile(cache / 'manifest.json', a.out / 'cache_manifest.json')
    shutil.copyfile(Path(cfg['data']) / 'manifest.json', a.out / 'data_manifest.json')
    (a.out / 'paired_bootstrap.json').write_text(json.dumps(comparisons, indent=2))
    with (a.out / 'paired_bootstrap.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = json.loads((a.out / 'summary.json').read_text())
    lines = ['# Pilot MLIC++ q2 — kết quả test crop256', '',
        '160 ảnh train / 32 validation / 32 test độc lập theo ảnh cho adapter; chưa audit overlap pretraining.', '',
        '| Nhánh | bpp thực | PSNR gộp | LPIPS-Alex ↓ | DISTS ↓ | Edge mean ↓ | Edge CVaR ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['method']} | {r['bpp']:.5f} | {r['psnr_pooled']:.2f} | {r['lpips']:.5f} | {r['dists']:.5f} | {r['edge_mean']:.5f} | {r['edge_cvar']:.5f} |")
    lines += ['', 'Ba nhánh × 1.000 bước, cùng khởi tạo/kiến trúc/data order; không GAN. Byte cố định, checkpoint chọn ở bước cuối.',
        '96/96 ảnh giải mã test bằng tiến trình mới khớp pixel với đường cache-evaluation trên cùng L40.',
        f"CVaR theo vùng, scale từ train; transform={cfg.get('edge_transform', 'clipped')}. Không so giá trị edge giữa hai transform.", '',
        '## Kiểm tra bão hòa surrogate', '']
    saturation = {}
    for mode in methods:
        edge = arrays[mode]['edge_regions']
        saturation[mode] = dict(regions=int(edge.size), fraction_clipped_one=float((edge >= 1-1e-6).mean()))
        lines.append(f"- {mode}: {saturation[mode]['fraction_clipped_one']:.1%} vùng ở ngưỡng 1.")
    (a.out / 'surrogate_saturation.json').write_text(json.dumps(saturation, indent=2))
    lines += ['', '## Giới hạn', '',
        '32 ảnh test, một seed, crop 256, legacy q2 với rate đo được; không phải kết quả full30 tại 0,15 bpp.',
        'LPIPS-Alex có trong loss; DISTS là metric ngoài loss. Chưa có Q̂ đầy đủ, OCR transcript, human study, L4 hoặc PerQ/PICO reproduction.',
        'Bootstrap 2.000 lần theo ảnh là kiểm tra thăm dò, không hiệu chỉnh nhiều phép so sánh và không thay thế chạy nhiều seed.',
        'suffix_s không gồm entropy/load/I/O, không dùng để suy ra thời gian submission.',
        'Checkpoint, stream, ảnh và cache lớn được giữ ngoài Git trên server.']
    (a.out / 'summary.md').write_text('\n'.join(lines)+'\n')
    hashes = {str(p.relative_to(a.out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in a.out.rglob('*') if p.is_file()}
    (a.out / 'evidence.sha256.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps(dict(evidence=str(a.out), files=len(hashes), bootstrap_iterations=2000)))


if __name__ == '__main__':
    main()
