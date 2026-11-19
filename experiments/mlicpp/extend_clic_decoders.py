"""Evaluate extra synthesis checkpoints against the frozen CLIC30 package bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from geometry_codec import Quantizer
from probe import ROOT, load, settings, sha, write_json
from stream import unpack
from spatial_synthesis import load_synthesis
from entropy_synthesis import decode


def run(args):
    free = int(subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits',
    ], text=True).strip())
    if free < 12288:
        raise RuntimeError('Need at least 12 GiB of free VRAM before starting')
    torch, _ = settings()
    torch.cuda.set_per_process_memory_fraction(8192 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    config = json.loads((args.source / 'config.json').read_text())
    model, _ = load(SimpleNamespace(upstream=Path(config['upstream']), weights=Path(config['weights']), legacy=True), 2)
    model.eval().requires_grad_(False)
    model.update(force=True)
    grid = args.source / 'grid'
    references = grid / 'development32-full/images'
    summary = json.loads((args.source / 'budgets/summary.json').read_text())
    by_rate, points = {}, {}
    for rate in summary:
        budget = rate['target_bpp']
        selection = json.loads((args.source / f'budgets/bpp-{budget:g}/selection.json').read_text())
        if not rate['feasible'] or len(selection) != 30 or len({r['name'] for r in selection}) != 30:
            raise ValueError('Expected 30 unique images at every budget')
        by_rate[budget] = selection
        for row in selection:
            points[row['factor'], row['name']] = row
    records = json.loads((grid / 'development32-full/manifest.json').read_text())['images']
    source_hashes = {row['name']: row['sha256'] for row in records}
    for name, expected in source_hashes.items():
        if sha(references / name) != expected:
            raise ValueError(f'Reference hash mismatch: {name}')
    expected = {}
    for factor in {factor for factor, _ in points}:
        folder = grid / f'original-factor{factor:g}'
        for row in json.loads((folder / 'metrics.json').read_text()):
            expected[factor, row['name']] = row
    sys.path.insert(0, str(ROOT / 'src'))
    from clic27.eval.metrics import Perceptual
    metric = Perceptual('cuda')
    outputs = {}
    synthesis = model.g_s
    for arm in ['original', *args.arms]:
        if arm != 'original':
            checkpoint = args.checkpoints / arm / 'last.pt'
            model.g_s = synthesis
            load_synthesis(model, torch.load(checkpoint, map_location='cpu', weights_only=True))
            model.g_s.requires_grad_(False)
            write_json(args.out / f'{arm}-checkpoint.json', {'sha256': sha(checkpoint)})
        quantizer = None
        active_factor = None
        try:
            with torch.inference_mode():
                for index, ((factor, name), point) in enumerate(sorted(points.items()), 1):
                    if active_factor != factor:
                        if quantizer:
                            quantizer.close()
                        quantizer = Quantizer(model, None, None) if factor != 1 else None
                        if quantizer:
                            quantizer.select('uniform', factor)
                        active_factor = factor
                    original = expected[factor, name]
                    stream = grid / f'original-factor{factor:g}' / 'streams' / original['stream']
                    blob = stream.read_bytes()
                    digest = hashlib.sha256(blob).hexdigest()
                    if digest != point['stream_sha256'] or len(blob) != point['bytes']:
                        raise ValueError(f'Bitstream mismatch: {name}')
                    quality, legacy, (h, w), shape, strings = unpack(blob)
                    if quality != 2 or not legacy or h * w != point['pixels']:
                        raise ValueError(f'Wrong stream metadata: {name}')
                    result = decode(model, strings, shape, factor)['x_hat'][0, :, :h, :w]
                    pixels = result.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                    pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if arm == 'original':
                        if pixel_digest != original['pixel_sha256']:
                            raise ValueError(f'Original pixel parity failed: {name} factor={factor:g}')
                    else:
                        recon = args.out / 'recon' / arm / f'{factor:g}' / name
                        recon.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(pixels).save(recon)
                        with Image.open(references / name) as ref:
                            image = np.asarray(ref.convert('RGB')).copy()
                        outputs[arm, factor, name] = dict(
                            name=name, factor=factor, pixels=h*w, bytes=len(blob), stream_sha256=digest,
                            pixel_sha256=pixel_digest, recon=recon.relative_to(args.out).as_posix(),
                            **metric.all(image, pixels),
                        )
                    write_json(args.out / 'status.json', {'state': 'running', 'arm': arm, 'images': index, 'total': len(points)})
        finally:
            if quantizer:
                quantizer.close()
    result_rows, per_image = [], []
    for rate in summary:
        budget = rate['target_bpp']
        for arm in args.arms:
            rows = [outputs[arm, r['factor'], r['name']] for r in by_rate[budget]]
            per_image.extend(dict(r, model=arm, target_bpp=budget) for r in rows)
            result_rows.append(dict(model=arm, target_bpp=budget, actual_bpp=rate['actual_package_bpp'],
                                    package_bytes=rate['package_bytes'], n=len(rows),
                                    **{key: float(np.mean([row[key] for row in rows])) for key in ['psnr', 'lpips', 'dists']}))
    write_json(args.out / 'comparison.json', result_rows)
    write_json(args.out / 'per_image.json', per_image)
    write_json(args.out / 'status.json', {'state': 'complete', 'verified_original': len(points), 'decoded': len(outputs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'checkpoints', 'out']:
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--arms', nargs='+', choices=['plain', 'global', 'local', 'single', 'mixed', 'entropy', 'entropy-blind'],
                        default=['single', 'mixed'])
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('Each arm may be listed once')
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_json(args.out / 'status.json', {'state': 'failed', 'error': repr(exc)})
        raise
