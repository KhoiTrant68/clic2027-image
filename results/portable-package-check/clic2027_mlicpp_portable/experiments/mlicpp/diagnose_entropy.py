"""Controlled one-image trace of encoder/decoder CDF indexes; does not rank quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--quality', type=int, default=1)
    p.add_argument('--entropy-no-cudnn', action='store_true')
    p.add_argument('--warmup-prefix', type=Path, help='Replay earlier validation images before tracing the target')
    p.add_argument('--reset-before-encode', action='store_true', help='Match the official test loop cache reset')
    a = p.parse_args()
    a.legacy = True
    a.out.mkdir(parents=True, exist_ok=False)
    torch, install = probe.settings()
    stop = install(a.out / 'trace.vram.jsonl')
    import numpy as np
    from PIL import Image
    model, meta = probe.load(a, a.quality)
    import models.mlicpp as mlic
    if a.entropy_no_cudnn:
        from compatibility import entropy_no_cudnn
        entropy_no_cudnn(model)
    if a.warmup_prefix:
        for previous in sorted(a.warmup_prefix.glob('*.png')):
            if previous.name >= a.image.name:
                break
            with Image.open(previous) as im:
                pixels = np.array(im.convert('RGB'))
            h, w = pixels.shape[:2]
            x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
            x = torch.nn.functional.pad(x, (0, (-w) % 64, 0, (-h) % 64))
            if a.reset_before_encode:
                model.update_resolutions(16, 16)
            with torch.inference_mode():
                encoded = model.compress(x)
                model.update_resolutions(16, 16)
                y = model.decompress(encoded['strings'], encoded['shape'])['x_hat']
            print(json.dumps(dict(warmup=previous.stem, finite=bool(torch.isfinite(y).all()))), flush=True)
            del x, y, encoded
    records, phase = [], ['encode']
    for name in ['compress_anchor', 'compress_nonanchor', 'decompress_anchor', 'decompress_nonanchor']:
        original = getattr(mlic, name)
        squeeze = mlic.ckbd_anchor_sequeeze if name.endswith('_anchor') else mlic.ckbd_nonanchor_sequeeze
        def traced(*args, original=original, name=name, squeeze=squeeze):
            gc = args[0]
            scales = args[2] if name.startswith('compress_') else args[1]
            idx = gc.build_indexes(squeeze(scales)).cpu().numpy().astype('<i4')
            value = original(*args)
            row = dict(phase=phase[0], function=name, index_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
                       index_count=idx.size, finite=bool(torch.isfinite(value).all()),
                       max_abs=float(value.abs().max()))
            records.append(row)
            print(json.dumps(row), flush=True)
            return value
        setattr(mlic, name, traced)
    with Image.open(a.image) as im:
        pixels = np.array(im.convert('RGB'))
    h, w = pixels.shape[:2]
    x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    x = torch.nn.functional.pad(x, (0, (-w) % 64, 0, (-h) % 64))
    if a.reset_before_encode:
        model.update_resolutions(16, 16)
    with torch.inference_mode():
        encoded = model.compress(x)
        blob = probe.pack(a.quality, True, (h, w), tuple(encoded['shape']), encoded['strings'])
        (a.out / 'stream.bin').write_bytes(blob)
        _, _, size, shape, strings = probe.unpack(blob)
        del encoded, x
        model.update_resolutions(16, 16)
        phase[0] = 'decode'
        y = model.decompress(strings, shape)['x_hat']
    enc = [r for r in records if r['phase'] == 'encode']
    dec = [r for r in records if r['phase'] == 'decode']
    bad = [i for i, (x, y) in enumerate(zip(enc, dec)) if x['index_sha256'] != y['index_sha256']]
    result = dict(metadata=meta, entropy_no_cudnn=a.entropy_no_cudnn, reset_before_encode=a.reset_before_encode,
                  stream_sha256=hashlib.sha256(blob).hexdigest(), records=records,
                  first_index_mismatch=bad[0] if bad else None, index_mismatches=len(bad),
                  decoded_finite=bool(torch.isfinite(y).all()), decoded_shape=list(y.shape))
    (a.out / 'trace.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'metadata')}), flush=True)
    stop.set()


if __name__ == '__main__':
    main()
