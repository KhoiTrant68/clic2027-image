"""Phase 0 inputs (plans/GC_entropy_MLICpp_design_2026-11-20.md): frozen MLIC++ analysis latents.

    python extract.py --quality Q --upstream refs/MLIC --weights mlicpp/weights --data div2k_train --valid valid30 --out DIR

Writes, for one MLIC++ MSE checkpoint:
  * clic30.pt: continuous y = g_a(x) of the 30 CLIC images (replicate-padded to multiples of 64, fp32) and, per image,
    the bits MLIC++'s own entropy model assigns to the same padded input (eval forward: discrete likelihoods of
    round(y - mu) + mu and of z; the reference R of the design, before any coding overhead);
  * pool.pt: y of a fixed pool of DIV2K crops (same GPU crop generator and seed as experiments/gc_backbone), fp16.
Every entropy-model variant of rate_only.py trains and is scored on exactly these tensors, so bpp differences come
from the entropy model alone. Runs in the MLIC++ env (torch 2.0.1 + CompressAI 1.2.0b3).
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/gc_backbone'))
import train_gs as T  # noqa: E402  (GPUCrops, build_codec; torch is imported lazily there)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality', type=int, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--valid', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pool', type=int, default=65536)
    p.add_argument('--pool-batch', type=int, default=64)
    p.add_argument('--crop', type=int, default=256)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))
    from half_vram import install
    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / 'extract.vram.jsonl')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 as in the runners
    a.arm = 'mlicpp'
    model, meta = T.build_codec(a)
    t0 = time.time()
    if not (a.out / 'clic30.pt').exists():
        ys, rows = [], []
        for f in sorted(a.valid.glob('*.png')):
            x = torch.from_numpy(np.array(Image.open(f).convert('RGB'))).permute(2, 0, 1)[None].float().div(255).cuda()
            h, w = x.shape[-2:]
            xp = F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode='replicate')
            with torch.no_grad():
                y = model.g_a(xp)
                o = model(xp)
            bits = sum(float(-torch.log2(l).sum()) for l in o['likelihoods'].values())
            ys.append(y.float().cpu())
            rows.append(dict(name=f.stem, H=h, W=w, pixels=h * w, mlicpp_est_bits=bits))
        if len(rows) != 30:
            raise SystemExit(f'expected 30 CLIC images, found {len(rows)}')
        torch.save(dict(y=ys, rows=rows, codec=meta), a.out / 'clic30.pt')
        est = sum(r['mlicpp_est_bits'] for r in rows) / sum(r['pixels'] for r in rows)
        print(f'clic30: MLIC++ estimated {est:.4f} bpp ({time.time() - t0:.0f} s)', flush=True)
    if not (a.out / 'pool.pt').exists():
        random.seed(12345)
        data = T.GPUCrops(a.data, a.crop, 'cuda')
        chunks = []
        with torch.no_grad():
            for i in range(0, a.pool, a.pool_batch):
                x = data.batch(min(a.pool_batch, a.pool - i)).mul(255).round().div(255)  # 8-bit crops
                chunks.append(model.g_a(x).half().cpu())
        del data
        y = torch.cat(chunks)
        torch.save(dict(y=y, seed=12345, crop=a.crop, data=str(a.data)), a.out / 'pool.tmp')
        (a.out / 'pool.tmp').rename(a.out / 'pool.pt')
        print(f'pool: {tuple(y.shape)} fp16, |y| max {float(y.abs().max()):.1f} ({time.time() - t0:.0f} s)', flush=True)
    (a.out / 'extract.json').write_text(json.dumps(dict(quality=a.quality, pool=a.pool, crop=a.crop, codec=meta),
                                                   indent=1, default=str))
    stop.set()
    print('EXTRACT_DONE', flush=True)


if __name__ == '__main__':
    main()
