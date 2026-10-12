"""DiT-IC latents for decoder distillation: each image -> encoder -> rANS -> entropy decode -> one-step DiT (bf16, as
deployed) -> the DC-AE latent the decoder sees, cut to the image. Saved per image as fp16 (+ size and bytes).

    python prep.py --q q3 --src ~/clic/data/DIV2K_train_HR --out ~/clic/distill/lat_q3/train
    python prep.py --q q3 --src ~/clic/work_full_gpu/bakeoff/valid --out ~/clic/distill/lat_q3/clic_val
Resumes: images whose file exists are skipped.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ditic_latent, load_ditic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q3")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in a.src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    todo = [p for p in files if not (a.out / f"{p.stem}.pt").exists()]
    print(f"{len(files)} images, {len(todo)} to do", flush=True)
    if not todo:
        return
    net = load_ditic(a.q)
    t0 = time.time()
    for i, p in enumerate(todo):
        x = np.asarray(Image.open(p).convert("RGB"))
        z, nbytes = ditic_latent(net, x)
        tmp = a.out / f"{p.stem}.pt.tmp"
        torch.save(dict(z=z.half().cpu(), H=x.shape[0], W=x.shape[1], bytes=nbytes, q=a.q), tmp)
        tmp.rename(a.out / f"{p.stem}.pt")
        if i % 25 == 0:
            print(f"[{i + 1}/{len(todo)}] {p.stem} {tuple(z.shape)} {nbytes} B  {(time.time() - t0) / (i + 1):.1f} s/img",
                  flush=True)


if __name__ == "__main__":
    main()
