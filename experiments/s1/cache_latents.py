"""Cache DC-AE f32c32 latents of whole training images (crops are taken in latent space at train time).

One fp16 .npy per image, UNscaled (multiply by the DC-AE scaling factor at load time), in --out.
Tiled encoding (512/448) exactly as in ratflow.nn.dcae / diffusers. Skips files that already exist.

    python cache_latents.py --images data/DIV2K_train_HR [more dirs] --out latents --dtype fp16
    python cache_latents.py --download-div2k --out latents
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ratflow.eval.metrics import device, images, load  # noqa: E402
from ratflow.nn.dcae import DCAE  # noqa: E402

DIV2K_TRAIN = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
AE_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


def download_div2k(root: Path) -> Path:
    d = root / "DIV2K_train_HR"
    if d.exists() and len(images(d)) >= 800:
        return d
    root.mkdir(parents=True, exist_ok=True)
    z = root / "div2k_train.zip"
    subprocess.run(["wget", "-q", "-O", str(z), DIV2K_TRAIN], check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(root)
    z.unlink()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--download-div2k", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("latents"))
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--min-side", type=int, default=256, help="skip images smaller than this")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.images]
    if args.download_div2k:
        dirs.append(download_div2k(args.data))
    files = [p for d in dirs for p in images(d)]
    assert files, "no images"
    args.out.mkdir(parents=True, exist_ok=True)

    dev = device()
    dtype = torch.float16 if args.dtype == "fp16" and dev == "cuda" else torch.float32
    ae = DCAE.from_pretrained(AE_REPO, "vae", dtype=dtype).to(dev).enable_tiling()
    meta = dict(ae_repo=AE_REPO, scaling_factor=ae.scaling_factor, dtype=str(dtype), f=ae.f, files={})
    t0, done = time.time(), 0
    for i, p in enumerate(files):
        key = f"{p.parent.name}__{p.stem}"
        dst = args.out / f"{key}.npy"
        if dst.exists():
            continue
        x = load(p)
        H, W = x.shape[:2]
        if min(H, W) < args.min_side:
            continue
        H32, W32 = H - H % ae.f, W - W % ae.f  # crop to a multiple of 32 (no padding artefacts in the latents)
        t = torch.from_numpy(np.ascontiguousarray(x[:H32, :W32])).permute(2, 0, 1)[None].to(dev, dtype)
        with torch.no_grad():
            z = ae.encode(t.div(127.5).sub(1))
        np.save(dst, z[0].float().cpu().numpy().astype(np.float16))
        meta["files"][key] = [int(z.shape[2]), int(z.shape[3])]
        done += 1
        if done % 50 == 0:
            print(f"{i + 1}/{len(files)}  {done / (time.time() - t0):.2f} img/s", flush=True)
    old = json.loads((args.out / "index.json").read_text()) if (args.out / "index.json").exists() else {"files": {}}
    meta["files"] = {**old.get("files", {}), **meta["files"]}
    (args.out / "index.json").write_text(json.dumps(meta, indent=1))
    print("cached", len(meta["files"]), "latents in", args.out)


if __name__ == "__main__":
    main()
