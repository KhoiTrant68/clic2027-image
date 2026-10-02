"""Prepare latents in the layout `denoiser_gonogo.py` expects (<out>/latents/*.npy scaled float32 + encode_config.json).

  --from-cache DIR   reuse the S1 cache (experiments/s1/cache_latents.py: unscaled fp16, full images); no re-encoding
  --kodak            download Kodak (24) and encode with ratflow.nn.DCAE (fp32, tiled; bit-exact with diffusers)
  --clic-valid       CLIC2020 professional valid (41 images, 135 MB), same encoder
  --images DIR       any folder, same encoder

Test images are centre-cropped to a multiple of 64 px so the 2x2 KLT tiles the 32x-downsampled latent exactly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ratflow.eval.metrics import images, load  # noqa: E402

AE_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
SCALING = 0.41407
CLIC_VALID = "https://downloads.compression.cc/clic2020_professional_valid.zip"


def write_config(out: Path, n: int, source: str):
    (out / "encode_config.json").write_text(json.dumps(
        {"vae": AE_REPO, "subfolder": "vae", "class": "AutoencoderDC", "downsample": 32, "shift_factor": 0.0,
         "scaling_factor": SCALING, "n_images": n, "max_side": 0, "source": source}, indent=2))


def from_cache(src: Path, out: Path):
    (out / "latents").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src).glob("*.npy"))
    for f in files:
        z = np.load(f).astype(np.float32) * SCALING
        z = z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2]
        np.save(out / "latents" / f.name, z)
    write_config(out, len(files), f"s1 cache {src}")
    print(len(files), "train latents ->", out)


def encode_dir(img_dir: Path, out: Path):
    import torch

    from ratflow.nn.dcae import DCAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae = DCAE.from_pretrained(AE_REPO, "vae").to(dev).enable_tiling()
    (out / "latents").mkdir(parents=True, exist_ok=True)
    files = images(img_dir)
    for p in files:
        x = load(p)
        H, W = x.shape[0] // 64 * 64, x.shape[1] // 64 * 64
        y0, x0 = (x.shape[0] - H) // 2, (x.shape[1] - W) // 2
        t = torch.from_numpy(np.ascontiguousarray(x[y0:y0 + H, x0:x0 + W])).permute(2, 0, 1)[None].float()
        with torch.no_grad():
            z = ae.encode(t.div(127.5).sub(1).to(dev)) * SCALING
        np.save(out / "latents" / f"{p.stem}.npy", z[0].cpu().numpy().astype(np.float32))
    write_config(out, len(files), str(img_dir))
    print(len(files), "test latents ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-cache", type=Path)
    ap.add_argument("--kodak", action="store_true")
    ap.add_argument("--clic-valid", action="store_true")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()
    if a.from_cache:
        return from_cache(a.from_cache, a.out)
    if a.kodak:
        d = a.data / "kodak"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, 25):
            f = d / f"kodim{i:02d}.png"
            if not f.exists():
                urllib.request.urlretrieve(f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", f)
        return encode_dir(d, a.out)
    if a.clic_valid:
        d = a.data / "clic2020_valid"
        if not (d.exists() and images(d)):
            d.mkdir(parents=True, exist_ok=True)
            z = d / "v.zip"
            subprocess.run(["wget", "-q", "-O", str(z), CLIC_VALID], check=True)
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
            z.unlink()
        return encode_dir(d, a.out)
    assert a.images, "nothing to do"
    encode_dir(a.images, a.out)


if __name__ == "__main__":
    main()
