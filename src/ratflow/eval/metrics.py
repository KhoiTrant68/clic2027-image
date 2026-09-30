"""Shared evaluation code: every number in the paper and in the CLIC analyses goes through here,
so ceilings, baselines and our codec stay comparable.

Conventions
- Images are HxWx3 uint8 numpy arrays.
- PSNR per image from the RGB MSE. Dataset-level "CLIC PSNR" pools MSE weighted by pixel count.
- LPIPS (alex, inputs in [-1, 1]) and DISTS are computed on 512x512 tiles, weighted by tile area
  (tiles with a side < 64 are skipped), so 2K+ images fit in GPU memory.
- MS-SSIM on the full image (NaN if the short side is < 161, the minimum for 5 scales).
- FID/KID on 256x256 patches, HiFiC protocol: a grid plus a grid shifted by 128 in both directions.
- OCR CER: easyocr on the same crops (boxes detected on the original) for reference and reconstruction.

torch, lpips, piq, torchmetrics and easyocr are imported lazily so the numpy helpers work without them.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 512
FID_PATCH = 256
IMG_EXTS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------- io
def load(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def save(a: np.ndarray, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)


def images(d: Path, exts=IMG_EXTS):
    """Sorted image files under d, skipping macOS zip debris (__MACOSX/, ._*)."""
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in exts
                  and not p.name.startswith("._") and "__MACOSX" not in p.parts)


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- torch helpers
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(a: np.ndarray, dev):
    """HxWx3 uint8 -> 1x3xHxW float in [0, 1]."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().div(255).to(dev)


# ---------------------------------------------------------------- fidelity
def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(np.float64) - b) ** 2).mean())


def psnr_from_mse(m: float) -> float:
    return 10 * math.log10(255 ** 2 / max(m, 1e-10))


def psnr(a, b) -> float:
    return psnr_from_mse(mse(a, b))


def pooled_psnr(mses, pixels) -> float:
    """CLIC-style dataset PSNR: MSE averaged with pixel-count weights."""
    return psnr_from_mse(sum(m * n for m, n in zip(mses, pixels)) / sum(pixels))


def msssim(a, b, dev) -> float:
    import piq
    import torch
    if min(a.shape[:2]) < 161:
        return float("nan")
    with torch.no_grad():
        return float(piq.multi_scale_ssim(to_tensor(a, dev), to_tensor(b, dev), data_range=1.0))


# ---------------------------------------------------------------- perceptual
def tiled(fn, a, b, dev, tile=TILE) -> float:
    """Area-weighted mean of fn(tile_a, tile_b) over tile x tile crops; fn takes [0,1] tensors."""
    import torch
    H, W = a.shape[:2]
    tot = wsum = 0.0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            pa, pb = a[y:y + tile, x:x + tile], b[y:y + tile, x:x + tile]
            if min(pa.shape[:2]) < 64:
                continue
            n = pa.shape[0] * pa.shape[1]
            with torch.no_grad():
                tot += float(fn(to_tensor(pa, dev), to_tensor(pb, dev))) * n
            wsum += n
    return tot / wsum


class Perceptual:
    """LPIPS-alex and DISTS with the tiling convention above."""

    def __init__(self, dev=None):
        import lpips
        import piq
        self.dev = dev or device()
        self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.dev)
        self._dists = piq.DISTS().to(self.dev)

    def lpips(self, a, b) -> float:
        return tiled(lambda x, y: self._lpips(x * 2 - 1, y * 2 - 1).mean(), a, b, self.dev)

    def dists(self, a, b) -> float:
        return tiled(lambda x, y: self._dists(x, y), a, b, self.dev)

    def all(self, a, b) -> dict:
        m = mse(a, b)
        return dict(mse=m, psnr=psnr_from_mse(m), msssim=msssim(a, b, self.dev),
                    lpips=self.lpips(a, b), dists=self.dists(a, b))


# ---------------------------------------------------------------- FID / KID
def patches(a: np.ndarray, size=FID_PATCH):
    """HiFiC protocol: size x size grid, plus the grid shifted by size // 2."""
    H, W = a.shape[:2]
    for off in (0, size // 2):
        for y in range(off, H - size + 1, size):
            for x in range(off, W - size + 1, size):
                yield a[y:y + size, x:x + size]


def fid_kid(pairs, dev=None, batch=64, kid_subset=1000) -> dict:
    """pairs: iterable of (original, reconstruction) uint8 arrays. Returns fid, kid, kid_std, n_patches."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    dev = dev or device()
    f = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    k = KernelInceptionDistance(subset_size=kid_subset, normalize=False).to(dev)

    def push(ps, real):
        t = torch.from_numpy(np.stack(ps)).permute(0, 3, 1, 2).to(dev)  # uint8 NCHW
        f.update(t, real=real)
        k.update(t, real=real)

    n = 0
    for x, y in pairs:
        pr, pf = list(patches(x)), list(patches(y))
        for i in range(0, len(pr), batch):
            push(pr[i:i + batch], True)
            push(pf[i:i + batch], False)
        n += len(pr)
    km, ks = k.compute()
    return dict(fid=float(f.compute()), kid=float(km), kid_std=float(ks), n_patches=n)


# ---------------------------------------------------------------- text legibility (OCR)
def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ocr_reader():
    import easyocr
    import torch
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


def box_xyxy(bbox, H, W, pad=4):
    """easyocr quadrilateral -> padded, clipped axis-aligned box (x0, y0, x1, y1)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(W, int(max(xs)) + pad), min(H, int(max(ys)) + pad))


def detect_text(reader, img, min_conf=0.4, min_len=2):
    """[(box_xyxy, text)] for confident detections on the original image."""
    H, W = img.shape[:2]
    return [(box_xyxy(b, H, W), t) for b, t, c in reader.readtext(img) if c > min_conf and len(t.strip()) >= min_len]


def read_boxes(reader, img, boxes):
    return ["".join(t for _, t, _ in reader.readtext(img[y0:y1, x0:x1])) for x0, y0, x1, y1 in boxes]


def cer(refs, hyps) -> float:
    """Character error rate pooled over boxes; NaN if the reference has no characters."""
    n = sum(len(r) for r in refs)
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / n if n else float("nan")


def region_psnr(a, b, boxes) -> float:
    mask = np.zeros(a.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return psnr_from_mse(float(((a[mask].astype(np.float64) - b[mask]) ** 2).mean())) if mask.any() else float("nan")
