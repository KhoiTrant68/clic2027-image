"""Training data: random crops from image folders (photos + optional screen/game content)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(dirs, strict=False, label="images") -> list[Path]:
    """Images under each folder (recursive) or listed in a .txt file. With strict=True a missing or empty
    source is an error: a mistyped data path must not silently shrink a long training run."""
    out, counts = [], []
    for d in dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".txt":  # a list file, one path per line
            found = [Path(p.strip()) for p in d.read_text().splitlines() if p.strip()]
        elif d.is_dir():
            found = sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._"))
        else:
            found = None
        if strict and not found:
            raise FileNotFoundError(f"{label}: {'missing' if found is None else 'no images in'} {d}. Fix data paths "
                                    f"(see docs/GC_HUONG_DAN.md, section on data) or remove it from the config.")
        counts.append((str(d), len(found or [])))
        out += found or []
    if strict:
        print(f"[data] {label}: " + ", ".join(f"{d} = {n}" for d, n in counts) + f" (total {len(out)})", flush=True)
    return out


def sources_fingerprint(dirs) -> dict:
    """What a patch set was cut from: each source and its image count."""
    return {"sources": [[str(Path(d).resolve()), len(list_images([d]))] for d in dirs]}


def patches_ready(out, src_dirs) -> bool:
    """True if `out` holds patches cut from exactly these sources (same folders, same image counts).
    Raises if patches exist but were cut from something else, so stale patches are never trained on."""
    done = Path(out) / ".done"
    if not done.exists():
        return False
    try:
        meta = json.loads(done.read_text())
    except ValueError:  # pre-fingerprint marker (a bare count): cannot verify the sources
        print(f"[prep] {out} has an old .done without sources; delete the folder to re-cut if data changed", flush=True)
        return True
    want = sources_fingerprint(src_dirs)["sources"]
    if meta.get("sources") != want:
        raise RuntimeError(f"patches in {out} were cut from {meta.get('sources')}, but the config now lists {want}. "
                           f"Delete {out} to re-cut them from the current data.")
    return True


class CropDataset(Dataset):
    def __init__(self, dirs, screen_dirs=(), screen_prob=0.0, crop=256, min_side=256, downscale=(0.5, 1.0),
                 length=None):
        self.photos = list_images(dirs, strict=True, label="train_dirs")
        self.screens = list_images(screen_dirs, strict=True, label="screen_dirs")
        if not self.photos and not self.screens:
            raise FileNotFoundError(f"no training images in {list(dirs) + list(screen_dirs)}")
        self.screen_prob = screen_prob if self.screens else 0.0
        self.crop, self.min_side, self.downscale = crop, min_side, downscale
        self.length = length or max(len(self.photos) + len(self.screens), 1000)

    def __len__(self):
        return self.length

    def _load(self, p: Path, photo: bool) -> Image.Image:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if photo and min(w, h) > 2 * self.crop:  # downscale big photos (removes JPEG/noise), keeps >= crop
            lo = max(self.downscale[0], self.crop / min(w, h))
            s = random.uniform(lo, max(lo, self.downscale[1]))
            if s < 0.999:
                im = im.resize((max(self.crop, round(w * s)), max(self.crop, round(h * s))), Image.BICUBIC)
        w, h = im.size
        if min(w, h) < self.crop:  # upscale tiny images just enough to crop
            s = self.crop / min(w, h)
            im = im.resize((max(self.crop, round(w * s)), max(self.crop, round(h * s))), Image.BICUBIC)
        return im

    def __getitem__(self, i):
        for _ in range(10):
            screen = self.screens and random.random() < self.screen_prob
            pool = self.screens if screen or not self.photos else self.photos
            p = random.choice(pool)
            try:
                im = self._load(p, photo=not screen)
                break
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
        w, h = im.size
        x0, y0 = random.randint(0, w - self.crop), random.randint(0, h - self.crop)
        a = np.asarray(im.crop((x0, y0, x0 + self.crop, y0 + self.crop)), np.uint8)
        if random.random() < 0.5:
            a = a[:, ::-1]
        if not screen and random.random() < 0.5:
            a = a[::-1]
        return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255


def _prep_one(args):
    p, out, k, size, seed = args
    rnd = random.Random(seed)
    try:
        im = Image.open(p).convert("RGB")
    except Exception:  # noqa: BLE001
        return 0
    w, h = im.size
    if min(w, h) > 2 * size:  # same downscale idea as training: removes compression noise of large photos
        s = rnd.uniform(max(0.5, size / min(w, h)), 1.0)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
    if min(w, h) < size:
        return 0
    n = 0
    for i in range(k):
        x0, y0 = rnd.randint(0, w - size), rnd.randint(0, h - size)
        im.crop((x0, y0, x0 + size, y0 + size)).save(out / f"{p.stem}_{seed % 100000}_{i}.png", compress_level=1)
        n += 1
    return n


def prep_patches(src_dirs, out, per_image=4, size=512, workers=32):
    """Pre-cut random size x size patches (PNG) from big training images, so the loader reads small files.
    Skips if `out/.done` exists."""
    from concurrent.futures import ProcessPoolExecutor
    out = Path(out)
    if patches_ready(out, src_dirs):
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} has leftover patches from an unfinished cut; delete it and run the data stage again")
    out.mkdir(parents=True, exist_ok=True)
    files = list_images(src_dirs, strict=True, label="train_dirs (for patches)")
    jobs = [(p, out, per_image, size, i) for i, p in enumerate(files)]
    total = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(_prep_one, jobs, chunksize=16)):
            total += n
            if i % 5000 == 0:
                print(f"[prep] {i}/{len(files)} images, {total} patches", flush=True)
    (out / ".done").write_text(json.dumps(dict(sources_fingerprint(src_dirs), patches=total)))
    return out


def random_cells(B, ch, cw, R, p_zero=0.5, gen=None):
    """Random quality maps for training: half all-zero, half smooth random integer fields in [-R, R]."""
    cells = torch.zeros(B, 1, ch, cw, dtype=torch.long)
    for b in range(B):
        if random.random() < p_zero:
            continue
        base = random.randint(-R // 2, R // 2)
        noise = torch.randn(1, 1, max(1, ch // 2), max(1, cw // 2))
        field = torch.nn.functional.interpolate(noise, size=(ch, cw), mode="bilinear", align_corners=False)
        cells[b] = (base + field * random.uniform(0, R / 2)).round().clamp(-R, R).long()[0]
    return cells
