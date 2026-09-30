"""Autoencoder ceilings for the RD plots: DC-AE f32c32 (our backbone) vs SD-VAE f8 (StableCodec's).

An autoencode round trip without quantisation is the best any latent codec on that AE can do,
so it is drawn as a horizontal line on every RD plot.

Datasets: kodak (24), clic2020 (professional + mobile test = 428), div2k (DIV2K valid HR, 100).
Metrics per image: PSNR, MS-SSIM, LPIPS (alex), DISTS (both tiled 512).
FID / KID on 256x256 patches, HiFiC protocol (grid + grid shifted by 128), for clic2020 and div2k.

Steps: download, recon, metrics, fid, summary   (or: all). Each step skips work already done.
python ceiling.py all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATASETS = {
    "kodak": [f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)],
    "clic2020": ["https://downloads.compression.cc/clic2020_professional_test.zip",
                 "https://downloads.compression.cc/clic2020_mobile_test.zip"],
    "div2k": ["https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"],
}
EXPECTED = {"kodak": 24, "clic2020": 428, "div2k": 100}
FID_SETS = ["clic2020", "div2k"]
CFGS = {  # name -> (repo, subfolder, class, spatial multiple)
    "dcae": [("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "vae", "AutoencoderDC", 32)],
    "sdvae": [("stabilityai/sd-turbo", "vae", "AutoencoderKL", 8),
              ("stabilityai/sd-vae-ft-ema", None, "AutoencoderKL", 8)],
}
TILE = 512


# ---------------------------------------------------------------- utils
def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
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


def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_t(a, d):
    import torch
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().div(255).to(d)


# ---------------------------------------------------------------- download
def download(args):
    for ds, urls in DATASETS.items():
        d = args.data / ds
        if d.exists() and len(images(d)) >= EXPECTED[ds]:
            print(ds, "ok", len(images(d)))
            continue
        d.mkdir(parents=True, exist_ok=True)
        for u in urls:
            dst = d / u.split("/")[-1]
            if not dst.exists():
                subprocess.run(["wget", "-q", "-O", str(dst), u], check=True)
            if dst.suffix == ".zip":
                with zipfile.ZipFile(dst) as z:
                    z.extractall(d)
                dst.unlink()
        n = len(images(d))
        print(ds, n, "images" + ("" if n == EXPECTED[ds] else f"  WARNING: expected {EXPECTED[ds]}"))


# ---------------------------------------------------------------- recon
def load_vae(cfg):
    import diffusers
    import torch
    for repo, sub, cls, mult in CFGS[cfg]:
        try:
            vae = getattr(diffusers, cls).from_pretrained(repo, subfolder=sub, torch_dtype=torch.float32)
            print(cfg, "->", repo)
            return vae, mult, repo
        except Exception as e:  # gated / missing -> fallback
            print("skip", repo, type(e).__name__)
    raise RuntimeError(f"no weights for {cfg}")


def recon(args):
    import torch
    d = dev()
    info = {}
    for cfg in CFGS:
        vae, mult, repo = load_vae(cfg)
        info[cfg] = repo
        vae = vae.to(d).eval()
        vae.enable_tiling()
        for ds in DATASETS:
            for p in images(args.data / ds):
                dst = args.out / "recon" / ds / cfg / f"{p.stem}.png"
                if dst.exists():
                    continue
                x = load(p)
                H, W = x.shape[:2]
                t = to_t(x, d).mul(2).sub(1)
                t = torch.nn.functional.pad(t, (0, (-W) % mult, 0, (-H) % mult), mode="reflect")
                with torch.no_grad():
                    z = vae.encode(t)
                    z = z.latent if cfg == "dcae" else z.latent_dist.mode()
                    y = vae.decode(z).sample
                y = y[..., :H, :W].clamp(-1, 1).add(1).mul(127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(y).save(dst)
            print(cfg, ds, "done")
        del vae
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(args.out / "weights_used.json", "w"), indent=1)


# ---------------------------------------------------------------- per-image metrics
def tiled(fn, a, b, d):
    H, W = a.shape[:2]
    tot = wsum = 0.0
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            pa, pb = a[y:y + TILE, x:x + TILE], b[y:y + TILE, x:x + TILE]
            if min(pa.shape[:2]) < 64:
                continue
            n = pa.shape[0] * pa.shape[1]
            tot += float(fn(to_t(pa, d), to_t(pb, d))) * n
            wsum += n
    return tot / wsum


def metrics(args):
    import lpips
    import piq
    import torch
    d = dev()
    lp = lpips.LPIPS(net="alex", verbose=False).to(d)
    dists = piq.DISTS().to(d)
    for ds in DATASETS:
        path = args.out / f"metrics_{ds}.csv"
        if path.exists():
            continue
        rows = []
        for p in images(args.data / ds):
            x = load(p)
            for cfg in CFGS:
                q = args.out / "recon" / ds / cfg / f"{p.stem}.png"
                if not q.exists():
                    continue
                y = load(q)
                mse = float(((x.astype(np.float64) - y) ** 2).mean())
                with torch.no_grad():
                    ms = float(piq.multi_scale_ssim(to_t(x, d), to_t(y, d), data_range=1.0)) \
                        if min(x.shape[:2]) >= 161 else float("nan")
                    rows.append(dict(
                        dataset=ds, name=p.stem, cfg=cfg, H=x.shape[0], W=x.shape[1], mse=mse,
                        psnr=10 * math.log10(255 ** 2 / max(mse, 1e-10)), msssim=ms,
                        lpips=tiled(lambda a, b: lp(a * 2 - 1, b * 2 - 1).mean(), x, y, d),
                        dists=tiled(lambda a, b: dists(a, b), x, y, d)))
        write_csv(rows, path)
        print(ds, len(rows), "rows")


# ---------------------------------------------------------------- FID / KID (HiFiC patch protocol)
def patches(a, size=256):
    H, W = a.shape[:2]
    for off in (0, size // 2):
        for y in range(off, H - size + 1, size):
            for x in range(off, W - size + 1, size):
                yield a[y:y + size, x:x + size]


def fid(args):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    d = dev()
    path = args.out / "fid.json"
    res = json.load(open(path)) if path.exists() else {}
    for ds in FID_SETS:
        for cfg in CFGS:
            key = f"{ds}/{cfg}"
            if key in res:
                continue
            f = FrechetInceptionDistance(feature=2048, normalize=False).to(d)
            k = KernelInceptionDistance(subset_size=1000, normalize=False).to(d)
            n = 0

            def push(batch, real):
                t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(d)  # uint8 NCHW
                f.update(t, real=real)
                k.update(t, real=real)

            for p in images(args.data / ds):
                q = args.out / "recon" / ds / cfg / f"{p.stem}.png"
                if not q.exists():
                    continue
                pr, pf = list(patches(load(p))), list(patches(load(q)))
                for i in range(0, len(pr), 64):
                    push(pr[i:i + 64], True)
                    push(pf[i:i + 64], False)
                n += len(pr)
            km, ks = k.compute()
            res[key] = dict(fid=float(f.compute()), kid=float(km), kid_std=float(ks), n_patches=n)
            print(key, res[key])
            json.dump(res, open(path, "w"), indent=1)


# ---------------------------------------------------------------- summary
def summary(args):
    fidr = json.load(open(args.out / "fid.json")) if (args.out / "fid.json").exists() else {}
    rows_out = []
    L = ["# Trần autoencoder (encode rồi decode, không lượng tử)", "",
         "PSNR/MS-SSIM/LPIPS/DISTS lấy trung bình theo ảnh. FID/KID tính trên patch 256 (lưới cộng lưới lệch 128, theo giao thức HiFiC).", "",
         "| dataset | AE | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | FID↓ | KID×10³↓ |", "|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        p = args.out / f"metrics_{ds}.csv"
        if not p.exists():
            continue
        rs = read_csv(p)
        for cfg in CFGS:
            r = [x for x in rs if x["cfg"] == cfg]
            if not r:
                continue
            m = {k: float(np.nanmean([float(x[k]) for x in r])) for k in ("psnr", "msssim", "lpips", "dists")}
            fr = fidr.get(f"{ds}/{cfg}", {})
            row = dict(dataset=ds, ae=cfg, n=len(r), **m, fid=fr.get("fid", float("nan")),
                       kid=fr.get("kid", float("nan")))
            rows_out.append(row)
            L.append(f"| {ds} | {cfg} | {m['psnr']:.2f} | {m['msssim']:.4f} | {m['lpips']:.4f} | {m['dists']:.4f} "
                     f"| {row['fid']:.2f} | {row['kid'] * 1e3:.2f} |")
    write_csv(rows_out, args.out / "ceiling_summary.csv")  # -> horizontal lines on the RD plots
    L += ["", "`ceiling_summary.csv` dùng để vẽ đường trần nằm ngang trên các RD plot.",
          "Khi đánh giá codec của mình và các baseline, **dùng lại đúng các hàm `tiled` và `fid` ở đây**, để số liệu so sánh được với nhau."]
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "recon", "metrics", "fid", "summary", "all"])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("ceiling_out"))
    args = ap.parse_args()
    for s in (["download", "recon", "metrics", "fid", "summary"] if args.step == "all" else [args.step]):
        print(f"===== {s}")
        globals()[s](args)


if __name__ == "__main__":
    main()
