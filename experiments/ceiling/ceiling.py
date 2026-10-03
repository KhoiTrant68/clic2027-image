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
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo checkout without `pip install -e .`
from ratflow.eval.metrics import (  # noqa: E402
    Perceptual, device, fid_kid, images, load, read_csv, save, to_tensor, write_csv,
)

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
    d = device()
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
                t = to_tensor(x, d).mul(2).sub(1)
                t = torch.nn.functional.pad(t, (0, (-W) % mult, 0, (-H) % mult), mode="reflect")
                with torch.no_grad():
                    z = vae.encode(t)
                    z = z.latent if cfg == "dcae" else z.latent_dist.mode()
                    y = vae.decode(z).sample
                y = y[..., :H, :W].clamp(-1, 1).add(1).mul(127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                save(y, dst)
            print(cfg, ds, "done")
        del vae
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(args.out / "weights_used.json", "w"), indent=1)


# ---------------------------------------------------------------- per-image metrics
def metrics(args):
    perc = Perceptual(device())
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
                rows.append(dict(dataset=ds, name=p.stem, cfg=cfg, H=x.shape[0], W=x.shape[1],
                                 **perc.all(x, load(q))))
        write_csv(rows, path)
        print(ds, len(rows), "rows")


# ---------------------------------------------------------------- FID / KID (HiFiC patch protocol)
def fid(args):
    path = args.out / "fid.json"
    res = json.load(open(path)) if path.exists() else {}
    for ds in FID_SETS:
        for cfg in CFGS:
            key = f"{ds}/{cfg}"
            if key in res:
                continue
            pairs = ((load(p), load(q)) for p in images(args.data / ds)
                     if (q := args.out / "recon" / ds / cfg / f"{p.stem}.png").exists())
            res[key] = fid_kid(pairs, device())
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
          "Khi đánh giá codec của mình và các baseline, **dùng lại đúng `ratflow.eval.metrics`** (`Perceptual.all`, `fid_kid`), để số liệu so sánh được với nhau."]
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
