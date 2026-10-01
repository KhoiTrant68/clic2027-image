"""End-to-end S1 evaluation with real bitstreams: image -> DC-AE -> S1 bytes -> S1 decode -> DC-AE -> image.

bpp from len(bytes) (mean over images); latent MSE; PSNR pooled over pixels (CLIC style); PSNR / MS-SSIM / LPIPS / DISTS via ratflow.eval.metrics (same code as the
ceilings). Also reports the DC-AE ceiling of each image, so the S1 points can be read against it.

    python eval_s1.py --ckpt s1_out/last.pt --dataset kodak --out s1_eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ratflow.codec.latent_codec import LatentCodec  # noqa: E402
from ratflow.eval.metrics import Perceptual, device, images, load, pooled_psnr, save, write_csv  # noqa: E402
from ratflow.nn.dcae import DCAE  # noqa: E402

AE_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


def kodak(root: Path) -> Path:
    d = root / "kodak"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, 25):
        f = d / f"kodim{i:02d}.png"
        if not f.exists():
            urllib.request.urlretrieve(f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="s1_out/last.pt")
    ap.add_argument("--dataset", default="kodak", help="'kodak' or a directory of images")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("s1_eval"))
    ap.add_argument("--rates", type=int, nargs="*", default=None)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    dev = device()
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = LatentCodec(**{k: v for k, v in s["config"].items()})
    m.load_state_dict(s["model"])
    m = m.to(dev).eval()
    dec = LatentCodec(**s["config"])  # decoder side: separate object, CPU entropy path, same checkpoint
    dec.load_state_dict(s["model"])
    dec = dec.to(dev).eval()
    scaling = s.get("scaling", 0.41407)
    ae = DCAE.from_pretrained(AE_REPO, "vae").to(dev).enable_tiling()
    perc = Perceptual(dev)
    files = images(kodak(args.data) if args.dataset == "kodak" else Path(args.dataset))
    rates = args.rates if args.rates is not None else list(range(m.n_rates))
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for p in files:
        x = load(p)
        H, W = x.shape[:2]
        H32, W32 = H - H % 32, W - W % 32
        x = x[:H32, :W32]
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        with torch.no_grad():
            lat = ae.encode(t) * scaling
            ceil = ae.decode(lat / scaling)
        to_img = lambda y: y[0].clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()  # noqa: E731
        rows.append(dict(name=p.stem, cfg="dcae_ceiling", px=H32 * W32, bpp=float("nan"), lat_mse=0.0,
                         **perc.all(x, to_img(ceil))))
        for r in rates:
            t0 = time.time()
            blob = m.compress(lat, r, seed=r)
            t_enc = time.time() - t0
            t0 = time.time()
            lat_hat = dec.decompress(blob, device=dev)
            t_dec = time.time() - t0
            with torch.no_grad():
                img = to_img(ae.decode(lat_hat / scaling))
            row = dict(name=p.stem, cfg=f"s1_r{r}", rate=r, px=H32 * W32, bytes=len(blob), bpp=len(blob) * 8 / (H32 * W32),
                       lat_mse=float((lat_hat - lat).pow(2).mean()), t_enc=t_enc, t_dec_s1=t_dec, **perc.all(x, img))
            rows.append(row)
            if args.save_images:
                save(img, args.out / "img" / f"r{r}" / f"{p.stem}.png")
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    write_csv(rows, args.out / "per_image.csv")

    L = [f"# S1 trên {args.dataset} (bitstream thật)", "", f"checkpoint: `{args.ckpt}` (step {s.get('step')})", "",
         "| cfg | bpp | latent MSE | PSNR (pooled) | MS-SSIM | LPIPS↓ | DISTS↓ |", "|---|---|---|---|---|---|---|"]
    summ = {}
    for cfg in ["dcae_ceiling"] + [f"s1_r{r}" for r in rates]:
        rs = [r for r in rows if r["cfg"] == cfg]
        px = [r["px"] for r in rs]
        mean = lambda k: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
        summ[cfg] = dict(bpp=mean("bpp"), lat_mse=mean("lat_mse"), psnr=pooled_psnr([r["mse"] for r in rs], px),
                         msssim=mean("msssim"), lpips=mean("lpips"), dists=mean("dists"))
        v = summ[cfg]
        L.append(f"| {cfg} | {v['bpp']:.4f} | {v['lat_mse']:.4f} | {v['psnr']:.2f} | {v['msssim']:.4f} | "
                 f"{v['lpips']:.4f} | {v['dists']:.4f} |")
    L += ["", "S1 được train bằng MSE trên latent (latent_hat ≈ E[latent | bitstream]), nên ảnh mờ là đúng thiết kế; "
          "bridge S2/S3 mới là phần khôi phục độ chân thực. Dòng `dcae_ceiling` là trần của backbone."]
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1))
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
