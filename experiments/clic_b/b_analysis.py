"""Branch B: where does a DC-AE-based codec break on the CLIC validation set?

Steps (each writes into --out):
  inventory  per-image size, byte budgets, colour/flatness/edge stats, OCR text boxes,
             suggested label natural/screen          -> inventory.csv, budgets.json
  recon      reconstructions: DC-AE f32c32 ceiling, SD-VAE f8 ceiling,
             HEVC-intra proxy (x265, ~HM baseline) at 0.075/0.15/0.3 bpp -> recon/<cfg>/<name>.png
  metrics    PSNR, MS-SSIM, LPIPS, DISTS (tiled), text-region PSNR, OCR CER -> metrics.csv
  crops      256x256 crops around the main text box: orig | each cfg     -> crops/<name>.png
  summary    per-label means, CLIC-style dataset PSNR, B3 decision table   -> summary.md

python b_analysis.py all --data data/valid --out b_out
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo checkout without `pip install -e .`
from ratflow.eval.metrics import (  # noqa: E402  (load/save/... re-exported for b6_vtm_residual.py)
    Perceptual, box_xyxy, cer, device, images, load, ocr_reader, read_boxes, read_csv, region_psnr, save, write_csv,
)

DATA_URL = "https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip"
RATES = [0.075, 0.15, 0.3]
DCAE_ID = ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "vae")
SDVAE_IDS = [("stabilityai/sd-turbo", "vae"), ("stabilityai/sd-vae-ft-ema", None)]  # first = StableCodec's VAE
get_reader = ocr_reader


# ---------------------------------------------------------------- download
def download(data: Path):
    if data.exists() and images(data):
        return
    data.mkdir(parents=True, exist_ok=True)
    z = data / "valid.zip"
    subprocess.run(["wget", "-q", "-O", str(z), DATA_URL], check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(data)
    z.unlink()
    print("downloaded", len(images(data)), "images")


# ---------------------------------------------------------------- inventory
def inventory(args):
    download(args.data)
    reader = get_reader()
    rows, boxes = [], {}
    for p in images(args.data):
        x = load(p)
        H, W = x.shape[:2]
        rgb = x.reshape(-1, 3).astype(np.uint32)
        n_colors = len(np.unique((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]))
        g = x.astype(np.float32).mean(2)
        hb, wb = H // 8 * 8, W // 8 * 8
        blocks = g[:hb, :wb].reshape(hb // 8, 8, wb // 8, 8).std(axis=(1, 3))
        flat = float((blocks < 1.0).mean())
        grad = np.abs(np.diff(g, axis=1))[:-1] + np.abs(np.diff(g, axis=0))[:, :-1]
        sharp = float((grad > 96).mean())
        det = [(b, t, c) for b, t, c in reader.readtext(x) if c > 0.4 and len(t.strip()) >= 2]
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in (box_xyxy(b, H, W, 0) for b, _, _ in det))
        text_frac = area / (H * W)
        screenish = (n_colors / (H * W) < 0.05 and flat > 0.3) or text_frac > 0.03
        rows.append(dict(name=p.stem, H=H, W=W, pixels=H * W, n_colors=n_colors,
                         colors_per_px=round(n_colors / (H * W), 4), flat_frac=round(flat, 3),
                         sharp_edge_frac=round(sharp, 4), n_text_boxes=len(det),
                         text_area_frac=round(text_frac, 4),
                         label_suggested="screen" if screenish else "natural",
                         label="",  # <- fill in by hand: natural / game / screen
                         ))
        boxes[p.stem] = [[box_xyxy(b, H, W), t] for b, t, _ in det]
        print(rows[-1])
    total = sum(r["pixels"] for r in rows)
    budgets = {str(r): math.floor(r * total / 8) for r in RATES}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "inventory.csv")
    json.dump(dict(total_pixels=total, budgets_bytes=budgets, boxes=boxes),
              open(args.out / "budgets.json", "w"), indent=1)
    print("total pixels", total, "budgets (bytes)", budgets)


# ---------------------------------------------------------------- reconstructions
def ae_roundtrip(args, cfg):
    import torch
    from diffusers import AutoencoderDC, AutoencoderKL

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg == "dcae":
        vae = AutoencoderDC.from_pretrained(DCAE_ID[0], subfolder=DCAE_ID[1], torch_dtype=torch.float32)
        mult = 32
    else:
        vae = None
        for rid, sub in SDVAE_IDS:
            try:
                vae = AutoencoderKL.from_pretrained(rid, subfolder=sub, torch_dtype=torch.float32)
                print("SD-VAE:", rid)
                break
            except Exception as e:  # gated / missing -> next
                print("skip", rid, type(e).__name__)
        mult = 8
    vae = vae.to(dev).eval()
    vae.enable_tiling()
    for p in images(args.data):
        dst = args.out / "recon" / cfg / f"{p.stem}.png"
        if dst.exists():
            continue
        x = load(p)
        H, W = x.shape[:2]
        t = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        ph, pw = (-H) % mult, (-W) % mult
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            if cfg == "dcae":
                y = vae.decode(vae.encode(t).latent).sample
            else:
                y = vae.decode(vae.encode(t).latent_dist.mode()).sample
        y = y[..., :H, :W].clamp(-1, 1).add(1).mul(127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        save(y, dst)
        print(cfg, p.stem)
    del vae
    torch.cuda.empty_cache()


def hevc_at(src: Path, target_bytes: int, dst: Path):
    """x265 intra, 4:4:4, fixed QP; smallest QP whose bitstream fits the budget.
    Proxy for the HM baseline (VTM is ~0.7 dB better at 0.075 bpp on the leaderboard)."""
    with tempfile.TemporaryDirectory() as td:
        bs = Path(td) / "o.hevc"

        def enc(qp):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-frames:v", "1",
                            "-c:v", "libx265", "-pix_fmt", "yuv444p",
                            "-x265-params", f"qp={qp}:keyint=1:log-level=error", "-f", "hevc", str(bs)],
                           check=True)
            return bs.stat().st_size

        lo, hi, best = 0, 51, None
        while lo <= hi:
            qp = (lo + hi) // 2
            if enc(qp) <= target_bytes:
                best, hi = qp, qp - 1
            else:
                lo = qp + 1
        if best is None:
            best = 51
        size = enc(best)
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(bs), "-pix_fmt", "rgb24", str(dst)],
                       check=True)
    return best, size


def recon(args):
    for cfg in ["dcae", "sdvae"]:
        ae_roundtrip(args, cfg)
    if shutil.which("ffmpeg") is None or "libx265" not in subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout:
        print("WARNING: ffmpeg/libx265 missing -> no HEVC proxy")
        return
    log = []
    for p in images(args.data):
        x = load(p)
        H, W = x.shape[:2]
        for r in RATES:
            dst = args.out / "recon" / f"hevc@{r}" / f"{p.stem}.png"
            qp, size = hevc_at(p, math.floor(r * H * W / 8), dst)
            log.append(dict(name=p.stem, rate=r, qp=qp, bytes=size, bpp=round(size * 8 / (H * W), 4)))
            print(log[-1])
    write_csv(log, args.out / "hevc_log.csv")


# ---------------------------------------------------------------- metrics
def configs(out: Path):
    return sorted(d.name for d in (out / "recon").iterdir() if d.is_dir())


def metrics(args):
    perc = Perceptual(device())
    reader = get_reader()
    boxes = json.load(open(args.out / "budgets.json"))["boxes"]
    rows = []
    for p in images(args.data):
        x = load(p)
        bx = [b for b, _ in boxes.get(p.stem, [])]
        # OCR of the original, per box (same crop protocol as for the reconstructions)
        ref_txt = read_boxes(reader, x, bx)
        for cfg in configs(args.out):
            q = args.out / "recon" / cfg / f"{p.stem}.png"
            if not q.exists():
                continue
            y = load(q)
            row = dict(name=p.stem, cfg=cfg, **perc.all(x, y))
            if bx:
                row["text_psnr"] = region_psnr(x, y, bx)
                n_ch = sum(len(r) for r in ref_txt)
                if n_ch:
                    row["ocr_cer"] = cer(ref_txt, read_boxes(reader, y, bx))
                    row["ocr_chars"] = n_ch
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    write_csv(rows, args.out / "metrics.csv")


# ---------------------------------------------------------------- crops
def crops(args, size=256):
    boxes = json.load(open(args.out / "budgets.json"))["boxes"]
    cfgs = configs(args.out)
    for p in images(args.data):
        x = load(p)
        H, W = x.shape[:2]
        bx = boxes.get(p.stem, [])
        if bx:
            x0, y0, x1, y1 = max(bx, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))[0]
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        else:
            cy, cx = H // 2, W // 2
        top = int(np.clip(cy - size // 2, 0, max(0, H - size)))
        left = int(np.clip(cx - size // 2, 0, max(0, W - size)))
        panels = [("orig", x)] + [(c, load(args.out / "recon" / c / f"{p.stem}.png")) for c in cfgs
                                  if (args.out / "recon" / c / f"{p.stem}.png").exists()]
        canvas = Image.new("RGB", (size * len(panels), size + 18), "white")
        d = ImageDraw.Draw(canvas)
        for i, (lab, a) in enumerate(panels):
            canvas.paste(Image.fromarray(a[top:top + size, left:left + size]), (i * size, 18))
            d.text((i * size + 4, 2), lab, fill="black")
        (args.out / "crops").mkdir(parents=True, exist_ok=True)
        canvas.save(args.out / "crops" / f"{p.stem}.png")


# ---------------------------------------------------------------- summary
def summary(args):
    inv = {r["name"]: r for r in read_csv(args.out / "inventory.csv")}
    rows = read_csv(args.out / "metrics.csv")
    lab = lambda n: inv[n]["label"] or inv[n]["label_suggested"]  # noqa: E731
    cfgs = sorted({r["cfg"] for r in rows})
    groups = ["all"] + sorted({lab(n) for n in inv})
    L = ["# Nhánh B: kết quả", ""]

    # CLIC-style dataset PSNR (MSE pooled, weighted by pixels)
    L += ["## PSNR toàn tập theo cách CLIC (MSE gộp, trọng số theo số pixel)", "",
          "Mốc so sánh trên leaderboard @0.075: HM 25.83, VTM 26.52, Vcoder (hạng 1) 23.99", "",
          "| cfg | PSNR |", "|---|---|"]
    for c in cfgs:
        rs = [r for r in rows if r["cfg"] == c]
        px = [int(inv[r["name"]]["pixels"]) for r in rs]
        mse = sum(float(r["mse"]) * n for r, n in zip(rs, px)) / sum(px)
        L.append(f"| {c} | {10 * math.log10(255 ** 2 / mse):.2f} |")

    def mean(rs, k):
        v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
        return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)

    for g in groups:
        L += ["", f"## Nhóm: {g}", "", "| cfg | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ (n ảnh) |",
              "|---|---|---|---|---|---|---|"]
        for c in cfgs:
            rs = [r for r in rows if r["cfg"] == c and (g == "all" or lab(r["name"]) == g)]
            if not rs:
                continue
            cer, n = mean(rs, "ocr_cer")
            L.append(f"| {c} | {mean(rs, 'psnr')[0]:.2f} | {mean(rs, 'msssim')[0]:.4f} | {mean(rs, 'lpips')[0]:.4f} "
                     f"| {mean(rs, 'dists')[0]:.4f} | {mean(rs, 'text_psnr')[0]:.2f} | {cer:.3f} ({n}) |")

    # B3 decision helpers
    def m(cfg, k, grp):
        rs = [r for r in rows if r["cfg"] == cfg and lab(r["name"]) in grp]
        return mean(rs, k)[0]

    scr = {"screen", "game"}
    VTM_OFFSET = 1.9  # VTM (26.52) - our x265 proxy (24.58) at 0.075, dataset PSNR
    L += ["", "## Kiểm tra điều kiện B3 (HEVC dùng thay VTM, cộng thêm 1.9 dB)", ""]
    if "hevc@0.075" in cfgs:
        c1 = m("dcae", "ocr_cer", scr) <= m("hevc@0.075", "ocr_cer", scr)
        c2 = m("dcae", "psnr", scr) >= m("hevc@0.3", "psnr", scr) + VTM_OFFSET - 1
        c3 = m("sdvae", "ocr_cer", scr) <= m("hevc@0.075", "ocr_cer", scr)
        c4 = m("dcae", "psnr", {"natural"}) < m("hevc@0.3", "psnr", {"natural"}) + VTM_OFFSET - 2
        L += [f"- [screen/game] CER trần của DC-AE ≤ CER HEVC@0.075: **{c1}**",
              f"- [screen/game] PSNR trần của DC-AE ≥ VTM@0.3 − 1 dB: **{c2}**",
              f"- [screen/game] CER trần của f8 ≤ CER HEVC@0.075: **{c3}**",
              f"- [natural] PSNR trần của DC-AE < VTM@0.3 − 2 dB (cần residual ở rate cao): **{c4}**", ""]
        if c1 and c2:
            L.append("→ **Không cần nhánh riêng**; chỉ cần phân bổ nhiều bit hơn cho ảnh có chữ (A3).")
        elif c3:
            L.append("→ **Thêm nhánh residual / latent f8** cho ảnh có chữ và ảnh màn hình.")
        else:
            L.append("→ **Thêm mode SCC cổ điển** (VVC) cho ảnh màn hình.")
        if c4:
            L.append("→ **Ảnh tự nhiên ở 0.15–0.3 bpp cần đường residual**.")
    L += ["", "*Chú ý: `label` trong inventory.csv cần được xác nhận bằng mắt. Nhãn hiện tại có thể chỉ là nhãn gợi ý tự động.*"]
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["inventory", "recon", "metrics", "crops", "summary", "all"])
    ap.add_argument("--data", type=Path, default=Path("data/valid"))
    ap.add_argument("--out", type=Path, default=Path("b_out"))
    args = ap.parse_args()
    steps = ["inventory", "recon", "metrics", "crops", "summary"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"===== {s}")
        globals()[s](args)


if __name__ == "__main__":
    main()
