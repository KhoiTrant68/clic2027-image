"""Rate vs Q-hat v0 / PSNR on the two validation images that have Cool-chic points (2684452d, a365e654).
Curves: merged server bake-off (results/2026-10-09_bakeoff_v0). Cool-chic points: cc_score.py on the server
(see summary.md). Run from the repo root:  uv run --no-project --with matplotlib python <this file>"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path("results/2026-10-09_coolchic2025_probe")
BAKE = Path("results/2026-10-09_bakeoff_v0")
QHAT = json.loads(Path("results/2026-10-06_qhat_v0/qhat_v0.json").read_text())
HEADER_BYTES = 4

PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "teal": "#42949E", "violet": "#9A4D8E",
           "neutral": "#9E9E9E"}
CURVES = [("msillm", "MS-ILLM", PALETTE["blue_main"], "-", "o"),
          ("codlite", "CoD-Lite", PALETTE["teal"], "-", "s"),
          ("vtmscc", "VTM-SCC", PALETTE["red_strong"], "-", "^"),
          ("mbt", "mbt2018 (MSE)", PALETTE["neutral"], "--", "v")]
# (image, label, bpp, Q-hat v0, PSNR, filled)
COOL = [("2684452d", "Cool-chic clic2025, λ 0.004", 0.5568, 1.33, 32.67, True),
        ("2684452d", "Cool-chic 5.0, 2000 vòng", 0.1114, -0.41, 29.11, False),
        ("a365e654", "Cool-chic, file Orange (WD)", 0.2345, -0.21, 24.37, True),
        ("a365e654", "Cool-chic, file Orange (MSE)", 0.2597, -2.24, 26.38, False)]
IMAGES = [("2684452d", "Ảnh 2684452d (tự nhiên)"), ("a365e654", "Ảnh a365e654 (tự nhiên)")]


def transform(name, v):  # experiments/qhat/qhat.py
    if name == "msssim":
        return -math.log10(max(1 - v, 1e-6))
    if name in ("lpips_alex", "lpips_vgg", "dists"):
        return -v
    return v


def qhat(r):
    return sum(w * transform(k, float(r[k])) for k, w in zip(QHAT["features"], QHAT["weights"]))


sizes = json.loads((BAKE / "budgets.json").read_text())["sizes"]
met = {(r["cand"], r["key"], r["name"]): r for r in csv.DictReader(open(BAKE / "metrics.csv"))}
pts = {}
for f in (BAKE / "points").glob("*.csv"):
    for r in csv.DictReader(open(f)):
        k = (r["cand"], r["key"], r["name"])
        if k in met:
            h, w = sizes[r["name"]]
            bpp = (int(r["bytes"]) - HEADER_BYTES) * 8 / (h * w)
            pts.setdefault((r["cand"], r["name"][:8]), []).append((bpp, qhat(met[k]), float(met[k]["psnr"])))

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 1.4, "legend.frameon": False,
                     "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
for col, (img, title) in enumerate(IMAGES):
    for row, (yi, ylabel) in enumerate([(1, "Q̂ v0 (cao hơn = người chấm thích hơn)"), (2, "PSNR (dB)")]):
        ax = axes[row, col]
        for cand, name, color, ls, mk in CURVES:
            c = sorted(p for p in pts.get((cand, img), []) if 0.02 <= p[0] <= 0.7)
            if c:
                ax.plot([p[0] for p in c], [p[yi] for p in c], ls=ls, marker=mk, color=color, lw=2, ms=6, label=name)
        for im, name, bpp, q, psnr, filled in COOL:
            if im == img:
                ax.scatter([bpp], [q if yi == 1 else psnr], marker="*", s=360, zorder=5, linewidths=1.6,
                           facecolors=PALETTE["violet"] if filled else "white", edgecolors=PALETTE["violet"],
                           label=name)
        ax.set_xscale("log")
        ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        ax.grid(alpha=0.25, lw=0.8)
        if row == 0:
            ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 1:
            ax.set_xlabel("bpp của ảnh này (thang log)")
fig.suptitle("Ở cùng bpp, Cool-chic có Q̂ v0 thấp hơn MS-ILLM và CoD-Lite trên cả hai ảnh",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
seen = {}
for ax in axes[0]:
    for h, lab in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(lab, h)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.05))
fig.text(0.01, 0.008, "Sao đặc: Cool-chic theo cấu hình cảm nhận (Wasserstein). Sao rỗng: bản thử ngắn hoặc bản tối ưu "
         "MSE. Đường: bake-off trên server L40, 09/10/2026.\nHàng dưới (PSNR) chỉ để tham khảo: PSNR một mình chỉ "
         "đoán đúng 57% lựa chọn của người chấm.", fontsize=9, color="#555555", linespacing=1.5)
fig.tight_layout(rect=(0, 0.115, 1, 0.96))
for ext in ("png", "svg"):
    fig.savefig(HERE / f"rd_coolchic.{ext}", dpi=200 if ext == "png" else None)
print("wrote", HERE / "rd_coolchic.png")
