"""Render saved codec comparisons, without inference or GPU use."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/.mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

DATA = ROOT / "results/2026-10-20_metrics_comparison"
RATES = (.075, .15, .3)
BG, INK, MUTED = "#FCFCFA", "#172B44", "#586577"
COLORS = {
    "vtm420": "#8B8F97", "vtmscc": "#637084", "mbt": "#A98C70",
    "turbo": "#BE719B", "msillm": "#2379B5", "codlite": "#DD8D26",
    "crdr": "#8A5BB1", "ditic": "#159C9C", "ditics": "#86B2A8",
    "pulse": "#509543", "mix": "#B35647", "expanded_mix": "#162E4E",
    "aeic_se": "#C78C2B", "aeic_me": "#D65E47",
}
LABELS = {
    "vtm420": "VTM 4:2:0", "vtmscc": "VTM-SCC", "mbt": "MBT2018",
    "turbo": "MS-ILLM + Turbo", "msillm": "MS-ILLM", "codlite": "CoD-Lite",
    "crdr": "CRDR", "ditic": "DiT-IC", "ditics": "DiT-IC student",
    "pulse": "PULSE-L", "mix": "Mix + PULSE-L", "expanded_mix": "Mix + PULSE-L + AEIC",
    "aeic_se": "AEIC-SE", "aeic_me": "AEIC-ME", "afp": "AFP-GIC",
    "dcvic": "DC-VIC", "pulse-l-perceptual": "PULSE-L",
    "baseline_mix": "Mix baseline",
}
ORDER = ["vtm420", "vtmscc", "mbt", "turbo", "msillm", "codlite", "crdr",
         "ditic", "ditics", "pulse", "mix", "expanded_mix"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#CCD3DB",
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "svg.hashsalt": "clic2027-metrics-20261020", "pdf.fonttype": 42,
})
sources, figures = {}, []


def read_csv(name, n):
    path = DATA / name
    sources[path] = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert int(row["n"]) == n
        for key in ("actual_bpp", "psnr_pooled", "msssim", "lpips_alex",
                    "lpips_vgg", "dists", "clipiqa", "qhat_v0"):
            row[key] = float(row[key])
            assert math.isfinite(row[key])
        row["budget_bpp"] = float(row["budget_bpp"]) if row["budget_bpp"] else None
        if row["budget_bpp"]:
            assert row["actual_bpp"] <= row["budget_bpp"]
    assert len({(r["method"], r["budget_bpp"]) for r in rows}) == len(rows)
    return rows


full = read_csv("full30_budget_metrics.csv", 30)
fixed = read_csv("aeic_fixed_metrics.csv", 30)
pilot = read_csv("pilot6_budget_metrics.csv", 6)
assert (len(full), len(fixed), len(pilot)) == (32, 2, 15)
weight_path = ROOT / "results/2026-10-15_new_codecs/qhat_v0.json"
sources[weight_path] = weight_path.read_bytes()
weights = json.loads(sources[weight_path])
floor_path = DATA / "rate_floors.json"
pilot_summary_path = ROOT / "results/2026-10-15_new_codecs/pilot_scores/summary.json"
s1_path = ROOT / "results/2026-10-06_sessiona/bakeoff/metrics.csv"
for path in (floor_path, pilot_summary_path, s1_path):
    sources[path] = path.read_bytes()
floors = json.loads(sources[floor_path])
pilot_summary = {(r["cand"], r["rate"]): r for r in json.loads(sources[pilot_summary_path])}
s1_rows = list(csv.DictReader(io.StringIO(sources[s1_path].decode("utf-8"))))
assert len({r["name"] for r in s1_rows if r["cand"] == "s1res"}) == 28
for rate in RATES:
    assert not pilot_summary[("pulse-perceptual", rate)]["feasible"]
all_q = [r["qhat_v0"] for r in full + pilot]
norm = Normalize(min(all_q), max(all_q))
cmap = plt.get_cmap("viridis")
pdf = PdfPages(OUT / "CLIC2027_metrics_charts.pdf",
               metadata={"Title": "CLIC2027 — So sánh metrics", "Author": "ratflow-codec"})


def frame(fig, title, subtitle, footer):
    fig.text(.04, .964, title, fontsize=22, weight="bold", va="top")
    fig.text(.04, .916, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.text(.04, .027, footer, fontsize=10, color=MUTED, va="bottom", linespacing=1.6)


def save(fig, name, inputs, description):
    fig.canvas.draw()
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    pdf.savefig(fig)
    figures.append({
        "name": name, "description": description,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "start": 1,
                     "end": len(sources[p].splitlines()),
                     "sha256": hashlib.sha256(sources[p].replace(b"\r\n", b"\n")).hexdigest()} for p in inputs],
    })
    plt.close(fig)


def heatmap(rows, order, names, title, subtitle, footer, filename, source, evidence):
    fig, ax = plt.subplots(figsize=(13, max(7.8, len(order) * .48 + 2.6)))
    fig.subplots_adjust(left=.28, right=.86, top=.84, bottom=.16)
    lookup = {(r["method"], r["budget_bpp"]): r for r in rows}
    data = np.full((len(order), 3), np.nan)
    for i, method in enumerate(order):
        for j, rate in enumerate(RATES):
            row = lookup.get((method, rate))
            if row:
                data[i, j] = row["qhat_v0"]
    cm = cmap.copy()
    cm.set_bad("#E8EBEE")
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cm, norm=norm, aspect="auto")
    for i, method in enumerate(order):
        for j, rate in enumerate(RATES):
            row = lookup.get((method, rate))
            if row:
                rgba = cmap(norm(row["qhat_v0"]))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                color = "#142238" if luminance > .57 else "white"
                ax.text(j, i-.12, f'{row["qhat_v0"]:+.4f}', ha="center", va="center",
                        fontsize=13, weight="bold", color=color)
                ax.text(j, i+.22, f'{row["actual_bpp"]:.4f} bpp', ha="center",
                        va="center", fontsize=9.5, color=color)
            else:
                if source.name == "full30_budget_metrics.csv":
                    assert floors[method]["minimum_corpus_bpp"] > rate
                else:
                    assert not pilot_summary[(method, rate)]["feasible"]
                ax.text(j, i, "Không vừa\nngân sách", ha="center", va="center",
                        color=MUTED, fontsize=10)
    ax.set(xticks=range(3), xticklabels=["≤ 0,075 bpp", "≤ 0,15 bpp", "≤ 0,3 bpp"],
           yticks=range(len(order)), yticklabels=[names[m] for m in order])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(image, ax=ax, fraction=.035, pad=.05)
    cb.set_label("Q̂ v0  •  cao hơn tốt hơn", labelpad=12)
    frame(fig, title, subtitle, footer)
    save(fig, filename, [source, *evidence], "Qhat cells and actual bpp read directly from CSV.")


heatmap(full, ORDER, LABELS,
        "01  /  Phương pháp nào tốt ở từng ngân sách?",
        "Cùng 30 ảnh • chữ lớn: Q̂ v0 • chữ nhỏ: bitrate thực • màu sáng hơn: Q̂ cao hơn",
        "Nguồn: full30_budget_metrics.csv • Q̂ là điểm đại diện, không phải điểm người chấm.\n"
        "S1 + residual chỉ đủ metric 28/30 ảnh nên chưa xếp hạng. AEIC đơn được vẽ riêng trong hình 02.",
        "01_qhat_overview", DATA / "full30_budget_metrics.csv", [floor_path, s1_path])

# Selected families keep this overview legible; every family remains in heatmaps and metric panels.
selected = ["vtmscc", "msillm", "codlite", "crdr", "ditic", "pulse",
            "expanded_mix", "aeic_me", "aeic_se"]
markers = ["s", "o", "^", "D", "v", "P", "*", "X", "h"]
fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
fig.subplots_adjust(left=.065, right=.965, top=.82, bottom=.31, wspace=.2)
for method, marker in zip(selected, markers):
    rows = sorted([r for r in full + fixed if r["method"] == method], key=lambda r: r["actual_bpp"])
    # Repeated allocations at larger budgets are the same measured point.
    unique = {(r["bytes"], r["qhat_v0"]): r for r in rows}
    rows = list(unique.values())
    for ax, metric in zip(axes, ("qhat_v0", "psnr_pooled")):
        ax.plot([r["actual_bpp"] for r in rows], [r[metric] for r in rows],
                marker=marker, color=COLORS[method], label=LABELS[method],
                markersize=10 if method == "expanded_mix" else 7,
                linewidth=2.5 if method == "expanded_mix" else 1.5,
                linestyle="-" if method == "expanded_mix" else "--")
for ax, title, ylabel in zip(axes, ["Chất lượng cảm nhận theo Q̂", "Độ trung thực pixel theo PSNR"],
                             ["Q̂ v0 ↑", "PSNR pooled (dB) ↑"]):
    ax.set(title=title, xlabel="Bitrate thực (bpp) →", ylabel=ylabel,
           xlim=(.018, .315), xticks=[.03, .075, .15, .225, .3])
    ax.grid(alpha=.18)
axes[0].axhline(0, color=MUTED, alpha=.35, linewidth=1)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .105),
           ncol=3, frameon=False, fontsize=11, columnspacing=3)
frame(fig, "02  /  Chất lượng theo bitrate thực: Q̂ và PSNR",
      "30 ảnh • một số họ tiêu biểu • AEIC-ME/SE là hai checkpoint cố định ở khoảng 0,03 bpp",
      "Điểm là số đo; đường nối chỉ hướng dẫn mắt, không phải nội suy đã kiểm chứng.\n"
      "DiT-IC dừng ở ~0,096 bpp trong các điểm đã thử. Không ngoại suy lên 0,15 hoặc 0,3 bpp.")
save(fig, "02_rate_quality", [DATA / "full30_budget_metrics.csv", DATA / "aeic_fixed_metrics.csv"],
     "Measured actual bitrate versus Qhat and pooled PSNR; selected methods only.")

metric_specs = [
    ("psnr_pooled", "PSNR pooled (dB) ↑", 2), ("msssim", "MS-SSIM ↑", 4),
    ("lpips_alex", "LPIPS-Alex ↓", 4), ("lpips_vgg", "LPIPS-VGG ↓", 4),
    ("dists", "DISTS ↓", 4), ("clipiqa", "CLIP-IQA ↑", 4),
]
for index, rate in enumerate(RATES, 3):
    rows = sorted([r for r in full if r["budget_bpp"] == rate],
                  key=lambda r: r["qhat_v0"], reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5), sharey=True)
    fig.subplots_adjust(left=.22, right=.94, top=.83, bottom=.13, hspace=.28, wspace=.33)
    y = np.arange(len(rows))
    for ax, (metric, title, precision) in zip(axes.flat, metric_specs):
        values = [r[metric] for r in rows]
        lo, hi = min(values), max(values)
        span = max(hi-lo, .01)
        ax.set_xlim(lo - .1*span, hi + .40*span)
        ax.set_title(title, loc="left", pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.18)
        for i in range(len(rows)):
            if i % 2 == 0:
                ax.axhspan(i-.48, i+.48, color="#EFF2F4", zorder=0)
        ax.scatter(values, y, c=[COLORS[r["method"]] for r in rows], s=50, zorder=3)
        for value, yi in zip(values, y):
            ax.annotate(f"{value:.{precision}f}", (value, yi), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(y, [f'{LABELS[r["method"]]}  |  {r["actual_bpp"]:.4f}' for r in rows])
        ax.set_ylim(len(rows)-.5, -.5)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.locator_params(axis="x", nbins=4)
    frame(fig, f"0{index}  /  Sáu metric tại ngân sách {rate:g} bpp",
          "30 ảnh • hàng sắp theo Q̂ giảm dần • số sau tên phương pháp là bpp thực • mỗi panel giữ đơn vị gốc",
          "↑ cao hơn tốt hơn   /   ↓ thấp hơn tốt hơn • Trục của dot plot được thu gọn để thấy khác biệt.\n"
          "Các codec không vừa ngân sách không xuất hiện. Điểm sát nhau chưa chứng minh tương đương thống kê.")
    save(fig, f"0{index}_metrics_{str(rate).replace('.', '')}",
         [DATA / "full30_budget_metrics.csv"], f"Six raw metrics for all available allocations at budget {rate}.")

pilot_labels = dict(LABELS, expanded_mix="Mix + AFP/DC/PULSE(-L)", baseline_mix="Mix baseline pilot")
heatmap(pilot, ["afp", "dcvic", "pulse-l-perceptual", "msillm", "baseline_mix", "expanded_mix"],
        pilot_labels, "06  /  Pilot 6 ảnh: đọc trong đúng nhóm",
        "Chữ lớn: Q̂ v0 • chữ nhỏ: bitrate thực • giữ cùng thang màu với hình 01",
        "Nguồn: pilot6_budget_metrics.csv • Không xếp hạng trực tiếp với kết quả 30 ảnh.\n"
        "PULSE base đã thử nhưng không vừa cả ba ngân sách. Mix baseline pilot chưa có PULSE-L.",
        "06_pilot6", DATA / "pilot6_budget_metrics.csv", [pilot_summary_path])

fig, ax = plt.subplots(figsize=(13, 8))
fig.subplots_adjust(left=.10, right=.94, top=.81, bottom=.21)
x = np.linspace(-3, 3, 401)
sigmoid = lambda value: 1 / (1 + np.exp(-value))
bias = weights["bias"]
ax.plot(x, 100*sigmoid(x), color=COLORS["ditic"], linewidth=3, label="Bỏ bias vị trí: sigmoid(ΔQ̂)")
ax.plot(x, 100*sigmoid(x+bias), color=MUTED, linewidth=1.5, linestyle="--",
        label=f"Có bias fit {bias:.5f}: sigmoid(bias + ΔQ̂)")
for value, offset in [(-2, (0, -25)), (-1, (0, -25)), (0, (-36, 10)), (.5, (12, -25)),
                      (1, (0, 15)), (2, (0, 15))]:
    prob = float(100*sigmoid(value))
    ax.scatter(value, prob, s=60, color=COLORS["ditic"], zorder=3)
    ax.annotate(f"Δ={value:+g} → {prob:.1f}%", (value, prob), xytext=offset,
                textcoords="offset points", ha="center", fontsize=11,
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 2})
ax.axhline(50, color=MUTED, linewidth=1, alpha=.3)
ax.axvline(0, color=MUTED, linewidth=1, alpha=.3)
ax.set(xlim=(-3, 3), ylim=(0, 100), xticks=np.arange(-3, 3.5, .5),
       xlabel="ΔQ̂ = Q̂(A) − Q̂(B), trên cùng một ảnh gốc",
       ylabel="Xác suất mô hình chọn A (%)")
ax.grid(alpha=.15)
ax.legend(loc="lower right", frameon=True, facecolor=BG, fontsize=10)
frame(fig, "07  /  Q̂ không phải thang điểm 0–100",
      "Minh họa công thức preference • điểm Q̂ có thể âm • chỉ chênh lệch của cùng cặp ảnh được diễn giải ở đây",
      "Đường cong là xác suất của mô hình, không phải tỷ lệ thắng đã được người chấm xác nhận.\n"
      "Không dùng sigmoid(ΔQ̂ trung bình toàn tập) để suy ra tỷ lệ thắng của codec.")
save(fig, "07_qhat_scale", [weight_path], "Analytical sigmoid illustration, not empirical preference measurements.")
pdf.close()
(OUT / "manifest.json").write_text(json.dumps({
    "date": "2026-10-20", "matplotlib": matplotlib.__version__,
    "hash_format": "SHA-256 of text bytes with CRLF normalized to LF; BOM is preserved",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    "figures": figures,
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Generated {len(figures)} figures (PNG + SVG), PDF and source manifest in {OUT}")
