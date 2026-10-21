"""Build traceable comparison tables from saved metrics; no GPU or new inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/bakeoff"))
from bakeoff import qhat_score

Q = json.loads((ROOT / "results/2026-10-15_new_codecs/qhat_v0.json").read_text())
BASE = ROOT / "results/2026-10-15_pulse_integration/integrated_bakeoff"
AEIC = ROOT / "results/2026-10-20_aeic_full30/full30_20261020"
BUDGETS = json.loads((BASE / "budgets.json").read_text())
SIZES = BUDGETS["sizes"]
FEATURES = ["msssim", "lpips_alex", "lpips_vgg", "dists", "clipiqa"]
NAMES = {
    "vtm420": "VTM 4:2:0", "vtmscc": "VTM-SCC 4:4:4", "s1res": "DC-AE + S1 + residual VTM",
    "mbt": "MBT2018-mean", "msillm": "MS-ILLM", "codlite": "CoD-Lite",
    "ditic": "DiT-IC", "ditics": "DiT-IC student", "crdr": "CRDR",
    "turbo": "MS-ILLM + SD-Turbo refiner", "pulse": "PULSE-L perceptual",
    "mix": "Mix có PULSE-L", "expanded_mix": "Mix có PULSE-L + AEIC",
    "aeic_se": "AEIC-SE ft2", "aeic_me": "AEIC-ME ft2",
    "afp": "AFP-GIC", "dcvic": "DC-VIC PatchGAN",
    "pulse-perceptual": "PULSE perceptual", "pulse-l-perceptual": "PULSE-L pilot",
    "baseline_mix": "Mix baseline pilot",
}
inputs = set()


def read(p):
    inputs.add(p)
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_map(p):
    return {(r["cand"], r["key"], r["name"]): r for r in read(p)}


def aggregate(label, rate, selection, metrics, source, expected_n):
    assert len(selection) == expected_n
    names = {r["name"] for r in selection}
    assert len(names) == expected_n and names <= set(SIZES)
    pixels, mses, scored, feats = [], [], [], []
    for r in selection:
        cand, key = (r["point_cand"], r["key"]) if "point_cand" in r else (
            r["key"].split("/", 1) if "/" in r["key"] else (r["cand"], r["key"]))
        m = metrics[(cand, key, r["name"])]
        px = math.prod(SIZES[r["name"]])
        pixels.append(px)
        mses.append(float(m["mse"]) if m.get("mse") else 255 ** 2 * 10 ** (-float(m["psnr"]) / 10))
        score = qhat_score(Q, m)
        assert abs(score - float(r["score"])) < 1e-7
        scored.append(score)
        feats.append(m)
    total_bytes = sum(int(r["bytes"]) for r in selection)
    if rate is not None:
        assert total_bytes <= math.floor(rate * sum(pixels) / 8)
    return dict(method=label, display=NAMES[label], budget_bpp=rate, n=expected_n,
                bytes=total_bytes, actual_bpp=8 * total_bytes / sum(pixels),
                psnr_pooled=10 * math.log10(255 ** 2 / (sum(m * p for m, p in zip(mses, pixels)) / sum(pixels))),
                **{k: statistics.mean(float(m[k]) for m in feats) for k in FEATURES},
                qhat_v0=statistics.mean(scored), source=source.relative_to(ROOT).as_posix())


def from_saved(path, cands):
    met = metric_map(path / "metrics.csv")
    alloc = read(path / "allocation.csv")
    result = []
    inputs.add(path / "summary.json")
    summary = json.loads((path / "summary.json").read_text())
    for cand in cands:
        for rate in (.075, .15, .3):
            selection = [r for r in alloc if r["cand"] == cand and float(r["rate"]) == rate]
            if not selection:
                continue
            row = aggregate(cand, rate, selection, met, path / "allocation.csv", 30)
            saved = summary[f"{cand}@{rate}/all"]
            assert abs(row["qhat_v0"] - saved["qhat"]) < 1e-7
            assert abs(row["psnr_pooled"] - saved["psnr"]) < 1e-7
            result.append(row)
    return result


rows = from_saved(BASE, ["vtmscc", "msillm", "codlite", "ditic", "ditics", "pulse", "mix"])
rows += from_saved(ROOT / "results/2026-10-13_bakeoff_v0", ["vtm420", "mbt"])
rows += from_saved(ROOT / "results/2026-10-10_bakeoff_v0", ["crdr"])
rows += from_saved(ROOT / "results/2026-10-08_bakeoff_merged", ["turbo"])
floors = {}
for cand, path, missing_rates in [
    ("mbt", ROOT / "results/2026-10-13_bakeoff_v0", [.075]),
    ("crdr", ROOT / "results/2026-10-10_bakeoff_v0", [.075]),
    ("pulse", BASE, [.075, .15]),
]:
    pts = read(path / f"points/{cand}.csv")
    minimum = {n: min(int(r["bytes"]) for r in pts if r["name"] == n) for n in SIZES}
    total = sum(minimum.values())
    for rate in missing_rates:
        assert total > BUDGETS["budgets_bytes"][str(rate)]
    floors[cand] = dict(minimum_bytes=total, minimum_corpus_bpp=8 * total / BUDGETS["total_pixels"])

# Historical S1 has points for 30 images but metrics for only 28; do not rank it as full30.
s1 = ROOT / "results/2026-10-06_sessiona/bakeoff"
s1m = metric_map(s1 / "metrics.csv")
s1pts = read(s1 / "points/s1res.csv")
options = {n: [] for n in sorted(SIZES)}
for r in s1pts:
    key = ("s1res", r["key"], r["name"])
    if key in s1m:
        options[r["name"]].append((int(r["bytes"]), qhat_score(Q, s1m[key]), r["key"]))
s1_coverage = sum(bool(v) for v in options.values())

base_metrics = metric_map(BASE / "metrics.csv")
aeic_metrics = metric_map(AEIC / "scores/metrics.csv")
joint = {**base_metrics, **aeic_metrics}
aeic_alloc = read(AEIC / "scores/allocation.csv")
for rate in (.075, .15, .3):
    selection = [r for r in aeic_alloc if r["cand"] == "expanded_mix" and float(r["rate"]) == rate]
    rows.append(aggregate("expanded_mix", rate, selection, joint, AEIC / "scores/allocation.csv", 30))
fixed = []
for cand in ("aeic_se", "aeic_me"):
    selection = [r for r in aeic_alloc if r["cand"] == cand and float(r["rate"]) == .075]
    fixed.append(aggregate(cand, None, selection, aeic_metrics, AEIC / "scores/allocation.csv", 30))

pilot = ROOT / "results/2026-10-15_new_codecs/pilot_scores"
pm = {**metric_map(ROOT / "results/2026-10-13_bakeoff_v0/metrics.csv"), **metric_map(pilot / "metrics.csv")}
pa = read(pilot / "allocation.csv")
pilot_rows = []
for cand in ("afp", "dcvic", "pulse-l-perceptual", "msillm", "baseline_mix", "expanded_mix"):
    for rate in (.075, .15, .3):
        selection = [r for r in pa if r["cand"] == cand and float(r["rate"]) == rate]
        if selection:
            row = aggregate(cand, rate, selection, pm, pilot / "allocation.csv", 6)
            if cand == "expanded_mix":
                row["display"] = "Mix pilot + AFP/DC-VIC/PULSE/PULSE-L"
            pilot_rows.append(row)


def write_csv(p, data):
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


write_csv(OUT / "full30_budget_metrics.csv", rows)
write_csv(OUT / "aeic_fixed_metrics.csv", fixed)
write_csv(OUT / "pilot6_budget_metrics.csv", pilot_rows)
(OUT / "rate_floors.json").write_text(json.dumps(floors, indent=2))
inputs.update([BASE / "budgets.json", ROOT / "results/2026-10-15_new_codecs/qhat_v0.json",
               ROOT / "experiments/qhat/qhat.py", ROOT / "experiments/bakeoff/bakeoff.py"])
(OUT / "sources.sha256.json").write_text(json.dumps(
    # Git stores text with LF; normalize CRLF for portable source verification.
    {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
     for p in sorted(inputs)}, indent=2))

order = ["vtm420", "vtmscc", "s1res", "mbt", "turbo", "msillm", "codlite",
         "crdr", "ditic", "ditics", "pulse", "mix", "expanded_mix"]
lines = ["### Bảng Q̂ v0 trên cùng 30 ảnh", "",
         "Mỗi ô là **Q̂ / bpp thực** dưới ngân sách cột. `Không vừa` nghĩa là các điểm đã có không đủ thấp để nén toàn tập. Bitrate thấp hơn ngân sách không được nội suy thành một điểm mới.", "",
         "| Phương pháp | ≤0,075 bpp | ≤0,15 bpp | ≤0,3 bpp |", "|---|---:|---:|---:|"]
for cand in order:
    cells = []
    for rate in (.075, .15, .3):
        r = next((r for r in rows if r["method"] == cand and r["budget_bpp"] == rate), None)
        cells.append(f"{r['qhat_v0']:.4f} / {r['actual_bpp']:.4f}" if r else
                     (f"Thiếu metric ({s1_coverage}/30 ảnh)" if cand == "s1res" else "Không vừa"))
    lines.append("| " + NAMES[cand] + " | " + " | ".join(cells) + " |")


def table(data):
    result = ["| Phương pháp | bpp thực | PSNR pooled ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | DISTS ↓ | CLIP-IQA ↑ | Q̂ v0 ↑ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data:
        result.append(f"| {r['display']} | {r['actual_bpp']:.5f} | {r['psnr_pooled']:.2f} | "
                      + " | ".join(f"{r[k]:.4f}" for k in [*FEATURES, "qhat_v0"]) + " |")
    return result


for rate in (.075, .15, .3):
    lines += ["", f"### Đầy đủ metrics — ngân sách {rate} bpp, 30 ảnh", ""]
    lines += table(sorted([r for r in rows if r["budget_bpp"] == rate], key=lambda r: order.index(r["method"])))
lines += ["", "### AEIC: checkpoint cố định, chưa khớp rate với bảng chính", ""]
lines += table(fixed)
lines += ["", "### Pilot 6 ảnh — chỉ so trong cùng nhóm này", "",
          "AFP-GIC/DC-VIC/PULSE-L dưới đây là pilot 6 ảnh. Baseline pilot chưa có PULSE-L; khác baseline tích hợp của bảng 30 ảnh. PULSE base đã chạy 24 điểm nhưng không vừa cả ba budget, nên không có hàng allocation hợp lệ.", ""]
for rate in (.075, .15, .3):
    lines += [f"#### Ngân sách {rate} bpp", ""] + table([r for r in pilot_rows if r["budget_bpp"] == rate]) + [""]
(OUT / "tables.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
report = ROOT / "reports/CLIC2027_metrics_comparison_2026-10-20.md"
if report.exists():
    text = report.read_text(encoding="utf-8")
    start, end = "<!-- GENERATED_TABLES -->", "<!-- END_GENERATED_TABLES -->"
    if end not in text:
        text = text.replace(start, start + "\n" + end)
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    report.write_text(before + start + "\n\n" + "\n".join(lines) + "\n\n" + end + after, encoding="utf-8")
print(json.dumps(dict(full30_rows=len(rows), fixed_rows=len(fixed), pilot_rows=len(pilot_rows),
                     s1_points_with_metrics=sum(len(v) for v in options.values()), s1_images_with_metrics=s1_coverage), indent=2))
