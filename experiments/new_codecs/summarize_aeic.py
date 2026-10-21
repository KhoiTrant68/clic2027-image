"""Render measured AEIC full30 quality/resource results after analysis completes."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    run = a.run
    assert json.loads((run / "analysis_status.json").read_text())["state"] == "complete"
    comparison = json.loads((run / "comparison.json").read_text())
    with (run / "scores/metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    resources = {}
    for variant in ("se", "me"):
        path = run / f"aeic_{variant}_ft2"
        records = [json.loads(p.read_text()) for p in sorted((path / "q2").glob("*.json"))]
        parity = json.loads((path / "parity.json").read_text())
        assert len(records) == len(parity) == 30 and all(p["same_pixels"] for p in parity)
        assert {r["name"] for r in records} == {r["name"] for r in parity}
        measured = []
        for phase in ("encode", "decode"):
            with (path / f"{phase}.vram.jsonl").open() as f:
                measured.extend(json.loads(line)["own_used_mib"] for line in f if line.strip())
        dec_times = [p["decode_s"] for p in parity]
        own = [n for n in measured if n is not None]
        pixels = sum(r["height"] * r["width"] for r in records)
        resources[variant] = dict(
            n=len(records), corpus_bpp=8 * sum(r["bytes"] for r in records) / pixels,
            bytes=sum(r["bytes"] for r in records),
            mean_qhat=statistics.mean(float(m["score"]) for m in metrics if m["cand"] == f"aeic_{variant}"),
            encode_mean_s=statistics.mean(r["encode_s"] for r in records),
            independent_decode_mean_s=statistics.mean(dec_times),
            independent_decode_median_s=statistics.median(dec_times),
            independent_decode_max_s=max(dec_times),
            independent_decode_total_s=sum(dec_times),
            own_peak_mib=max(own), nvml_samples=len(own),
            encode_timing=json.loads((path / "encode.timing.json").read_text()),
            decode_timing=json.loads((path / "decode.timing.json").read_text()))
    with (run / "scores/allocation.csv").open() as f:
        allocation = list(csv.DictReader(f))
    selected_costs = []
    for c in comparison:
        selected = [r for r in allocation if r["cand"] == "expanded_mix" and float(r["rate"]) == c["rate"] and r["key"].startswith("aeic_")]
        seconds = 0
        for r in selected:
            variant = r["key"].split("/")[0].removeprefix("aeic_")
            parity = json.loads((run / f"aeic_{variant}_ft2/parity.json").read_text())
            seconds += next(p["decode_s"] for p in parity if p["name"] == r["name"])
        selected_costs.append(dict(rate=c["rate"], aeic_images=len(selected), aeic_decode_seconds=seconds))
    result = dict(models=resources, selected_aeic_costs=selected_costs,
                  timing_caveat="Measured on shared L40 GPUs; model loading, process startup and PNG I/O excluded from per-image API times. Not an official isolated decoder benchmark.",
                  memory_caveat="Peak from 0.25-second NVML samples; brief peaks between samples may be missed. VAEHook resets PyTorch peak counters, so those counters are not used as whole-run peak.")
    (run / "resource_summary.json").write_text(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(len(comparison))
    ax = axes[0]
    ax.bar(x - .18, [c["baseline_qhat"] for c in comparison], .36, label="Baseline incl. PULSE", color="#56687a")
    ax.bar(x + .18, [c["expanded_qhat"] for c in comparison], .36, label="Baseline + AEIC", color="#14806a")
    for i, c in enumerate(comparison):
        ax.annotate(f"Delta {c['delta_qhat']:+.4f}", (i, max(c["baseline_qhat"], c["expanded_qhat"])), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x, [str(c["rate"]) for c in comparison])
    ax.set(xlabel="Corpus budget (bpp)", ylabel="Mean Qhat-v0 (higher is better)", title="Same 30 images; real bitstream bytes")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + .2)
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    bottom = np.zeros(3)
    colors = {"pulse": "#4575b4", "msillm": "#91bfdb", "ditic": "#fee090", "ditics": "#e0b866", "vtmscc": "#bd7d70", "codlite": "#b8b8b8", "aeic_me": "#14806a", "aeic_se": "#7bc8a4"}
    for cand in colors:
        values = np.array([c["counts"].get(cand, 0) for c in comparison])
        if values.sum():
            ax.bar(x, values, bottom=bottom, label=cand, color=colors[cand])
            for i, n in enumerate(values):
                if n:
                    ax.text(i, bottom[i] + n / 2, str(n), ha="center", va="center", fontsize=10)
            bottom += values
    ax.set_xticks(x, [str(c["rate"]) for c in comparison])
    ax.set(xlabel="Corpus budget (bpp)", ylabel="Selected images", ylim=(0, 30), title="Expanded allocation by codec")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    fig.savefig(run / "quality_and_selection.png", dpi=160)
    fig.savefig(run / "quality_and_selection.svg")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
