"""Figures for experiments that run on Kaggle/TPU. Each one is skipped until its results exist.

fig_toyB      toy B  (toy_gaussian/results_B/*.json): minibatch-OT pool size vs. coupling quality and one-step decoding
fig_gonogo1   go/no-go 1 (quant_noise/runs/gonogo1/results.json): latent MSE, exact vs. Gaussian-assumption denoiser
fig_quantstat quantization-error statistics (quant_noise/runs/<name>/quant_noise_stats.json) vs. bpp
"""
import json
import sys
from pathlib import Path

import numpy as np

from pubstyle import PALETTE, REPO, apply_publication_style, finalize_figure, legend_panel, missing, plt

RES_B = REPO / "toy_gaussian/results_B"
GONOGO = REPO / "quant_noise/runs/gonogo1/results.json"
QSTAT = REPO / "quant_noise/runs/kodak/quant_noise_stats.json"


def fig_toyB(res_dir=RES_B):
    runs = [json.loads(f.read_text()) for f in sorted(res_dir.glob("B_*.json"))]
    if not runs:
        print(f"[skip] toy B: no results in {res_dir}")
        return
    apply_publication_style(font_size=15)
    fig, axs = plt.subplots(1, 4, figsize=(21, 5), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.45]))
    dcol = {16: PALETTE["blue_secondary"], 64: PALETTE["blue_main"]}
    for d in (16, 64):
        rows = [r for r in runs if r["d"] == d]
        ref = {r["coupling"]: r for r in rows if r["coupling"] in ("independent", "exact_ot")}
        mb = sorted([r for r in rows if r["coupling"] in ("lsa", "sinkhorn")], key=lambda r: (r["pool"], r["coupling"]))
        for coupling, mk, ls in (("lsa", "o", "-"), ("sinkhorn", "s", "--")):
            pts = [r for r in mb if r["coupling"] == coupling]
            if not pts:
                continue
            x = [r["pool"] for r in pts]
            lab = f"d={d}, {'exact assignment' if coupling == 'lsa' else 'Sinkhorn'}"
            axs[0].plot(x, [r["pair_cost_over_W2"] for r in pts], mk + ls, color=dcol[d], ms=9, mec="black", lw=2, label=lab)
            axs[1].plot(x, [r["eval"]["euler1"]["P_marg_lb"] for r in pts], mk + ls, color=dcol[d], ms=9, mec="black", lw=2)
            axs[2].plot(x, [r["eval"]["euler1"]["D_over_opt"] for r in pts], mk + ls, color=dcol[d], ms=9, mec="black", lw=2)
        for key, ls in (("exact_ot", ":"), ("independent", "-.")):
            if key in ref:
                e = ref[key]
                axs[1].axhline(e["eval"]["euler1"]["P_marg_lb"], color=dcol[d], ls=ls, lw=1.3)
                axs[2].axhline(e["eval"]["euler1"]["D_over_opt"], color=dcol[d], ls=ls, lw=1.3)
        if "exact_ot" in ref:  # natural-coupling cost D* for comparison: pool OT degenerates towards it
            axs[0].axhline(ref["exact_ot"]["theory"]["D_star"] / ref["exact_ot"]["theory"]["W2"], color=dcol[d], ls="-.", lw=1.3)
    axs[0].axhline(1.0, color=PALETTE["ink"], ls=":", lw=1.3)
    axs[0].set_ylabel(r"pair cost $/\,W^2$  (1 = population OT)")
    axs[1].set_ylabel(r"1-step $P$ lower bound")
    axs[2].set_ylabel(r"1-step $D/(D^*+W^2)$")
    for ax, t in zip(axs[:3], ("(a) Coupling quality", "(b) One-step perception", "(c) One-step distortion")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("OT pool size")
        ax.set_title(t, loc="left", fontsize=15)
    h, l = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=PALETTE["ink"], ls=":"), Line2D([], [], color=PALETTE["ink"], ls="-.")]
    l += ["exact OT coupling (panels b, c) / population OT (a)", r"independent (b, c) / natural coupling $D^*/W^2$ (a)"]
    legend_panel(axs[3], h, l, fontsize=12)
    finalize_figure(fig, "fig_toyB")


def fig_gonogo1(src=GONOGO):
    if missing(src, "go/no-go 1 results"):
        return
    r = json.loads(src.read_text())
    tests = list(r["tests"])
    rates = r["rates"]
    apply_publication_style(font_size=15)
    fig, axs = plt.subplots(1, len(tests) + 1, figsize=(7 * len(tests) + 3.5, 5.2),
                            gridspec_kw=dict(width_ratios=[1] * len(tests) + [0.45]))
    axs = np.atleast_1d(axs)
    arms = [("raw_dither", r"no denoiser ($\hat z$)", PALETTE["neutral"], None),
            ("gaussian", "trained on Gaussian assumption", PALETTE["red_2"], None),
            ("exact", "trained on exact noise (ours)", PALETTE["blue_main"], None),
            ("nodither", "deterministic rounding (ref., lower rate)", "white", "//")]
    width = 0.2
    for ax, name in zip(axs, tests):
        t = r["tests"][name]
        per = list(t["per_delta"].values())
        x = np.arange(len(per))
        for i, (key, lab, col, hatch) in enumerate(arms):
            ax.bar(x + (i - 1.5) * width, [p[key] for p in per], width, color=col, edgecolor="black", lw=1.2,
                   hatch=hatch, label=lab)
        for xi, p in zip(x, per):
            gain = 1 - p["exact"] / p["gaussian"]
            ax.text(xi + 0.5 * width, p["exact"], f"{gain:+.0%}", ha="center", va="bottom", fontsize=12,
                    color=PALETTE["blue_main"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q['target_bpp']:g}" for q in rates])
        ax.set_xlabel("target bpp")
        ax.set_ylabel("latent MSE")
        ax.set_yscale("log")
        verdict = "PASS" if t["pass"] else "FAIL"
        ax.set_title(f"{name}: {t['wins']}/{len(per)} rates ≥ threshold -> {verdict}", loc="left", fontsize=14)
    h, l = axs[0].get_legend_handles_labels()
    legend_panel(axs[-1], h, l, fontsize=12)
    finalize_figure(fig, "fig_gonogo1")


def fig_quantstat(src=QSTAT):
    if missing(src, "quantization-error statistics"):
        return
    r = json.loads(src.read_text())
    stats = [("latent_channel", "kurt", "excess kurtosis (latent)", 0.0),
             ("latent_channel", "corr_ey", r"corr$(e, z)$ (latent)", 0.0),
             ("latent", "offdiag_abs_corr_mean", "mean |channel corr| of $e$", None),
             ("latent", "content_spearman_patch", "content dependence (Spearman)", 0.0)]
    modes = [("no_dither", "deterministic rounding", PALETTE["red_strong"], "o"),
             ("dither", "subtractive dither", PALETTE["blue_main"], "s"),
             ("gauss_ref", "Gaussian assumption (null)", PALETTE["neutral_dark"], "^")]
    apply_publication_style(font_size=14)
    fig, axs = plt.subplots(1, len(stats) + 1, figsize=(24, 4.8), gridspec_kw=dict(width_ratios=[1] * len(stats) + [0.5]))
    coder = "klt2" if "klt2" in r["coders"] else list(r["coders"])[0]
    rows = r["coders"][coder]["rates"]
    x = [e["target_bpp"] for e in rows]
    for ax, (dom, key, lab, zero) in zip(axs, stats):
        for mode, mlab, col, mk in modes:
            v = [e[mode][dom][key]["median"] if isinstance(e[mode][dom][key], dict) else e[mode][dom][key] for e in rows]
            ax.plot(x, v, "-" + mk, color=col, ms=9, mec="black", lw=2, label=mlab)
        if zero is not None:
            ax.axhline(zero, color=PALETTE["ink"], lw=1, ls=":")
        ax.set_xlabel("bpp")
        ax.set_title(lab, loc="left", fontsize=14)
    h, l = axs[0].get_legend_handles_labels()
    legend_panel(axs[-1], h, l, fontsize=12)
    fig.suptitle(f"Quantization error on {src.parent.name} latents ({coder} proxy coder)", x=0.02, ha="left", fontsize=15)
    finalize_figure(fig, "fig_quantstat")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"toyB", "gonogo1", "quantstat"}
    if "toyB" in which:
        fig_toyB()
    if "gonogo1" in which:
        fig_gonogo1()
    if "quantstat" in which:
        fig_quantstat()
