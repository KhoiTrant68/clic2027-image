"""Why the one-step map is hard to learn at d = 64, and what start noise buys.

(a) round 1 (toy_gaussian/results/d64_delta2_samples.npz): marginal lower bound on P^2, split over coordinate
    groups by prior variance, for X*, the exact map T(X*) and learned one-step / ODE decoders.
(b) experiment A (toy_gaussian/results_A/*.json, drawn when available): distortion vs perception of the
    one-step decoder for A1-A7, with the noisy oracle maps for A5-A7 (start-noise trade-off).
"""
import json

import numpy as np
from scipy.special import ndtri

from pubstyle import COUPLING_STYLE, PALETTE, REPO, apply_publication_style, finalize_figure, legend_panel, missing, plt

SAMPLES = REPO / "toy_gaussian/results/d64_delta2_samples.npz"
RES_A = REPO / "toy_gaussian/results_A"
SIGMA = np.arange(1, 65) ** -0.5
GROUPS = {"high variance\n(coords 1-8)": slice(0, 8), "mid variance\n(9-32)": slice(8, 32), "low variance\n(33-64)": slice(32, 64)}


def marginal_w2sq(x):
    n = x.shape[0]
    q = SIGMA[None, :] * ndtri((np.arange(1, n + 1)[:, None] - 0.5) / n)
    return ((np.sort(x, 0) - q) ** 2).mean(0)


def panel_groups(ax):
    s = np.load(SAMPLES)
    rows = [("X*", r"MMSE $X^*$", PALETTE["neutral"]),
            ("natural__euler1", "Natural, 1 step", PALETTE["red_1"]),
            ("natural__ode", "Natural, ODE", PALETTE["red_strong"]),
            ("minibatch_ot__euler1", "Minibatch OT, 1 step", PALETTE["blue_secondary"]),
            ("exact_ot__euler1", "Exact OT (learned), 1 step", PALETTE["blue_main"]),
            ("T(X*)", r"Exact map $T(X^*)$", PALETTE["green_3"])]
    width = 0.13
    for i, (key, lab, col) in enumerate(rows):
        w = marginal_w2sq(s[key])
        vals = [w[sl].sum() for sl in GROUPS.values()]
        ax.bar(np.arange(3) + (i - 2.5) * width, vals, width, color=col, edgecolor="black", lw=1.2, label=lab)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(list(GROUPS))
    ax.set_ylabel(r"$P^2$ lower bound (sum of marginal $W_2^2$)")
    ax.set_title("(a) Error of learned maps sits in compressed coordinates", loc="left", fontsize=14)
    return ax.get_legend_handles_labels()


def panel_A(ax):
    runs = sorted(RES_A.glob("A*.json"))
    if not runs:
        ax.text(0.5, 0.5, "experiment A pending\n(toy_gaussian/results_A)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color=PALETTE["neutral_dark"])
        ax.set_axis_off()
        return [], []
    style = {"A1": ("A1 base", PALETTE["blue_main"], "s"), "A2": ("A2 capacity", PALETTE["blue_secondary"], "s"),
             "A3": ("A3 precond.", PALETTE["teal"], "D"), "A4": ("A4 capacity+precond.", PALETTE["violet"], "D")}
    noise_pts, oracle_pts = [], []
    for f in runs:
        r = json.loads(f.read_text())
        e1 = r["eval"]["euler1"]
        tag = r["tag"].split("_")[0]
        if r.get("noise_c", 0) > 0:
            o = r["eval"]["oracle_noisy_map"]
            noise_pts.append((r["noise_c"], e1["P_marg_lb"], e1["D_over_opt"]))
            oracle_pts.append((r["noise_c"], o["P_marg_lb"], o["D_over_opt"]))
        else:
            lab, col, mk = style.get(tag, (tag, PALETTE["ink"], "o"))
            ax.plot(e1["P_marg_lb"], e1["D_over_opt"], mk, color=col, ms=12, mec="black", label=lab)
    for pts, lab, col, mk in ((sorted(noise_pts), "start noise, learned", PALETTE["green_3"], "o"),
                              (sorted(oracle_pts), "start noise, oracle map", PALETTE["green_2"], "^")):
        if pts:
            c, p, d = zip(*pts)
            ax.plot(p, d, "-" + mk, color=col, ms=11, mec="black", lw=2, label=lab)
            for ci, pi, di in pts:
                ax.annotate(f"c={ci:g}", (pi, di), textcoords="offset points", xytext=(8, 4), fontsize=11)
    ax.axhline(1.0, color=PALETTE["blue_main"], ls=(0, (1, 2)), lw=1.2)
    ax.text(ax.get_xlim()[1], 1.0, r"$D^*+W^2$ ", ha="right", va="bottom", fontsize=11, color=PALETTE["blue_main"])
    ax.set_xlabel(r"$P$ lower bound (1 step)")
    ax.set_ylabel(r"$D / (D^* + W^2)$ (1 step)")
    ax.set_title("(b) Start noise trades distortion for learnability", loc="left", fontsize=14)
    return ax.get_legend_handles_labels()


def main():
    if missing(SAMPLES, "round-1 d=64 samples"):
        return
    apply_publication_style(font_size=14)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw=dict(width_ratios=[1.15, 1, 0.5]))
    h1, l1 = panel_groups(axs[0])
    h2, l2 = panel_A(axs[1])
    legend_panel(axs[2], h1 + h2, l1 + l2, fontsize=12)
    finalize_figure(fig, "fig_learnability")


if __name__ == "__main__":
    main()
