"""Fig. 3 candidate: toy D-P plane. Theory curve, exact OT geodesic, one-step vs ODE decoders per coupling.

(a) separable toy (KLT basis, d=2, Delta=4, closed form)   <- toy_gaussian/results/d2_delta4.json
(b) non-separable toy (correlated Sigma, Delta=4)         <- toy_gaussian/results_C/C_corr2d.json
"""
import json

import numpy as np
from matplotlib.lines import Line2D

from pubstyle import COUPLING_STYLE, PALETTE, REPO, apply_publication_style, finalize_figure, legend_panel, missing, plt

SRC_A = REPO / "toy_gaussian/results/d2_delta4.json"
SRC_B = REPO / "toy_gaussian/results_C/C_corr2d.json"


def load_a():
    r = json.loads(SRC_A.read_text())
    th = r["theory"]
    return dict(Ds=th["D_star"], W=th["W"], floor=r["w2_floor"], path=r["exact_path"], flows=r["flows"],
                title="(a) Separable toy (closed form)")


def load_b():
    r = json.loads(SRC_B.read_text())
    return dict(Ds=r["D_star"], W=r["W"], floor=r["w2_floor"], path=r["exact_path"], flows=r["flows"],
                title="(b) Non-separable toy")


def panel(ax, d):
    Ds, W = d["Ds"], d["W"]
    xmax, ytop = 1.25 * W, Ds + 1.9 * W**2          # zoom on the informative region; off-scale points get arrows
    P = np.linspace(0, xmax, 300)
    curve = Ds + np.clip(W - P, 0, None) ** 2
    ax.fill_between(P, curve, Ds - W**2, color=PALETTE["neutral"], alpha=0.35, lw=0)
    ax.plot(P, curve, color=PALETTE["ink"], lw=2.5)
    ax.axhline(Ds + W**2, color=PALETTE["blue_main"], ls=(0, (1, 2)), lw=1.2)
    ax.axvline(d["floor"], color=PALETTE["neutral_dark"], ls="--", lw=1.2)
    ax.plot([e["P"] for e in d["path"]], [e["D"] for e in d["path"]], "o", mfc="white", mec=PALETTE["ink"], ms=7, mew=1.5)
    for name, f in d["flows"].items():
        st = COUPLING_STYLE[name]
        for key, marker, size in (("ode", "^", 10), ("euler1", "s", 12)):
            p, dd = f[key]["P"], f[key]["D"]
            if p > xmax or dd > ytop:  # e.g. independent coupling, 1 step: collapses to the mean
                x, y = min(p, xmax), min(dd, ytop)
                ax.annotate(f"{st['label'].split(' (')[0]}, 1 step:\nP={p:.2f}, D={dd:.2f}", xy=(x, y),
                            xytext=(xmax * 0.97, Ds + 1.42 * W**2), ha="right", va="top", fontsize=11,
                            color=st["color"], arrowprops=dict(arrowstyle="->", color=st["color"], lw=1.8))
                continue
            ax.plot(p, dd, marker, color=st["color"], ms=size, mec="black", mew=1.0, zorder=5)
    # the two messages of the figure
    nat = d["flows"]["natural"]["euler1"]
    ax.annotate("natural coupling, 1 step\n= MMSE $X^*$ (Prop. 2)", xy=(nat["P"], nat["D"]), xytext=(nat["P"] * 0.78, Ds + 0.5 * W**2),
                ha="center", fontsize=11, color=PALETTE["red_strong"], arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=1.5))
    ot_key = "exact_ot" if "exact_ot" in d["flows"] else "near_exact_ot_2048"
    ot = d["flows"][ot_key]["euler1"]
    ax.annotate("OT coupling, 1 step:\nperfect perception at $D^* + W^2$", xy=(ot["P"], ot["D"]), xytext=(W * 0.30, Ds + 1.62 * W**2),
                ha="left", fontsize=11, color=PALETTE["blue_main"], arrowprops=dict(arrowstyle="->", color=PALETTE["blue_main"], lw=1.5))
    ax.text(W * 0.30, Ds - 0.55 * W**2, "unachievable", fontsize=12, color=PALETTE["neutral_dark"], style="italic")
    ax.text(xmax * 0.99, Ds + 0.97 * W**2, "$D^*+W^2$", fontsize=11, color=PALETTE["blue_main"], ha="right", va="top")
    ax.set_xlim(0, xmax)
    ax.set_ylim(Ds - 0.8 * W**2, ytop)
    ax.set_xlabel(r"Perception  $P = W_2(p_{\hat X}, p_X)$")
    ax.set_ylabel(r"Distortion  $D = \mathbb{E}\|\hat X - X\|^2$")
    ax.set_title(d["title"] + f"   ($2D^* = {2 * Ds:.2f}$, off-scale)", fontsize=15, loc="left")


def main():
    if missing(SRC_A, "separable toy") or missing(SRC_B, "non-separable toy"):
        return
    apply_publication_style(font_size=15)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.62]))
    panel(axs[0], load_a())
    panel(axs[1], load_b())
    h = [Line2D([], [], color=PALETTE["ink"], lw=2.5),
         Line2D([], [], ls="none", marker="o", mfc="white", mec=PALETTE["ink"], ms=8),
         Line2D([], [], color=PALETTE["neutral_dark"], ls="--", lw=1.2),
         Line2D([], [], ls="none", marker="s", color="white", mec="black", ms=10),
         Line2D([], [], ls="none", marker="^", color="white", mec="black", ms=10)]
    l = [r"$D(P) = D^* + (W - P)_+^2$", r"OT geodesic $\hat x_t$ (Thm. 3)", r"sample-$W_2$ floor",
         "1 Euler step", "ODE, 50 steps"]
    for key in ("exact_ot", "minibatch_ot", "natural", "independent"):
        h.append(Line2D([], [], ls="none", marker="s", color=COUPLING_STYLE[key]["color"], mec="black", ms=10))
        l.append(COUPLING_STYLE[key]["label"].replace("Exact OT (ours)", "OT coupling (ours)"))
    legend_panel(axs[2], h, l, fontsize=13)
    finalize_figure(fig, "fig_toy_dp")


if __name__ == "__main__":
    main()
