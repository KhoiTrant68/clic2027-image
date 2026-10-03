"""Dither rate penalty (1-D Gaussian source, exact): MMSE after dithered vs. deterministic uniform quantization
at equal entropy-coded rate, with the per-latent-element rate ranges of SD3.5's VAE (f8c16) and SANA's DC-AE
(f32c32) at 0.01-0.05 bpp. Curves are cached in figures/data/dither_rate.json (needs toy_gaussian/toy_dp.py).
"""
import json
import sys

import numpy as np
from scipy.special import ndtr

from pubstyle import PALETTE, REPO, apply_publication_style, finalize_figure, plt

CACHE = REPO / "paper/figures/data/dither_rate.json"


def compute():
    sys.path.insert(0, str(REPO / "experiments/toy_gaussian"))
    import toy_dp as T  # noqa: E402

    s = np.array([1.0])
    Rd, Dd, Rn, Dn = [], [], [], []
    for dl in np.exp(np.linspace(np.log(0.25), np.log(400), 110)):
        th = T.theory(s, dl, n_grid=20001)
        Rd.append(th["rate_bits"]); Dd.append(th["D_star"])
        ks = np.arange(-(int(12 / dl) + 3), int(12 / dl) + 4)
        p = np.clip(ndtr(ks * dl + dl / 2) - ndtr(ks * dl - dl / 2), 1e-300, 1)
        _, v = T.mmse(ks * dl * 1.0, s, dl)
        Rn.append(float(-(p * np.log2(p)).sum())); Dn.append(float((p * v).sum()))
    data = {"dither": {"R": Rd, "D": Dd}, "nodither": {"R": Rn, "D": Dn}}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def main():
    data = json.loads(CACHE.read_text()) if CACHE.exists() else compute()
    R = np.exp(np.linspace(np.log(0.005), np.log(2.5), 300))
    interp = lambda k: np.interp(R, np.array(data[k]["R"])[::-1], np.array(data[k]["D"])[::-1])
    Dd, Dn, Dsh = interp("dither"), interp("nodither"), 2.0 ** (-2 * R)
    valid = R >= min(data["dither"]["R"])

    apply_publication_style(font_size=15)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    bands = [((0.01 * 4, 0.05 * 4), "SD3.5 VAE (f8c16)", PALETTE["red_1"]),
             ((0.01 * 32, 0.05 * 32), "SANA DC-AE (f32c32)", PALETTE["green_1"])]
    for ax in axs:
        for (lo, hi), lab, col in bands:
            ax.axvspan(lo, hi, color=col, alpha=0.9, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Rate per latent element (bits)")

    ax = axs[0]
    ax.plot(R, Dsh, color=PALETTE["ink"], ls=":", lw=2, label=r"Shannon bound $2^{-2R}$")
    ax.plot(R, Dn, color=PALETTE["red_strong"], lw=2.5, label="Deterministic rounding + MMSE")
    ax.plot(R[valid], Dd[valid], color=PALETTE["blue_main"], lw=2.5, label="Subtractive dither + MMSE")
    ax.set_ylabel(r"MMSE $D^*$ (unit-variance source)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Distortion at equal rate", loc="left", fontsize=15)
    ax.legend(fontsize=12, loc="lower left")

    ax = axs[1]
    frac = (1 - Dd) / (1 - Dn)
    ax.plot(R[valid], frac[valid], color=PALETTE["blue_main"], lw=2.5)
    ax.axhline(1, color=PALETTE["ink"], lw=1, ls="--")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Distortion reduction kept by dither\n" + r"$(1 - D^*_{\rm dither}) / (1 - D^*_{\rm round})$")
    ax.set_title("(b) Dither penalty vanishes above ~1 bit", loc="left", fontsize=15)
    for (lo, hi), lab, col in bands:
        ax.text(np.sqrt(lo * hi), 0.43, lab + "\n0.01-0.05 bpp", ha="center", va="bottom", fontsize=11,
                color=PALETTE["ink"])
    finalize_figure(fig, "fig_dither_rate")


if __name__ == "__main__":
    main()
