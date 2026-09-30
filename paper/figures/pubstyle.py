"""Shared publication style (figures4papers house style, see ~/.claude/skills/scientific-figure-making)."""
from __future__ import annotations

from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "neutral_dark": "#767676", "ink": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}

# Semantic roles used across all figures of this paper (keep them identical everywhere).
COUPLING_STYLE = {
    "exact_ot": dict(color=PALETTE["blue_main"], label="Exact OT (ours)"),
    "near_exact_ot_2048": dict(color=PALETTE["blue_main"], label="OT, pool 2048 (ours)"),
    "minibatch_ot": dict(color=PALETTE["blue_secondary"], label="Minibatch OT"),
    "minibatch_ot_256": dict(color=PALETTE["blue_secondary"], label="Minibatch OT, pool 256"),
    "natural": dict(color=PALETTE["red_strong"], label=r"Natural $(X^*, X)$"),
    "independent": dict(color=PALETTE["neutral_dark"], label="Independent"),
}

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"


def apply_publication_style(font_size: int = 16, axes_linewidth: float = 2.0):
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"],  # Helvetica per house style is absent on Windows
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "xtick.major.width": axes_linewidth * 0.8,
        "ytick.major.width": axes_linewidth * 0.8,
    })


def legend_panel(ax, handles, labels, fontsize=14, loc="center left", ncol=1):
    """Dedicated legend axis (keeps data panels clean)."""
    ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=False, ncol=ncol)
    ax.set_axis_off()


def finalize_figure(fig, name: str, formats=("png", "pdf"), dpi=300, pad=2.0):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    paths = []
    for fmt in formats:
        p = OUT / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    print("saved", ", ".join(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in paths))
    return paths


def missing(path: Path, what: str) -> bool:
    if not path.exists():
        print(f"[skip] {what}: {path.relative_to(REPO) if path.is_relative_to(REPO) else path} not found yet")
        return True
    return False
