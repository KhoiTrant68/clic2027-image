# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figures + summary table for the toy D-P experiment (reads results/*.json)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
COUPLINGS = {"natural": ("Natural $(X^*,X)$", "#d62728"), "independent": ("Independent", "#7f7f7f"),
             "minibatch_ot": ("Minibatch OT", "#ff7f0e"), "exact_ot": ("Exact OT", "#1f77b4")}


def dp_figure(r, p_key, fname, title):
    th = r["theory"]
    Ds, W = th["D_star"], th["W"]
    P = np.linspace(0, 1.15 * max(W, r["w2_floor"] if p_key == "P" else 0), 200)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(P, Ds + np.clip(W - P, 0, None) ** 2, "k-", lw=2, label=r"$D(P)=D^*+(W-P)_+^2$")
    ax.axhline(2 * Ds, color="k", ls=":", lw=1, label=r"$2D^*$ (posterior sampling)")
    ep = r["exact_path"]
    ax.plot([e[p_key] for e in ep], [e["D"] for e in ep], "o", mfc="none", color="k", ms=5,
            label=r"exact OT path $\hat x_t$ (measured)")
    for c, (lab, col) in COUPLINGS.items():
        f = r["flows"][c]
        pts = [f[k] for k in ["ode_t0.25", "ode_t0.5", "ode_t0.75", "ode"]]
        ax.plot([f["euler1"][p_key]], [f["euler1"]["D"]], "s", color=col, ms=8, label=f"{lab}: 1 Euler step")
        ax.plot([p[p_key] for p in pts], [p["D"] for p in pts], "^-", color=col, ms=6, lw=1, alpha=0.8,
                label=f"{lab}: ODE, $t$=.25→1")
    if p_key == "P":
        ax.axvline(r["w2_floor"], color="gray", ls="--", lw=1, label="sample-$W_2$ floor")
    ax.set_xlabel("P = $W_2(p_{\\hat X}, p_X)$" + (" (sample estimate)" if p_key == "P" else " (marginal lower bound)"))
    ax.set_ylabel(r"D = $E\|\hat X - X\|^2$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(RES / fname, dpi=160)
    plt.close(fig)


def scatter_figure(tag):
    s = np.load(RES / f"{tag}_samples.npz")
    panels = [("X", "X (target)"), ("X*", "X* (MMSE)"), ("T(X*)", "T(X*) exact OT"),
              ("natural__euler1", "natural, 1 Euler"), ("natural__ode", "natural, ODE"),
              ("independent__euler1", "independent, 1 Euler"), ("minibatch_ot__euler1", "minibatch OT, 1 Euler"),
              ("exact_ot__euler1", "exact OT, 1 Euler")]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True)
    for ax, (k, t) in zip(axs.flat, panels):
        ax.scatter(s[k][:1500, 0], s[k][:1500, 1], s=2, alpha=0.5)
        ax.set_title(t, fontsize=9)
        ax.set_aspect("equal")
    axs[0, 0].set_xlim(-3.5, 3.5)
    axs[0, 0].set_ylim(-2, 2)
    fig.tight_layout()
    fig.savefig(RES / f"{tag}_scatter.png", dpi=140)
    plt.close(fig)


def table(r):
    th = r["theory"]
    Ds, W2 = th["D_star"], th["W2"]
    lines = [f"### {r['tag']}  (d={r['d']}, Δ={r['delta']}, rate={th['rate_bits']:.2f} bits total)",
             f"D*={Ds:.4f}  W²={W2:.4f}  W={th['W']:.4f}  D*+W²={Ds + W2:.4f}  2D*={2 * Ds:.4f}  "
             f"sample-W2 floor={r['w2_floor']:.4f}", "",
             "| decoder | D | D / (D*+W²) | P (sample) | P marg. LB | ‖v(X*,0)‖² |", "|---|---|---|---|---|---|"]
    e0, e1 = r["exact_path"][0], r["exact_path"][-1]
    lines.append(f"| MMSE X* | {e0['D']:.4f} | {e0['D'] / (Ds + W2):.3f} | {e0['P']:.4f} | {e0['P_marg_lb']:.4f} | |")
    lines.append(f"| exact T(X*) | {e1['D']:.4f} | {e1['D'] / (Ds + W2):.3f} | {e1['P']:.4f} | {e1['P_marg_lb']:.4f} | |")
    for c, (lab, _) in COUPLINGS.items():
        f = r["flows"][c]
        for k, name in [("euler1", "1 Euler"), ("ode", "ODE 50 RK2")]:
            lines.append(f"| {c}, {name} | {f[k]['D']:.4f} | {f[k]['D'] / (Ds + W2):.3f} | {f[k]['P']:.4f} | "
                         f"{f[k]['P_marg_lb']:.4f} | {f['mean_sq_v0']:.4f} |")
    lines += ["", "Exact OT path check (Thm 3): t, D measured vs D*+t²W², P_marg_lb vs (1-t)W", ""]
    for e in r["exact_path"][::2]:
        lines.append(f"- t={e['t']:.1f}: D={e['D']:.4f} (theory {e['D_theory']:.4f}); "
                     f"P={e['P']:.4f}, P_lb={e['P_marg_lb']:.4f} (theory {e['P_theory']:.4f})")
    return "\n".join(lines)


def main():
    out = []
    for tag in ["d2_delta4", "d64_delta2"]:
        p = RES / f"{tag}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r["d"] <= 2:
            dp_figure(r, "P", f"{tag}_dp.png", f"Toy D–P, d={r['d']}, Δ={r['delta']}")
            scatter_figure(tag)
        dp_figure(r, "P_marg_lb", f"{tag}_dp_marglb.png", f"Toy D–P, d={r['d']}, Δ={r['delta']} (marginal LB on P)")
        out.append(table(r))
    (RES / "summary.md").write_text("\n\n".join(out), encoding="utf-8")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
