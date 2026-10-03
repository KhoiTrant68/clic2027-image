# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "torch", "matplotlib"]
# ///
"""Round-2 toy C: a NON-separable 2-D source, for Fig. 3.

X ~ N(0, Sigma) with Sigma = R(45deg) diag(1, 0.3^2) R^T, quantized in the STANDARD basis with a
subtractive dither of step Delta (so quantization cells are not aligned with the principal axes).
Y = X + E, E ~ Unif([-Delta/2, Delta/2)^2).

Nothing is closed form any more:
  X* = E[X | Y] = mean of N(0, Sigma) restricted to the cell Y + [-Delta/2, Delta/2)^2, by 2-D
       Gauss-Legendre quadrature (checked against a finer rule);
  D* = E||X - X*||^2 by Monte Carlo;
  W  = W2(p_X*, p_X) by exact discrete OT between n-point samples (reported for several n, with the
       sample-W2 floor W2(Z, Z') for reference), so the D-P curve is "near exact".
Learned flows: natural, independent, minibatch OT (pool 256) and near-exact OT (pool 2048).
The question: do couplings now end at different points (Thm. 4: natural lies in [D*+W^2, 2D*])?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from toy_dp import VelocityMLP, w2_empirical

ANGLE, S_MAJOR, S_MINOR = np.pi / 4, 1.0, 0.3
R = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])
SIGMA = R @ np.diag([S_MAJOR**2, S_MINOR**2]) @ R.T
SIGMA_INV = np.linalg.inv(SIGMA)
CHOL = np.linalg.cholesky(SIGMA)


def gl_nodes(n: int, delta: float):
    z, w = np.polynomial.legendre.leggauss(n)
    z, w = z * delta / 2, w * delta / 2
    zz = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
    ww = np.outer(w, w).reshape(-1)
    return zz, np.log(ww)


def mmse(y: np.ndarray, delta: float, n_nodes: int = 40, chunk: int = 4096) -> np.ndarray:
    """E[X | X in y + cell] for X ~ N(0, Sigma); log-domain quadrature, chunked."""
    zz, logw = gl_nodes(n_nodes, delta)
    out = np.empty_like(y)
    for s in range(0, len(y), chunk):
        pts = y[s:s + chunk, None, :] + zz[None, :, :]                     # (b, m, 2)
        logp = -0.5 * np.einsum("bmi,ij,bmj->bm", pts, SIGMA_INV, pts) + logw[None, :]
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        out[s:s + chunk] = (p[:, :, None] * pts).sum(1) / p.sum(1, keepdims=True)
    return out


def sample(n: int, delta: float, rng):
    x = rng.standard_normal((n, 2)) @ CHOL.T
    u = rng.uniform(-delta / 2, delta / 2, (n, 2))
    y = delta * np.round((x + u) / delta) - u
    return x, y, mmse(y, delta)


def sample_x(n, rng):
    return rng.standard_normal((n, 2)) @ CHOL.T


def lsa_pair(a, b):
    c = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r, k = linear_sum_assignment(c)
    return a[r], b[k]


# ----------------------------------------------------------------------------- flows
def make_pairs(coupling, n, delta, rng):
    if coupling == "natural":
        x, _, xs = sample(n, delta, rng)
        return xs, x
    _, _, xs = sample(n, delta, rng)
    z = sample_x(n, rng)
    if coupling == "independent":
        return xs, z
    return lsa_pair(xs, z)  # minibatch OT with pool n


def train(coupling, pool, delta, steps, batch, width, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = VelocityMLP(2, width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    buf0 = buf1 = np.zeros((0, 2))
    for it in range(steps):
        while len(buf0) < batch:
            a, b = make_pairs(coupling, max(pool, batch) if coupling in ("natural", "independent") else pool, delta, rng)
            perm = rng.permutation(len(a))
            buf0, buf1 = np.concatenate([buf0, a[perm]]), np.concatenate([buf1, b[perm]])
        x0, x1 = torch.from_numpy(buf0[:batch]).float(), torch.from_numpy(buf1[:batch]).float()
        buf0, buf1 = buf0[batch:], buf1[batch:]
        t = torch.rand(batch)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 1000 == 0:
            print(f"    [{coupling}/{pool}] step {it + 1}/{steps} loss {loss.item():.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def integrate(model, x0, t_end, n_steps):
    x = torch.from_numpy(x0).float()
    h = t_end / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * h)
        if n_steps == 1:
            x = x + h * model(x, t)
        else:
            xm = x + 0.5 * h * model(x, t)
            x = x + h * model(xm, t + 0.5 * h)
    return x.double().numpy()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="results_C")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    delta = args.delta
    rng = np.random.default_rng(args.seed + 12345)

    # --- quadrature check and D*
    x_big, y_big, xs_big = sample(200_000, delta, rng)
    fine = mmse(y_big[:5000], delta, n_nodes=80)
    quad_err = float(np.abs(fine - xs_big[:5000]).max())
    d_star = float(((x_big - xs_big) ** 2).sum(1).mean())
    d_star_se = float(((x_big - xs_big) ** 2).sum(1).std() / np.sqrt(len(x_big)))

    # --- W by discrete OT at several n; floor = W2 between two X samples of the same size
    w_est = []
    for n in (1000, 2000, 4000, 8000):
        xs_n = xs_big[rng.choice(len(xs_big), n, replace=False)]
        w_est.append({"n": n, "W": w2_empirical(xs_n, sample_x(n, rng)),
                      "floor": w2_empirical(sample_x(n, rng), sample_x(n, rng))})
        print("W estimate", w_est[-1], flush=True)
    W = w_est[-1]["W"]
    res = {"delta": delta, "sigma": SIGMA.tolist(), "D_star": d_star, "D_star_se": d_star_se,
           "quad_max_abs_err": quad_err, "W_estimates": w_est, "W": W, "W2": W**2}
    print(f"D*={d_star:.4f}±{d_star_se:.4f}  W≈{W:.4f}  D*+W²≈{d_star + W**2:.4f}  2D*={2 * d_star:.4f}  "
          f"quad err {quad_err:.1e}", flush=True)

    # --- evaluation set; empirical OT geodesic (Thm. 3) using a discrete OT pairing
    x, _, xs = sample(args.n_eval, delta, rng)
    x_ref = sample_x(args.n_eval, rng)
    res["w2_floor"] = w2_empirical(sample_x(args.n_eval, rng), x_ref)
    xs_p, z_p = lsa_pair(xs, sample_x(args.n_eval, rng))  # pairs (X*_i, Z_i) with Z ~ p_X
    order = {tuple(v): i for i, v in enumerate(xs)}
    x_p = x[[order[tuple(v)] for v in xs_p]]              # keep X aligned with its X*
    res["exact_path"] = []
    for t in np.linspace(0, 1, 6):
        xh = (1 - t) * xs_p + t * z_p
        res["exact_path"].append({"t": float(t), "D": float(((xh - x_p) ** 2).sum(1).mean()),
                                  "P": w2_empirical(xh, x_ref)})

    # --- learned flows
    samples = {"X": x[:2000], "X*": xs[:2000], "OT(X*)": z_p[:2000]}
    res["flows"] = {}
    for name, coupling, pool in [("natural", "natural", 0), ("independent", "independent", 0),
                                 ("minibatch_ot_256", "ot", 256), ("near_exact_ot_2048", "ot", 2048)]:
        print(f"  training {name}", flush=True)
        model = train(coupling, pool, delta, args.steps, args.batch, args.width, args.seed)
        entry = {}
        for k, t_end, n_steps in [("euler1", 1.0, 1), ("ode", 1.0, 50), ("ode_t0.25", 0.25, 13),
                                  ("ode_t0.5", 0.5, 25), ("ode_t0.75", 0.75, 38)]:
            xh = integrate(model, xs, t_end, n_steps)
            entry[k] = {"D": float(((xh - x) ** 2).sum(1).mean()), "P": w2_empirical(xh, x_ref)}
            if k in ("euler1", "ode"):
                samples[f"{name}/{k}"] = xh[:2000]
        res["flows"][name] = entry
        print(f"  {name}: " + ", ".join(f"{k}: D={v['D']:.4f} P={v['P']:.4f}" for k, v in entry.items()), flush=True)
    res["seconds"] = time.time() - t0
    (out / "C_corr2d.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / "C_samples.npz", **{k.replace("/", "__"): v for k, v in samples.items()})
    plot(res, samples, out)


def plot(res, samples, out):
    Ds, W = res["D_star"], res["W"]
    cols = {"natural": "#d62728", "independent": "#7f7f7f", "minibatch_ot_256": "#ff7f0e",
            "near_exact_ot_2048": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    P = np.linspace(0, 1.1 * W, 200)
    ax.plot(P, Ds + np.clip(W - P, 0, None) ** 2, "k-", lw=2, label=r"$D^*+(W-P)_+^2$ ($W$ by discrete OT)")
    ax.axhline(2 * Ds, color="k", ls=":", lw=1, label=r"$2D^*$")
    ax.axvline(res["w2_floor"], color="gray", ls="--", lw=1, label="sample-$W_2$ floor")
    ep = res["exact_path"]
    ax.plot([e["P"] for e in ep], [e["D"] for e in ep], "o", mfc="none", color="k", label="empirical OT path")
    for name, f in res["flows"].items():
        c = cols[name]
        ax.plot(f["euler1"]["P"], f["euler1"]["D"], "s", color=c, ms=8, label=f"{name}: 1 Euler step")
        pts = [f[k] for k in ("ode_t0.25", "ode_t0.5", "ode_t0.75", "ode")]
        ax.plot([p["P"] for p in pts], [p["D"] for p in pts], "^-", color=c, lw=1, label=f"{name}: ODE t=.25→1")
    ax.set_xlabel(r"P = $W_2(p_{\hat X}, p_X)$ (sample estimate)")
    ax.set_ylabel(r"D = $E\|\hat X - X\|^2$")
    ax.set_title(f"Non-separable 2-D toy, Δ={res['delta']}", fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out / "C_dp.png", dpi=160)
    plt.close(fig)

    keys = [("X", "X"), ("X*", "X* (MMSE)"), ("OT(X*)", "OT partner (discrete)"),
            ("natural__euler1", "natural, 1 Euler"), ("natural__ode", "natural, ODE"),
            ("independent__ode", "independent, ODE"), ("minibatch_ot_256__euler1", "minibatch OT 256, 1 Euler"),
            ("near_exact_ot_2048__euler1", "near-exact OT, 1 Euler")]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.8), sharex=True, sharey=True)
    for a, (k, t) in zip(axs.flat, keys):
        v = samples[k.replace("/", "__")] if k in samples else samples[k.replace("__", "/")]
        a.scatter(v[:1500, 0], v[:1500, 1], s=2, alpha=0.5)
        a.set_title(t, fontsize=9)
        a.set_aspect("equal")
    axs[0, 0].set_xlim(-3, 3)
    axs[0, 0].set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig(out / "C_scatter.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
