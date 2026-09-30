# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "torch", "matplotlib"]
# ///
"""Toy Gaussian check for the Rate-as-Time Flow Bridge plan (Prop. 1-2, Thm. 3-4).

Setup. X ~ N(0, diag(sigma^2)) in R^d (quantization in the KLT basis, as in transform
coding). Subtractively dithered uniform quantizer with step Delta:
    Q = round((X + U) / Delta),  U ~ Unif(-Delta/2, Delta/2)^d shared by enc/dec,
    Y = Delta * Q - U = X + E,   E ~ Unif(-Delta/2, Delta/2)^d independent of X.
Everything factorizes over coordinates, so all "theory" quantities are exact (1-D quadrature):
    X* = E[X|Y]              truncated-normal mean on [Y - Delta/2, Y + Delta/2]
    D* = E||X - X*||^2       E[Var(X|Y)]
    T  = Brenier map p_X* -> p_X, coordinatewise monotone: T(X*) = sigma * Phi^{-1}(F_Y(Y))
    W  = W2(p_X*, p_X),      D(P) = D* + (W - P)_+^2   (Freirich et al. 2021)
Note: p_X* is NOT Gaussian (uniform noise), so the Bures closed form does not apply here.

Empirical part: learn a flow-matching velocity v(x, t) on the bridge X* -> X with four couplings
(natural (X*, X), independent, minibatch OT, exact OT), decode with 1 Euler step and with a
multi-step ODE, and place every decoder on the D-P plane next to the theoretical curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.special import log_ndtr, ndtr, ndtri

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 (e.g. Kaggle images) has only trapz


# ----------------------------------------------------------------------------- exact 1-D pieces
def _trunc_moments(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of a standard normal truncated to [a, b] (a < b), numerically stable."""
    flip = a > 0  # use symmetry so that the interval is never deep in the right tail
    a2, b2 = np.where(flip, -b, a), np.where(flip, -a, b)
    lb, la = log_ndtr(b2), log_ndtr(a2)
    log_z = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    ra = np.exp(-0.5 * a2**2 - LOG_SQRT_2PI - log_z)  # phi(a)/Z
    rb = np.exp(-0.5 * b2**2 - LOG_SQRT_2PI - log_z)  # phi(b)/Z
    mean = ra - rb
    var = 1.0 + a2 * ra - b2 * rb - mean**2
    return np.where(flip, -mean, mean), np.maximum(var, 0.0)


def mmse(y: np.ndarray, sigma: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """E[X|Y=y] and Var[X|Y=y], coordinatewise."""
    a, b = (y - delta / 2) / sigma, (y + delta / 2) / sigma
    m, v = _trunc_moments(a, b)
    return sigma * m, sigma**2 * v


def _G(z: np.ndarray) -> np.ndarray:  # antiderivative of Phi: z*Phi(z) + phi(z)
    return z * ndtr(z) + np.exp(-0.5 * z**2 - LOG_SQRT_2PI)


def cdf_y_left(y: np.ndarray, sigma: np.ndarray, delta: float) -> np.ndarray:
    """F_Y(y) for y <= 0, where Y = X + Unif(-delta/2, delta/2)."""
    return sigma / delta * (_G((y + delta / 2) / sigma) - _G((y - delta / 2) / sigma))


def ot_map(y: np.ndarray, sigma: np.ndarray, delta: float) -> np.ndarray:
    """T(X*) expressed through y: sigma * Phi^{-1}(F_Y(y)). Odd in y, computed on the left tail."""
    ay = -np.abs(y)
    q = np.clip(cdf_y_left(ay, sigma, delta), 1e-300, 0.5)
    return -np.sign(y) * sigma * ndtri(q)


def pdf_y(y: np.ndarray, sigma: np.ndarray, delta: float) -> np.ndarray:
    return (ndtr((y + delta / 2) / sigma) - ndtr((y - delta / 2) / sigma)) / delta


def theory(sigma: np.ndarray, delta: float, n_grid: int = 40001) -> dict:
    """Exact D*, W^2 and the dithered rate H(Q|U), summed over coordinates."""
    d_star, w2, rate_bits = 0.0, 0.0, 0.0
    for s in sigma:
        s_ = np.array([s])
        L = 9 * s + delta
        y = np.linspace(-L, L, n_grid)
        p = pdf_y(y, s_, delta)
        m, v = mmse(y, s_, delta)
        t = ot_map(y, s_, delta)
        d_star += _trapz(p * v, y)
        w2 += _trapz(p * (t - m) ** 2, y)
        # H(Q | U): entropy of the quantization index given the shared dither
        us = np.linspace(-delta / 2, delta / 2, 401)
        kmax = int(np.ceil((9 * s) / delta)) + 2
        ks = np.arange(-kmax, kmax + 1)
        hi = (ks[None, :] * delta + delta / 2 - us[:, None]) / s
        lo = (ks[None, :] * delta - delta / 2 - us[:, None]) / s
        pk = np.clip(ndtr(hi) - ndtr(lo), 1e-300, 1.0)
        rate_bits += float(np.mean(-(pk * np.log2(pk)).sum(1)))
    return {"D_star": float(d_star), "W2": float(w2), "W": float(np.sqrt(w2)), "rate_bits": rate_bits}


# ----------------------------------------------------------------------------- sampling
def sample(n: int, sigma: np.ndarray, delta: float, rng: np.random.Generator):
    d = len(sigma)
    x = rng.standard_normal((n, d)) * sigma
    u = rng.uniform(-delta / 2, delta / 2, (n, d))
    y = delta * np.round((x + u) / delta) - u
    xs, _ = mmse(y, sigma, delta)
    return x, y, xs


# ----------------------------------------------------------------------------- metrics
def w2_empirical(a: np.ndarray, b: np.ndarray) -> float:
    """Exact W2 between two equal-size empirical measures (assignment problem)."""
    a_t, b_t = torch.from_numpy(a), torch.from_numpy(b)
    c = torch.cdist(a_t, b_t).pow(2).numpy()
    r, k = linear_sum_assignment(c)
    return float(np.sqrt(c[r, k].mean()))


def w2_marginal_lb(xhat: np.ndarray, sigma: np.ndarray) -> float:
    """sqrt(sum_i W2^2(law of xhat_i, N(0, sigma_i^2))) <= W2(p_xhat, p_X); low-variance estimate."""
    n = xhat.shape[0]
    q = sigma[None, :] * ndtri((np.arange(1, n + 1)[:, None] - 0.5) / n)
    return float(np.sqrt(((np.sort(xhat, 0) - q) ** 2).mean(0).sum()))


def dp_point(xhat: np.ndarray, x: np.ndarray, x_ref: np.ndarray, sigma: np.ndarray) -> dict:
    """Distortion E||xhat - X||^2 and perception W2(p_xhat, p_X): the sample estimate against an
    independent X sample (biased upward, compare with `w2_floor`) and the marginal lower bound."""
    return {"D": float(((xhat - x) ** 2).sum(1).mean()), "P": w2_empirical(xhat, x_ref),
            "P_marg_lb": w2_marginal_lb(xhat, sigma)}


# ----------------------------------------------------------------------------- flow matching
class VelocityMLP(nn.Module):
    def __init__(self, d: int, width: int = 512, depth: int = 4, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n_freq + 1).float() * np.pi)
        layers, h = [], d + 2 * n_freq
        for _ in range(depth):
            layers += [nn.Linear(h, width), nn.SiLU()]
            h = width
        layers.append(nn.Linear(h, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tf = t[:, None] * self.freq[None, :]
        return self.net(torch.cat([x, torch.sin(tf), torch.cos(tf)], 1))


def make_pairs(coupling: str, n: int, sigma, delta, rng):
    x, y, xs = sample(n, sigma, delta, rng)
    if coupling == "natural":  # the true joint law (X*, X): rectified-flow setting of Thm. 4
        return xs, x
    if coupling == "exact_ot":  # (X*, T(X*)): Thm. 3
        return xs, ot_map(y, sigma, delta)
    x_ind, _, _ = sample(n, sigma, delta, rng)
    if coupling == "independent":
        return xs, x_ind
    if coupling == "minibatch_ot":
        c = ((xs[:, None, :] - x_ind[None, :, :]) ** 2).sum(-1)
        r, k = linear_sum_assignment(c)
        return xs[r], x_ind[k]
    raise ValueError(coupling)


def train_flow(coupling, sigma, delta, steps, batch, seed, log_every=1000):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    d = len(sigma)
    model = VelocityMLP(d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        x0, x1 = make_pairs(coupling, batch, sigma, delta, rng)
        x0, x1 = torch.from_numpy(x0).float(), torch.from_numpy(x1).float()
        t = torch.rand(batch)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % log_every == 0:
            print(f"    [{coupling}] step {it + 1}/{steps} loss {loss.item():.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def integrate(model, x0: np.ndarray, t_end: float, n_steps: int) -> np.ndarray:
    x = torch.from_numpy(x0).float()
    if n_steps == 0 or t_end == 0:
        return x0.copy()
    h = t_end / n_steps
    for k in range(n_steps):  # midpoint (RK2) for multi-step, plain Euler when n_steps == 1
        t = torch.full((x.shape[0],), k * h)
        if n_steps == 1:
            x = x + h * model(x, t)
        else:
            xm = x + 0.5 * h * model(x, t)
            x = x + h * model(xm, t + 0.5 * h)
    return x.double().numpy()


# ----------------------------------------------------------------------------- experiment
def run(d, delta, sigma, n_eval, steps, batch, seed, out_dir: Path, tag: str):
    t0 = time.time()
    th = theory(sigma, delta)
    rng = np.random.default_rng(seed + 12345)
    x, y, xs = sample(n_eval, sigma, delta, rng)
    x_ref, _, _ = sample(n_eval, sigma, delta, rng)
    x_ref2, _, _ = sample(n_eval, sigma, delta, rng)
    tx = ot_map(y, sigma, delta)
    res = {"tag": tag, "d": d, "delta": delta, "sigma": sigma.tolist(), "theory": th}
    res["w2_floor"] = w2_empirical(x_ref2, x_ref)  # finite-sample bias of the W2 estimator
    res["mc_check"] = {
        "D_star_mc": float(((xs - x) ** 2).sum(1).mean()),
        "W2_mc_paired": float(((tx - xs) ** 2).sum(1).mean()),
    }
    print(f"[{tag}] theory: {th}  |  MC: {res['mc_check']}  |  W2 floor {res['w2_floor']:.4f}")

    # Exact OT geodesic (Thm. 3): measured vs. D* + t^2 W^2, (1-t) W
    ts = np.linspace(0, 1, 11)
    res["exact_path"] = []
    for t in ts:
        xh = (1 - t) * xs + t * tx
        res["exact_path"].append({"t": float(t), **dp_point(xh, x, x_ref, sigma),
                                  "D_theory": th["D_star"] + t**2 * th["W2"],
                                  "P_theory": (1 - t) * th["W"]})

    # Learned flows
    res["flows"] = {}
    samples_for_plot = {"X": x[:2000], "X*": xs[:2000], "T(X*)": tx[:2000]}
    for coupling in ["natural", "independent", "minibatch_ot", "exact_ot"]:
        print(f"  training {coupling} ...", flush=True)
        model = train_flow(coupling, sigma, delta, steps, batch, seed)
        entry = {}
        with torch.no_grad():  # Prop. 2: velocity at t = 0
            v0 = model(torch.from_numpy(xs).float(), torch.zeros(len(xs))).double().numpy()
        entry["mean_sq_v0"] = float((v0**2).sum(1).mean())
        for name, t_end, n_steps in [("euler1", 1.0, 1), ("ode", 1.0, 50),
                                     ("ode_t0.25", 0.25, 13), ("ode_t0.5", 0.5, 25),
                                     ("ode_t0.75", 0.75, 38)]:
            xh = integrate(model, xs, t_end, n_steps)
            entry[name] = {**dp_point(xh, x, x_ref, sigma),
                           "dist_to_Xstar": float(((xh - xs) ** 2).sum(1).mean())}
            if name in ("euler1", "ode"):
                samples_for_plot[f"{coupling}/{name}"] = xh[:2000]
        res["flows"][coupling] = entry
        print(f"  {coupling}: " + ", ".join(
            f"{k}: D={v['D']:.4f} P={v['P']:.4f} Plb={v['P_marg_lb']:.4f}"
            for k, v in entry.items() if isinstance(v, dict)), flush=True)
    res["seconds"] = time.time() - t0
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out_dir / f"{tag}_samples.npz", **{k.replace("/", "__"): v for k, v in samples_for_plot.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n_eval", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default="", help="comma list of tags to run: d2,d64,sweep")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))

    sigma64 = (np.arange(1, 65) ** -0.5).astype(np.float64)  # decaying KLT spectrum, 1 -> 0.125
    if not only or "sweep" in only:  # theory only: D*, W, rate across step sizes
        rows = []
        for delta in [0.25, 0.5, 1.0, 2.0, 4.0]:
            th = theory(sigma64, delta)
            rows.append({"delta": delta, **th, "bits_per_dim": th["rate_bits"] / 64,
                         "D_star_over_var": th["D_star"] / float((sigma64**2).sum())})
            print("sweep", rows[-1])
        (out / "sweep_d64.json").write_text(json.dumps(rows, indent=2))
    if not only or "d2" in only:
        # coarse step so that W (0.47) is well above the 2-D sample-W2 bias
        run(2, 4.0, np.array([1.0, 0.5]), args.n_eval, args.steps, args.batch, args.seed, out, "d2_delta4")
    if not only or "d64" in only:
        run(64, 2.0, sigma64, args.n_eval, args.steps, args.batch, args.seed, out, "d64_delta2")  # ~0.3 bit/dim


if __name__ == "__main__":
    main()
