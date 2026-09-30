# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "torch", "matplotlib"]
# ///
"""Round-2 toy experiments A and B (GPU-ready). Reuses the exact pieces of toy_dp.py.

A — why does the one-step decoder fail at d=64 even with the exact OT coupling?
    A1 baseline (6k steps, width 512)       A2 capacity (24k steps, width 1024)
    A3 preconditioned (6k, 512)             A4 capacity + preconditioning (24k, 1024)
    A5/A6 start noise (6k, 512): decode from X* + eps, eps_i ~ N(0, (c * s_i)^2) with s_i^2 = E Var(X_i|Y)
    (c = 0.25 / 0.5 / 1.0; A7 = 0.25), trained on the exact OT map of the NOISY source (per-coordinate monotone
    rearrangement from empirical quantiles). Tests whether spreading the near-degenerate source makes
    the map learnable, and at what distortion price (the "oracle" row = the noisy OT map itself).
    Preconditioning: the network sees x / s(t) with s(t) = (1-t) std(X*) + t sigma per coordinate,
    and its output is multiplied by sigma, so every coordinate is O(1) for the network.
B — how large must the minibatch be for minibatch-OT to still help as d grows?
    d in {16, 64}; coupling pool size B with exact assignment (B <= 1024) or entropic OT
    (log-domain Sinkhorn on GPU, pairs sampled from the plan) for B in {1024, 4096, 16384};
    anchors: independent and exact-OT couplings. The training batch is always 1024; the pool of
    B coupled pairs is consumed in chunks of 1024, so only the coupling quality changes with B.

Metrics (sample-W2 is useless at d >= 16, see round 1): distortion D, D / (D* + W^2), the
marginal lower bound on P = W2(p_xhat, p_X) and its split over coordinate groups.

Usage on one GPU:
    uv run toy_ab.py --exp A --device cuda --out results_A
    uv run toy_ab.py --exp B --device cuda --out results_B
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
from scipy.special import ndtri

from toy_dp import VelocityMLP, mmse, ot_map, sample, theory

DELTA = 2.0


def spectrum(d: int) -> np.ndarray:
    return (np.arange(1, d + 1) ** -0.5).astype(np.float64)


# ----------------------------------------------------------------------------- models
class Preconditioned(nn.Module):
    def __init__(self, d, width, s0: np.ndarray, s1: np.ndarray):
        super().__init__()
        self.net = VelocityMLP(d, width=width)
        self.register_buffer("s0", torch.tensor(s0, dtype=torch.float32))
        self.register_buffer("s1", torch.tensor(s1, dtype=torch.float32))

    def forward(self, x, t):
        s = (1 - t[:, None]) * self.s0 + t[:, None] * self.s1
        return self.net(x / s, t) * self.s1


# ----------------------------------------------------------------------------- start noise
class NoisyOT:
    """Source X* + eps (product law) and its exact OT map to p_X: coordinatewise monotone rearrangement."""

    def __init__(self, sigma, c, rng, n=200_000):
        _, y, xs = sample(n, sigma, DELTA, rng)
        _, var = mmse(y, sigma, DELTA)
        self.sigma, self.scale = sigma, c * np.sqrt(var.mean(0))
        self.table = np.sort(self.add_noise(xs, rng), 0)
        self.grid = (np.arange(n) + 0.5) / n

    def add_noise(self, xs, rng):
        return xs + rng.standard_normal(xs.shape) * self.scale

    def map(self, v):
        q = np.stack([np.interp(v[:, i], self.table[:, i], self.grid) for i in range(v.shape[1])], 1)
        return self.sigma * ndtri(q)


# ----------------------------------------------------------------------------- couplings
def sinkhorn_pairs(a: torch.Tensor, b: torch.Tensor, eps_rel: float, n_iter: int, gen: torch.Generator):
    """Entropic OT between two equal-size clouds (log domain); one partner sampled per row."""
    c = torch.cdist(a, b).pow(2)
    eps = eps_rel * c.mean()
    n = a.shape[0]
    log_w = -np.log(n)
    f = torch.zeros(n, device=a.device)
    g = torch.zeros(n, device=a.device)
    for _ in range(n_iter):
        f = eps * log_w - eps * torch.logsumexp((g[None, :] - c) / eps, dim=1)
        g = eps * log_w - eps * torch.logsumexp((f[:, None] - c) / eps, dim=0)
    logp = (f[:, None] + g[None, :] - c) / eps
    col_err = (logp.exp().sum(0) * n - 1).abs().max().item()  # row sums are exact after the f-update
    j = torch.multinomial(torch.softmax(logp, dim=1), 1, generator=gen).squeeze(1)
    return j, col_err


class PairStream:
    """Yields (x0, x1) training batches of size `batch` for a given coupling and pool size."""

    def __init__(self, coupling, sigma, batch, pool, rng, device, eps_rel=0.01, n_iter=300, noisy=None):
        self.coupling, self.sigma, self.batch, self.pool, self.noisy = coupling, sigma, batch, pool, noisy
        self.rng, self.device, self.eps_rel, self.n_iter = rng, device, eps_rel, n_iter
        self.gen = torch.Generator(device=device).manual_seed(int(rng.integers(1 << 31)))
        self.buf0, self.buf1 = [], []
        self.n_buf, self.sinkhorn_col_err, self.pair_cost = 0, [], []

    def _block(self):
        x0, x1 = self._coupled_block()
        self.pair_cost.append(float(((x0 - x1) ** 2).sum(1).mean()))
        return x0, x1

    def _coupled_block(self):
        n = self.batch if self.coupling in ("natural", "independent", "exact_ot", "noisy_ot") else self.pool
        _, y, xs = sample(n, self.sigma, DELTA, self.rng)
        if self.coupling == "exact_ot":
            return xs, ot_map(y, self.sigma, DELTA)
        if self.coupling == "noisy_ot":
            x0 = self.noisy.add_noise(xs, self.rng)
            return x0, self.noisy.map(x0)
        x_ind, _, _ = sample(n, self.sigma, DELTA, self.rng)
        if self.coupling == "independent":
            return xs, x_ind
        if self.coupling == "lsa":
            c = ((xs[:, None, :] - x_ind[None, :, :]) ** 2).sum(-1)
            r, k = linear_sum_assignment(c)
            return xs[r], x_ind[k]
        if self.coupling == "sinkhorn":
            a = torch.from_numpy(xs).float().to(self.device)
            b = torch.from_numpy(x_ind).float().to(self.device)
            j, err = sinkhorn_pairs(a, b, self.eps_rel, self.n_iter, self.gen)
            self.sinkhorn_col_err.append(err)
            return xs, x_ind[j.cpu().numpy()]
        raise ValueError(self.coupling)

    def next(self):
        while self.n_buf < self.batch:
            x0, x1 = self._block()
            perm = self.rng.permutation(len(x0))  # decorrelate chunks taken from one pool
            self.buf0.append(x0[perm]); self.buf1.append(x1[perm]); self.n_buf += len(x0)
        x0, x1 = np.concatenate(self.buf0), np.concatenate(self.buf1)
        self.buf0, self.buf1 = [x0[self.batch:]], [x1[self.batch:]]
        self.n_buf -= self.batch
        to = lambda z: torch.from_numpy(z[: self.batch]).float().to(self.device)
        return to(x0), to(x1)


# ----------------------------------------------------------------------------- train / eval
def train(model, stream: PairStream, steps, device, tag, log_every=2000):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        x0, x1 = stream.next()
        t = torch.rand(x0.shape[0], device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % log_every == 0:
            print(f"    [{tag}] step {it + 1}/{steps} loss {loss.item():.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def integrate(model, x0, n_steps, device):
    x = torch.from_numpy(x0).float().to(device)
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * h, device=device)
        if n_steps == 1:
            x = x + h * model(x, t)
        else:
            xm = x + 0.5 * h * model(x, t)
            x = x + h * model(xm, t + 0.5 * h)
    return x.double().cpu().numpy()


def marginal_w2sq(xhat, sigma):
    n = xhat.shape[0]
    q = sigma[None, :] * ndtri((np.arange(1, n + 1)[:, None] - 0.5) / n)
    return ((np.sort(xhat, 0) - q) ** 2).mean(0)  # per coordinate


def groups(d):  # coordinate groups by variance rank: top 1/8, next 3/8, last 1/2
    a, b = max(1, d // 8), max(2, d // 2)
    return {"high_var": slice(0, a), "mid_var": slice(a, b), "low_var": slice(b, d)}


def evaluate(model, sigma, th, n_eval, seed, device, noisy=None):
    rng = np.random.default_rng(seed + 999)
    x, _, xs = sample(n_eval, sigma, DELTA, rng)
    oracle = None
    if noisy is not None:  # decode from X* + eps; the oracle is the exact noisy OT map
        xs = noisy.add_noise(xs, rng)
        oracle = noisy.map(xs)
    with torch.no_grad():
        xs_t = torch.from_numpy(xs).float().to(device)
        v0 = model(xs_t, torch.zeros(n_eval, device=device)).double().cpu().numpy()
    ref = th["D_star"] + th["W2"]
    out = {"mean_sq_v0": float((v0**2).sum(1).mean())}
    for name, k in [("euler1", 1), ("ode50", 50)]:
        xh = integrate(model, xs, k, device)
        w = marginal_w2sq(xh, sigma)
        D = float(((xh - x) ** 2).sum(1).mean())
        out[name] = {"D": D, "D_over_opt": D / ref, "P_marg_lb": float(np.sqrt(w.sum())),
                     **{f"P2_lb_{g}": float(w[s].sum()) for g, s in groups(len(sigma)).items()}}
    if oracle is not None:
        w = marginal_w2sq(oracle, sigma)
        D = float(((oracle - x) ** 2).sum(1).mean())
        out["oracle_noisy_map"] = {"D": D, "D_over_opt": D / ref, "P_marg_lb": float(np.sqrt(w.sum()))}
    w0 = marginal_w2sq(xs, sigma)
    out["reference_Xstar"] = {"P_marg_lb": float(np.sqrt(w0.sum())),
                              **{f"P2_lb_{g}": float(w0[s].sum()) for g, s in groups(len(sigma)).items()}}
    return out


def one_run(tag, d, coupling, pool, steps, width, precond, noise_c=0.0, *, args, out_dir):
    t0 = time.time()
    sigma = spectrum(d)
    th = theory(sigma, DELTA)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if precond:
        _, _, xs_big = sample(200_000, sigma, DELTA, np.random.default_rng(args.seed + 7))
        model = Preconditioned(d, width, xs_big.std(0), sigma)
    else:
        model = VelocityMLP(d, width=width)
    model = model.to(args.device)
    noisy = NoisyOT(sigma, noise_c, np.random.default_rng(args.seed + 5)) if coupling == "noisy_ot" else None
    stream = PairStream(coupling, sigma, args.batch, pool, rng, args.device, args.eps_rel, args.sk_iter, noisy)
    print(f"[{tag}] d={d} coupling={coupling} pool={pool} steps={steps} width={width} precond={precond}", flush=True)
    train(model, stream, steps, args.device, tag)
    res = {"tag": tag, "d": d, "delta": DELTA, "coupling": coupling, "pool": pool, "steps": steps,
           "width": width, "precond": precond, "noise_c": noise_c, "theory": th,
           "eval": evaluate(model, sigma, th, args.n_eval, args.seed, args.device, noisy),
           "seconds": time.time() - t0}
    res["pair_cost_over_W2"] = float(np.mean(stream.pair_cost)) / th["W2"]  # 1.0 = population OT
    if stream.sinkhorn_col_err:
        res["sinkhorn_col_marginal_err_max"] = float(np.max(stream.sinkhorn_col_err))
    e = res["eval"]
    print(f"[{tag}] euler1 D/opt={e['euler1']['D_over_opt']:.3f} Plb={e['euler1']['P_marg_lb']:.3f} | "
          f"ode50 D/opt={e['ode50']['D_over_opt']:.3f} Plb={e['ode50']['P_marg_lb']:.3f} | "
          f"X* Plb={e['reference_Xstar']['P_marg_lb']:.3f} | {res['seconds']:.0f}s"
          + (f" | oracle D/opt={e['oracle_noisy_map']['D_over_opt']:.3f} Plb={e['oracle_noisy_map']['P_marg_lb']:.3f}"
             if "oracle_noisy_map" in e else ""), flush=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def orc(r):
    o = r["eval"].get("oracle_noisy_map")
    return f"{o['D_over_opt']:.3f}, {o['P_marg_lb']:.3f}" if o else "—"


def table(rows):
    g = list(groups(64).keys())
    header = ("| run | d | coupling | pool | steps | width | precond | 1-step D/(D*+W²) | 1-step P_lb | "
              + " | ".join(f"1-step P²_lb {k}" for k in g) + " | ODE D/(D*+W²) | ODE P_lb | pair cost / W² | oracle D/(D*+W²), P_lb | time (s) |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in rows:
        e1, eo = r["eval"]["euler1"], r["eval"]["ode50"]
        lines.append(f"| {r['tag']} | {r['d']} | {r['coupling']} | {r['pool']} | {r['steps']} | {r['width']} | "
                     f"{r['precond']} | {e1['D_over_opt']:.3f} | {e1['P_marg_lb']:.3f} | "
                     + " | ".join(f"{e1['P2_lb_' + k]:.3f}" for k in g)
                     + f" | {eo['D_over_opt']:.3f} | {eo['P_marg_lb']:.3f} | {r['pair_cost_over_W2']:.2f} | {orc(r)} | {r['seconds']:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="", help="comma list of run tags")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n_eval", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_rel", type=float, default=0.01, help="Sinkhorn eps relative to mean cost")
    ap.add_argument("--sk_iter", type=int, default=300)
    ap.add_argument("--steps_scale", type=float, default=1.0, help="multiply all step counts (smoke tests)")
    args = ap.parse_args()
    out = Path(args.out or f"results_{args.exp}")
    out.mkdir(parents=True, exist_ok=True)
    s = lambda n: max(1, int(n * args.steps_scale))

    if args.exp == "A":  # (tag, d, coupling, pool, steps, width, precond)
        runs = [("A1_base", 64, "exact_ot", 0, s(6000), 512, False),
                ("A2_capacity", 64, "exact_ot", 0, s(24000), 1024, False),
                ("A3_precond", 64, "exact_ot", 0, s(6000), 512, True),
                ("A4_capacity_precond", 64, "exact_ot", 0, s(24000), 1024, True),
                ("A5_noise0.5", 64, "noisy_ot", 0, s(6000), 512, False, 0.5),
                ("A6_noise1.0", 64, "noisy_ot", 0, s(6000), 512, False, 1.0),
                ("A7_noise0.25", 64, "noisy_ot", 0, s(6000), 512, False, 0.25)]
    else:
        runs = []
        for d in (16, 64):
            runs += [(f"B_d{d}_independent", d, "independent", 0, s(6000), 512, False),
                     (f"B_d{d}_lsa256", d, "lsa", 256, s(6000), 512, False),
                     (f"B_d{d}_lsa1024", d, "lsa", 1024, s(6000), 512, False),
                     (f"B_d{d}_sk1024", d, "sinkhorn", 1024, s(6000), 512, False),
                     (f"B_d{d}_sk4096", d, "sinkhorn", 4096, s(6000), 512, False),
                     (f"B_d{d}_sk16384", d, "sinkhorn", 16384, s(6000), 512, False),
                     (f"B_d{d}_exact_ot", d, "exact_ot", 0, s(6000), 512, False)]
    only = set(filter(None, args.only.split(",")))
    rows = [one_run(*r, args=args, out_dir=out) for r in runs if not only or r[0] in only]
    md = table(rows)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
