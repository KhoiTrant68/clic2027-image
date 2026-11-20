"""Encoder side: quality-map search, optional latent refinement, corpus byte allocation.

Raters judge 512x512 crops, so one bad region can lose a comparison. For each image and rate level we walk a
greedy trajectory of quality maps: at every step the cells under the worst-scoring crop get a finer quantiser
(+1) and the cells under the best crops a coarser one (-1). Every point of every trajectory is a candidate
(estimated bytes, crop scores). A multiple-choice knapsack then picks one candidate per image so that the
corpus objective (mean / max / mean + kappa * max of crop distances, summed over images) is minimal under the
byte budget. The chosen candidates are entropy-coded for real and the budget is re-checked on actual bytes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GCodec, pad64


# ---------------------------------------------------------------- crop scoring
def crop_boxes(H, W, crop=512, stride=256):
    """Top-left corners of crop x crop windows covering the image (last row/column flush with the border)."""
    ys = sorted({min(y, max(H - crop, 0)) for y in range(0, max(H - crop, 0) + stride, stride)})
    xs = sorted({min(x, max(W - crop, 0)) for x in range(0, max(W - crop, 0) + stride, stride)})
    return [(y, x, min(crop, H), min(crop, W)) for y in ys for x in xs]


class CropScorer:
    """Distance per crop; uses the Q-hat v2 reward if given, else LPIPS-Alex."""

    def __init__(self, dev, reward=None, crop=512, stride=256):
        self.dev, self.crop, self.stride = dev, crop, stride
        if reward is not None:
            self.fn = reward
        else:
            from .losses import LPIPS
            self.fn = LPIPS().to(dev)

    @torch.no_grad()
    def __call__(self, o: torch.Tensor, x: torch.Tensor, batch=8) -> np.ndarray:
        """o, x: (1, 3, H, W) in [0, 1]. Returns distances (n_crops,)."""
        H, W = o.shape[-2:]
        boxes = crop_boxes(H, W, self.crop, self.stride)
        out = []
        for i in range(0, len(boxes), batch):
            bs = boxes[i:i + batch]
            oc = torch.cat([o[..., y:y + h, xx:xx + w] for y, xx, h, w in bs])
            xc = torch.cat([x[..., y:y + h, xx:xx + w] for y, xx, h, w in bs])
            out.append(self.fn(oc.to(self.dev), xc.to(self.dev).float()).float().cpu())
        return torch.cat(out).numpy()


def aggregate(d: np.ndarray, how: str, kappa: float) -> float:
    if how == "mean":
        return float(d.mean())
    if how == "max":
        return float(d.max())
    return float(d.mean() + kappa * d.max())


# ---------------------------------------------------------------- candidates
@dataclass
class Cand:
    level: int
    beta: int
    cells: np.ndarray
    est_bytes: float
    crop_d: np.ndarray
    obj: float = 0.0
    meta: dict = field(default_factory=dict)


@torch.no_grad()
def estimate(model: GCodec, xp, level, beta_idx, cells, dev):
    lv = torch.tensor([level], device=dev)
    b = model.beta_value(torch.tensor([beta_idx], device=dev))
    with torch.autocast("cuda", dtype=torch.bfloat16) if str(dev).startswith("cuda") else torch.autocast("cpu",
                                                                                                         enabled=False):
        o = model(xp, lv, b, torch.as_tensor(cells, device=dev).view(1, 1, *cells.shape), noise=False)
    return float(o["bits"]) / 8, o["x_hat"].float().clamp(0, 1)


def qmap_trajectory(model: GCodec, x, level, beta_idx, scorer: CropScorer, iters: int, how: str, kappa: float,
                    dev) -> list[Cand]:
    """x: (1, 3, H, W) in [0, 1] on dev. Returns iters + 1 candidates along the greedy worst-crop walk."""
    xp, (H, W) = pad64(x)
    ch, cw = model.cell_shape(*xp.shape[-2:])
    qb, R = model.qmap_block, model.qmap_range
    cells = np.zeros((ch, cw), np.int64)
    boxes = crop_boxes(H, W, scorer.crop, scorer.stride)
    out = []
    for it in range(iters + 1):
        est, xh = estimate(model, xp, level, beta_idx, cells, dev)
        d = scorer(x, xh[..., :H, :W])
        out.append(Cand(level, beta_idx, cells.copy(), est, d, aggregate(d, how, kappa), dict(iter=it)))
        if it == iters or len(boxes) < 2:
            break
        order = np.argsort(d)
        y, xx, h, w = boxes[order[-1]]
        cells[y // qb:-(-(y + h) // qb), xx // qb:-(-(xx + w) // qb)] += 1
        for j in order[: max(1, len(order) // 4)]:
            y, xx, h, w = boxes[j]
            cells[y // qb:-(-(y + h) // qb), xx // qb:-(-(xx + w) // qb)] -= 1
        np.clip(cells, -R, R, out=cells)
    return out


# ---------------------------------------------------------------- latent refinement (optional)
def refine_latent(model: GCodec, xp, level, beta_idx, cells, steps, lr, lam, lp=None, k_mse=152.0):
    """Encoder-side latent optimisation (decoder unchanged): minimise D + lam * bpp over y_s with STE rounding."""
    dev = xp.device
    lv = torch.tensor([level], device=dev)
    b = model.beta_value(torch.tensor([beta_idx], device=dev))
    cl = torch.as_tensor(cells, device=dev).view(1, 1, *cells.shape)
    with torch.no_grad():
        y = model.analysis(xp, lv, cl).float()
    y = y.clone().requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    n = xp.shape[-1] * xp.shape[-2]
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        o = model.from_latent(y, lv, b, cl, noise=False)
        d = ((o["x_hat"] - xp) ** 2).mean() * k_mse
        if lp is not None:
            d = d + lp(o["x_hat"], xp).mean()
        loss = d + lam * o["bits"].sum() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return y.detach()


# ---------------------------------------------------------------- multiple-choice knapsack
def knapsack(cands: list[list[Cand]], budget_bytes: float, overhead, units: int = 6000):
    """Pick one candidate per image minimising sum(obj) with sum(est_bytes + overhead_i) <= budget.
    overhead: bytes per image (scalar or list). Returns list of chosen indices, or None if infeasible."""
    n = len(cands)
    ovs = list(overhead) if hasattr(overhead, "__len__") else [overhead] * n
    u = budget_bytes / units
    INF = 1e18
    dp = np.full(units + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((n, units + 1), np.int32)
    for i, cs in enumerate(cands):
        new = np.full(units + 1, INF)
        arg = np.zeros(units + 1, np.int32)
        for j, c in enumerate(cs):
            w = int(math.ceil((c.est_bytes + ovs[i]) / u))
            if w > units:
                continue
            cand = np.full(units + 1, INF)
            cand[w:] = dp[: units + 1 - w] + c.obj
            better = cand < new
            new[better], arg[better] = cand[better], j
        dp, choice[i] = new, arg
    best = int(np.argmin(dp))
    if dp[best] >= INF:
        return None
    picks, b = [], best
    for i in range(n - 1, -1, -1):
        j = int(choice[i, b])
        picks.append(j)
        b -= int(math.ceil((cands[i][j].est_bytes + ovs[i]) / u))
    return picks[::-1]
