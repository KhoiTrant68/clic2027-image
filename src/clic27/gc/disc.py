"""Conditional multi-scale PatchGAN discriminator (HiFiC-style: conditioned on the decoded latent)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm as sn


class PatchD(nn.Module):
    def __init__(self, M, cond_ch=12, width=64):
        super().__init__()
        self.proj = nn.Conv2d(M, cond_ch, 1)
        w = width
        self.net = nn.Sequential(
            sn(nn.Conv2d(3 + cond_ch, w, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(w, 2 * w, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(2 * w, 4 * w, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(4 * w, 8 * w, 4, 1, 1)), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * w, 1, 3, 1, 1),
        )

    def forward(self, x, y_hat):
        c = F.interpolate(self.proj(y_hat), size=x.shape[-2:], mode="nearest")
        return self.net(torch.cat([x * 2 - 1, c], 1))


class MultiScaleD(nn.Module):
    """Two PatchGANs at full and half resolution; returns a list of logit maps."""

    def __init__(self, M, scales=2, width=64):
        super().__init__()
        self.ds = nn.ModuleList([PatchD(M, width=width) for _ in range(scales)])

    def forward(self, x, y_hat):
        out = []
        for i, d in enumerate(self.ds):
            xi = x if i == 0 else F.avg_pool2d(x, 2 ** i)
            out.append(d(xi, y_hat.detach()))
        return out


def d_loss(real_logits, fake_logits):
    return sum(F.softplus(-r).mean() + F.softplus(f).mean() for r, f in zip(real_logits, fake_logits))


def g_loss(fake_logits, weight=None):
    """Non-saturating generator loss; `weight` (B,) multiplies per-sample terms (realism beta)."""
    tot = 0.0
    for f in fake_logits:
        per = F.softplus(-f).flatten(1).mean(1)
        tot = tot + (per * weight if weight is not None else per).mean()
    return tot
