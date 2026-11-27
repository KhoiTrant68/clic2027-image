"""Float building blocks for the analysis / synthesis transforms (decoder output need not be bit-exact)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv(cin, cout, k=3, stride=1):
    return nn.Conv2d(cin, cout, k, stride, padding=k // 2)


class FiLM(nn.Module):
    """Per-channel scale/shift from a conditioning vector; initialised to the identity."""

    def __init__(self, cond_dim, ch):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):
        g, b = self.lin(c).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class ResBlock(nn.Module):
    """Pre-activation residual block with optional FiLM."""

    def __init__(self, ch, cond_dim=0):
        super().__init__()
        self.c1, self.c2 = conv(ch, ch), conv(ch, ch)
        self.film = FiLM(cond_dim, ch) if cond_dim else None
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, c=None):
        h = self.c1(F.leaky_relu(x, 0.2))
        if self.film is not None:
            h = self.film(h, c)
        return x + self.c2(F.leaky_relu(h, 0.2))


class ResBottleneck(nn.Module):
    """ELIC residual bottleneck (1x1 -> 3x3 -> 1x1)."""

    def __init__(self, ch):
        super().__init__()
        m = ch // 2
        self.net = nn.Sequential(conv(ch, m, 1), nn.ReLU(True), conv(m, m, 3), nn.ReLU(True), conv(m, ch, 1))

    def forward(self, x):
        return x + self.net(x)


class Up(nn.Module):
    """x2 sub-pixel upsampling."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = conv(cin, cout * 4, 3)

    def forward(self, x):
        return F.pixel_shuffle(self.conv(x), 2)


class CondEmbed(nn.Module):
    """Embedding of (rate level, realism beta) -> conditioning vector."""

    def __init__(self, n_levels, dim=128):
        super().__init__()
        self.level = nn.Embedding(n_levels, dim)
        self.beta = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))
        self.dim = dim

    def forward(self, level, beta=None):
        e = self.level(level)
        if beta is not None:
            e = e + self.beta(beta.view(-1, 1).float())
        return self.out(e)


class Analysis(nn.Module):
    """g_a: 3 -> M at 1/16 resolution (ELIC layout without attention)."""

    def __init__(self, N, M, blocks, cond_dim):
        super().__init__()
        self.stem = conv(3, N, 5, 2)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(3):
            self.stages.append(nn.Sequential(*[ResBottleneck(N) for _ in range(blocks)]))
            self.downs.append(conv(N, N if i < 2 else M, 5, 2))
        self.film = nn.ModuleList([FiLM(cond_dim, N) for _ in range(3)])

    def forward(self, x, c):
        h = self.stem(x)
        for st, dn, fm in zip(self.stages, self.downs, self.film):
            h = dn(fm(st(h), c))
        return h


class Synthesis(nn.Module):
    """g_s: (M latent + 1 quality-map channel) -> RGB, FiLM-conditioned on (level, beta) at every block."""

    def __init__(self, M, width, blocks, cond_dim):
        super().__init__()
        self.inp = conv(M + 1, width * 2, 3)
        chans = [width * 2, width * 2, width, width // 2, width // 2]
        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, nb in enumerate(blocks):
            self.stages.append(nn.ModuleList([ResBlock(chans[i], cond_dim) for _ in range(nb)]))
            self.ups.append(Up(chans[i], chans[i + 1]))
        self.out = conv(chans[-1], 3, 3)

    def forward(self, y, qmap, c):
        h = self.inp(torch.cat([y, qmap], dim=1))
        for blocks, up in zip(self.stages, self.ups):
            for b in blocks:
                h = b(h, c)
            h = up(h)
        return self.out(F.leaky_relu(h, 0.2))


class HyperAnalysis(nn.Module):
    """h_a: latent -> hyper-latent at 1/4 of the latent resolution (float, encoder only)."""

    def __init__(self, M, Z):
        super().__init__()
        self.net = nn.Sequential(conv(M, Z, 3), nn.LeakyReLU(0.2, True), conv(Z, Z, 5, 2),
                                 nn.LeakyReLU(0.2, True), conv(Z, Z, 5, 2))

    def forward(self, y):
        return self.net(y)


# ---------------------------------------------------------------- g_s v2 (plans/GC_v2_architecture_2026-11-23.md 4.3)
class UpHaar(nn.Module):
    """x2 upsampling by conv -> inverse Haar (ablation of pixel shuffle). The conv predicts 4 sub-bands (LL, LH, HL,
    HH) per output channel; the fixed orthonormal 2x2 inverse transform places them. Same expressive power as
    pixel shuffle (a fixed invertible 4-channel mix); any gain would come from optimisation, so it is measured."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = conv(cin, cout * 4, 3)

    def forward(self, x):
        ll, lh, hl, hh = self.conv(x).chunk(4, dim=1)
        x00, x01 = (ll + lh + hl + hh) / 2, (ll - lh + hl - hh) / 2
        x10, x11 = (ll + lh - hl - hh) / 2, (ll - lh - hl + hh) / 2
        B, C, H, W = ll.shape
        s = torch.stack([x00, x01, x10, x11], dim=2).reshape(B, C * 4, H, W)  # per channel: (0,0) (0,1) (1,0) (1,1)
        return F.pixel_shuffle(s, 2)


class WindowAttn(nn.Module):
    """Pre-norm multi-head self-attention in non-overlapping windows (shifted by half a window when `shift`),
    zero-initialised output projection (identity at init). Float only: the synthesis need not be bit-exact."""

    def __init__(self, ch, heads=8, window=8, shift=False):
        super().__init__()
        self.h, self.w, self.shift = heads, window, shift
        self.norm = nn.GroupNorm(1, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        ph, pw = (-H) % w, (-W) % w
        q = self.qkv(self.norm(x))
        if ph or pw:
            q = F.pad(q, (0, pw, 0, ph))
        if self.shift:
            q = torch.roll(q, (-(w // 2), -(w // 2)), (2, 3))
        Hp, Wp = H + ph, W + pw
        d = C // self.h
        t = q.reshape(B, 3, self.h, d, Hp // w, w, Wp // w, w).permute(1, 0, 4, 6, 2, 5, 7, 3)
        t = t.reshape(3, B * (Hp // w) * (Wp // w), self.h, w * w, d)
        o = F.scaled_dot_product_attention(t[0], t[1], t[2])  # (Bn, h, w*w, d)
        o = o.reshape(B, Hp // w, Wp // w, self.h, w, w, d).permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        if self.shift:
            o = torch.roll(o, (w // 2, w // 2), (2, 3))
        return x + self.proj(o[:, :, :H, :W])


class SynthesisV2(nn.Module):
    """g_s v2: capacity moved to 1/16 and 1/8 (cheap per pixel: a 384-ch block at 1/16 is ~10 kMAC/px), window
    attention at 1/16, thin fine stages (memory-bound on the L4). Same interface as Synthesis.

    blocks: residual blocks per stage (1/16, 1/8, 1/4, 1/2); attn: window-attention layers interleaved in the 1/16
    stage; up: "shuffle" | "haar". width scales every stage (chosen by L4 timing, step C)."""

    def __init__(self, M, width, cond_dim, blocks=(6, 4, 2, 2), attn=2, window=8, up="shuffle"):
        super().__init__()
        chans = [width * 2, width * 2, width, width // 2, width // 2]
        Upc = {"shuffle": Up, "haar": UpHaar}[up]
        self.inp = conv(M + 1, chans[0], 3)
        self.stages, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, nb in enumerate(blocks):
            st = nn.ModuleList([ResBlock(chans[i], cond_dim) for _ in range(nb)])
            self.stages.append(st)
            self.ups.append(Upc(chans[i], chans[i + 1]))
        every = max(1, blocks[0] // max(1, attn)) if attn else 0
        self.attn = nn.ModuleList([WindowAttn(chans[0], 8, window, shift=bool(j % 2)) for j in range(attn)])
        self.attn_after = [min(blocks[0], (j + 1) * every) - 1 for j in range(attn)]  # block index in stage 0
        self.out = conv(chans[-1], 3, 3)

    def forward(self, y, qmap, c):
        h = self.inp(torch.cat([y, qmap], dim=1))
        for i, (blocks, up) in enumerate(zip(self.stages, self.ups)):
            for j, b in enumerate(blocks):
                h = b(h, c)
                if i == 0:
                    for a, at in zip(self.attn, self.attn_after):
                        if at == j:
                            h = a(h)
            h = up(h)
        return self.out(F.leaky_relu(h, 0.2))
