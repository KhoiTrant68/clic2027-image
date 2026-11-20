"""GCodec: perceptual learned image codec with a bit-exact entropy model.

Signal path (B = batch, l = rate level, beta = realism, k = quality-map cell value):

    x --g_a--> y (M ch, 1/16)        y_s = y * gain[l] * 2**(k/4)           (gain units; encoder side, float)
    y_s --h_a--> z --round--> z_hat   coded with a learned per-channel pmf     (DiscretePrior)
    z_hat --h_s (integer net)--> hyper features at 1/16
    for slice s in uneven channel groups, for pass p in (anchor, non-anchor) of a checkerboard:
        integer net(hyper, previous slices, [anchors of this slice], k) -> (scale index, dyadic mean)
        q = round(y_s - mu) coded with rANS on integer CDF tables
    y_hat = y_hat_s / gain  --g_s(level, beta, k)--> x_hat

Everything that decides a CDF runs through clic27.entropy.intnet (integers carried in float64), so the encoder
machine and the CLIC L4 derive identical tables; the synthesis is ordinary float/fp16 and only affects pixels.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from clic27.entropy.intnet import IntSequential, QConv2d, Up2, check_exact_range, ste
from clic27.entropy import tables
from clic27.entropy.models import GaussianConditional, ScaleMeanHead, gaussian_bits

from .layers import Analysis, CondEmbed, HyperAnalysis, Synthesis

FRAC = 4            # fixed-point fractional bits of every integer-net input (hyper features, y_hat_s, k)
Y_CLAMP = 2047.0    # |y_hat_s| fed to context nets is clamped (bounds the integer accumulators)
Z_CLAMP = 2.0 ** 15  # |z_hat| fed to h_s is clamped
COND_DIM = 128


def checkerboard(h, w, device):
    """1 at anchor positions ((i + j) even), 0 elsewhere; shape (1, 1, h, w)."""
    i = torch.arange(h, device=device).view(-1, 1)
    j = torch.arange(w, device=device).view(1, -1)
    return ((i + j) % 2 == 0).float().view(1, 1, h, w)


def ctx_net(cin, cout, hidden, n_cond):
    """1x1 -> depthwise 3x3 -> depthwise 3x3 -> 1x1 -> 1x1 (accumulator). All integer."""
    q = dict(in_frac=FRAC, out_frac=FRAC, n_cond=n_cond)
    return IntSequential(
        QConv2d(cin, hidden, 1, **q),
        QConv2d(hidden, hidden, 3, groups=hidden, **q),
        QConv2d(hidden, hidden, 3, groups=hidden, **q),
        QConv2d(hidden, hidden, 1, **q),
        QConv2d(hidden, 2 * cout, 1, in_frac=FRAC, act=None, n_cond=n_cond),
    )


class ZPrior(nn.Module):
    """Zero-mean discretised Gaussian per hyper-latent channel with a learned scale. Training: additive uniform
    noise (differentiable rate w.r.t. z, which keeps z small); coding: the channel's scale is frozen to an index
    of the shared Gaussian CDF tables by `update_tables` and stored as an integer buffer."""

    def __init__(self, C):
        super().__init__()
        self.channels = C
        self.log_scale = nn.Parameter(torch.zeros(C))
        self.register_buffer("idx", torch.zeros(C, dtype=torch.long), persistent=True)

    def sigma(self):
        return torch.exp(self.log_scale).clamp(tables.SCALE_MIN, tables.SCALE_MAX)

    def forward(self, z, noise=True):
        s = self.sigma().view(1, -1, 1, 1)
        z_hat = ste(torch.round(z), z)
        zt = z + torch.empty_like(z).uniform_(-0.5, 0.5) if noise else z_hat
        return z_hat, gaussian_bits(zt, s, torch.zeros_like(s))

    @torch.no_grad()
    def update_tables(self):
        st = torch.from_numpy(tables.scale_table()).float().log()
        self.idx.copy_((self.sigma().cpu().log().view(-1, 1) - st.view(1, -1)).abs().argmin(1))


class GCodec(nn.Module):
    def __init__(self, m: dict):
        super().__init__()
        self.cfg = dict(m)
        N, M, Z, L = m["N"], m["M"], m["Z"], m["n_levels"]
        self.M, self.L, self.slices = M, L, list(m["slices"])
        self.qmap_block, self.qmap_range, self.beta_levels = m["qmap_block"], m["qmap_range"], m["beta_levels"]
        self.cond = CondEmbed(L, COND_DIM)
        self.g_a = Analysis(N, M, m["ga_blocks"], COND_DIM)
        self.g_s = Synthesis(M, m["gs_width"], m["gs_blocks"], COND_DIM)
        self.h_a = HyperAnalysis(M, Z)
        H = m["hyper_hidden"]
        self.h_s = IntSequential(
            QConv2d(Z, H, 3, in_frac=0, out_frac=FRAC, n_cond=L), Up2(),
            QConv2d(H, H, 3, in_frac=FRAC, out_frac=FRAC, n_cond=L), Up2(),
            QConv2d(H, H, 3, in_frac=FRAC, out_frac=FRAC, n_cond=L),
        )
        self.prior = ZPrior(Z)
        self.gauss = GaussianConditional()
        self.ctx_a, self.ctx_n = nn.ModuleList(), nn.ModuleList()
        prev = 0
        for c in self.slices:
            self.ctx_a.append(ctx_net(H + prev + 1, c, m["ctx_hidden"], L))
            self.ctx_n.append(ctx_net(H + prev + c + 1, c, m["ctx_hidden"], L))
            prev += c
        self.head = ScaleMeanHead(acc_frac=7 + FRAC, mu_frac=FRAC)
        # gain units: level 0 = coarsest quantisation (lowest rate)
        self.log_gain = nn.Parameter(torch.linspace(math.log(0.3), math.log(2.5), L).view(L, 1).repeat(1, M))

    # ------------------------------------------------------------------ helpers
    def check_ranges(self):
        """Assert every integer accumulator stays < 2**53 (exactness of the float64 trick)."""
        check_exact_range(self.h_s, Z_CLAMP)
        for net in list(self.ctx_a) + list(self.ctx_n):
            check_exact_range(net, 2.0 ** 16)  # hyper features < 2**16 (a_bits), |y_int| <= 2047 * 16

    def qmap_latent(self, cells: torch.Tensor, hl: int, wl: int) -> torch.Tensor:
        """Quality-map cells (B, 1, ch, cw) integers -> latent grid (B, 1, hl, wl) by nearest repetition."""
        r = self.qmap_block // 16
        q = cells.float().repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)
        return q[:, :, :hl, :wl]

    def gain(self, level: torch.Tensor, qlat: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_gain[level])[:, :, None, None] * torch.pow(2.0, qlat / 4)

    def beta_value(self, beta_idx: torch.Tensor) -> torch.Tensor:
        return beta_idx.float() / (self.beta_levels - 1)

    def cell_shape(self, h, w):
        return -(-h // self.qmap_block), -(-w // self.qmap_block)

    # ------------------------------------------------------------------ training / estimation forward
    def forward(self, x, level, beta, cells, noise: bool = True):
        """x in [0, 1], H and W multiples of 64. level (B,) long, beta (B,) float in [0, 1], cells (B,1,ch,cw).
        Returns dict(x_hat, bits (B,) total bits incl. hyper, y_hat, bits_map)."""
        y = self.g_a(x * 2 - 1, self.cond(level))
        qlat = self.qmap_latent(cells, *y.shape[-2:])
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_s = y.float() * self.gain(level, qlat)
        return self.from_latent(y_s, level, beta, cells, noise)

    def from_latent(self, y_s, level, beta, cells, noise: bool = True):
        """Everything after the analysis transform (also used for encoder-side latent refinement)."""
        c_dec = self.cond(level, beta)
        hl, wl = y_s.shape[-2:]
        qlat = self.qmap_latent(cells, hl, wl)
        with torch.autocast(device_type=y_s.device.type, enabled=False):
            g = self.gain(level, qlat)
            z = self.h_a(y_s)
            z_hat, bits_z = self.prior(z, noise)
            hyp = self.h_s(torch.clamp(z_hat, -Z_CLAMP, Z_CLAMP), level)
            y_hat_s, bits_y = self._slices(y_s, hyp, qlat, level, noise)
            y_hat = y_hat_s / g
        x_hat = self.g_s(y_hat, qlat / 4, c_dec)
        bits = bits_y.flatten(1).sum(1) + bits_z.flatten(1).sum(1)
        return dict(x_hat=(x_hat + 1) / 2, bits=bits, y_hat=y_hat, y_s=y_s, bits_map=bits_y.sum(1, keepdim=True))

    def _slices(self, y_s, hyp, qlat, level, noise):
        B, _, h, w = y_s.shape
        ma = checkerboard(h, w, y_s.device)
        mn = 1 - ma
        prev, outs, bits = [], [], []
        start = 0
        for k, c in enumerate(self.slices):
            yk = y_s[:, start:start + c]
            start += c
            sig_a, mu_a = self.head(self.ctx_a[k](torch.cat([hyp, *prev, qlat], 1), level))
            ya = ste(torch.round(yk - mu_a) + mu_a, yk)
            anchors = ya * ma
            ctx = torch.clamp(anchors, -Y_CLAMP, Y_CLAMP)
            sig_n, mu_n = self.head(self.ctx_n[k](torch.cat([hyp, *prev, ctx, qlat], 1), level))
            yn = ste(torch.round(yk - mu_n) + mu_n, yk)
            yk_hat = anchors + yn * mn
            if noise:
                yt = yk + torch.empty_like(yk).uniform_(-0.5, 0.5)
                b = gaussian_bits(yt, sig_a, mu_a) * ma + gaussian_bits(yt, sig_n, mu_n) * mn
            else:
                b = gaussian_bits(ya, sig_a, mu_a) * ma + gaussian_bits(yn, sig_n, mu_n) * mn
            bits.append(b)
            outs.append(yk_hat)
            prev.append(torch.clamp(yk_hat, -Y_CLAMP, Y_CLAMP))
        return torch.cat(outs, 1), torch.cat(bits, 1)

    # ------------------------------------------------------------------ synthesis from decoded latents
    def synthesize(self, y_hat_s, level, beta_idx, cells):
        hl, wl = y_hat_s.shape[-2:]
        qlat = self.qmap_latent(cells, hl, wl)
        y_hat = y_hat_s / self.gain(level, qlat)
        c = self.cond(level, self.beta_value(beta_idx))
        return (self.g_s(y_hat, qlat / 4, c) + 1) / 2

    def analysis(self, x, level, cells):
        """Encoder side: x in [0, 1] -> y_s (scaled latent, float32)."""
        y = self.g_a(x * 2 - 1, self.cond(level)).float()
        qlat = self.qmap_latent(cells, *y.shape[-2:])
        return y * self.gain(level, qlat)

    @torch.no_grad()
    def update_tables(self):
        """Freeze the hyper-prior scales into integer table indices (call once before export / coding)."""
        self.prior.update_tables()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def pad64(x: torch.Tensor):
    """Replicate-pad (B,3,H,W) to multiples of 64; returns (padded, (H, W))."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % 64, (-W) % 64
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (H, W)
