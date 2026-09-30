"""Entropy models with bit-exact decoding.

- DiscretePrior: learned per-channel pmf for the hyper-latent z (factorised, integer tables).
- ScaleMeanHead: turns the integer accumulator of an IntSequential into (scale index, dyadic mean).
- GaussianConditional: rate for training; compress/decompress y given (scale index, mean).

CDF tables are module buffers. They are built from floats when the module is created or when
`update_tables()` runs, and must then be loaded from the SAME checkpoint on the encoder and the
decoder side (never rebuilt on the decoder), so libm differences cannot change them.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import coding, tables
from .intnet import ste

LN2 = math.log(2.0)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class _Tables(nn.Module):
    """Holds (cdf, half) as persistent buffers and a lazily built inverse table."""

    def _set_tables(self, cdf: np.ndarray, half: np.ndarray):
        self.register_buffer("cdf", torch.from_numpy(cdf.astype(np.int64)), persistent=True)
        self.register_buffer("half", torch.from_numpy(half.astype(np.int64)), persistent=True)
        self._inv = None

    def _load_from_state_dict(self, *args, **kwargs):  # tables come from the checkpoint: drop the cache
        super()._load_from_state_dict(*args, **kwargs)
        self._inv = None

    def np_tables(self):
        cdf, half = _np(self.cdf).astype(np.uint32), _np(self.half)
        if self._inv is None:
            self._inv = tables.inverse(cdf)
        return cdf, half, self._inv


class DiscretePrior(_Tables):
    """p(z_c = v) = softmax(logits_c)[v + half], v in [-half, half], plus an escape bin.

    Training: z is rounded with a straight-through estimator, so the rate trains the logits (and the
    rest of the model through z_hat) but gives no gradient w.r.t. z itself. The hyper-rate is small."""

    def __init__(self, channels: int, half: int = 32, init_scale: float = 4.0):
        super().__init__()
        self.channels, self.half_width = channels, half
        v = torch.arange(-half, half + 1, dtype=torch.float32)
        base = torch.cat([-(v / init_scale) ** 2 / 2, torch.tensor([-10.0])])
        self.logits = nn.Parameter(base.repeat(channels, 1))
        self._set_tables(np.zeros((channels, 2 * half + 3), np.int64), np.full(channels, half, np.int64))

    def forward(self, z: torch.Tensor):
        """Returns (z_hat, bits per element)."""
        z_hat = ste(torch.round(z), z)
        q = torch.round(z).detach().long()
        esc = q.abs() > self.half_width
        idx = torch.where(esc, torch.full_like(q, 2 * self.half_width + 1), q + self.half_width)
        logp = F.log_softmax(self.logits, dim=1)  # (C, W)
        C = z.shape[1]
        lp = logp[torch.arange(C, device=z.device).view(1, C, 1, 1).expand_as(idx), idx]
        extra = torch.where(esc, 2 * torch.log2(q.abs().float() + 1) + 1, torch.zeros_like(z))  # varint-ish
        return z_hat, -lp / LN2 + extra

    @torch.no_grad()
    def update_tables(self):
        pmf = _np(F.softmax(self.logits.double(), dim=1))
        self._set_tables(*tables.pmf_tables(pmf))

    def compress(self, z: torch.Tensor) -> bytes:
        cdf, half, _ = self.np_tables()
        q = _np(torch.round(z)).astype(np.int64)
        t = np.broadcast_to(np.arange(q.shape[1]).reshape(1, -1, 1, 1), q.shape)
        return coding.encode_values(q, t, cdf, half)

    def decompress(self, blob: bytes, shape) -> torch.Tensor:
        cdf, half, inv = self.np_tables()
        t = np.broadcast_to(np.arange(shape[1]).reshape(1, -1, 1, 1), shape)
        return torch.from_numpy(coding.decode_values(blob, t, cdf, half, inv).reshape(shape)).float()


class ScaleMeanHead(nn.Module):
    """Maps the last-layer accumulator (2*M channels: scale part, mean part) of an IntSequential to
    sigma (training) or an integer scale index + a dyadic mean with MU_FRAC fractional bits (coding)."""

    def __init__(self, acc_frac: int, mu_frac: int = 4):
        super().__init__()
        self.acc_frac, self.mu_frac = acc_frac, mu_frac
        s = tables.scale_table()
        # idx = #{j >= 1 : sigma(acc) >= s_j}, sigma(a) = s_0 + softplus(a)  =>  a >= log(expm1(s_j - s_0))
        thr = np.ceil(np.log(np.expm1(s[1:] - s[0])) * 2 ** acc_frac).astype(np.int64)
        self.register_buffer("thr", torch.from_numpy(thr), persistent=True)
        self.s0 = float(s[0])

    def forward(self, acc: torch.Tensor):
        """Training path. Returns (sigma, mu); mu is floored to the MU_FRAC grid with STE."""
        a_s, a_m = acc.chunk(2, dim=1)
        sigma = self.s0 + F.softplus(a_s)
        mu = ste(torch.floor(a_m * 2 ** self.mu_frac) / 2 ** self.mu_frac, a_m)
        return sigma, mu

    @torch.no_grad()
    def coding_params(self, acc_int: torch.Tensor):
        """acc_int: integer accumulator (float64). Returns (scale index int64, mu_int int64)."""
        a_s, a_m = acc_int.chunk(2, dim=1)
        idx = torch.bucketize(a_s, self.thr.to(a_s.device).to(torch.float64), right=True)
        mu_int = torch.floor(a_m / 2 ** (self.acc_frac - self.mu_frac))
        return idx.long().cpu(), mu_int.long().cpu()


class GaussianConditional(_Tables):
    def __init__(self):
        super().__init__()
        self._set_tables(*tables.gaussian_tables())

    @staticmethod
    def bits(y: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, training: bool):
        """Returns (y_hat, bits per element). Training: additive uniform noise; else rounding around mu (STE)."""
        if training:
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = ste(torch.round(y - mu) + mu, y)
        v = (y_hat - mu).abs()
        s = sigma.clamp_min(tables.SCALE_MIN)
        c = lambda x: 0.5 * torch.erfc(-x / math.sqrt(2.0))  # noqa: E731
        p = c((0.5 - v) / s) - c((-0.5 - v) / s)
        return y_hat, -torch.log2(p.clamp_min(1e-9))

    def compress(self, y: torch.Tensor, idx: torch.Tensor, mu_int: torch.Tensor, mu_frac: int) -> bytes:
        cdf, half, _ = self.np_tables()
        mu = _np(mu_int).astype(np.float64) / 2 ** mu_frac
        q = np.round(_np(y).astype(np.float64) - mu).astype(np.int64)
        return coding.encode_values(q, _np(idx), cdf, half)

    def decompress(self, blob: bytes, idx: torch.Tensor, mu_int: torch.Tensor, mu_frac: int):
        """Returns (y_hat float32, y_hat_int = y_hat * 2**mu_frac as int64 for integer context nets)."""
        cdf, half, inv = self.np_tables()
        q = coding.decode_values(blob, _np(idx), cdf, half, inv).reshape(tuple(idx.shape))
        y_int = q * (1 << mu_frac) + _np(mu_int)
        return torch.from_numpy(y_int / 2 ** mu_frac).float(), torch.from_numpy(y_int)
