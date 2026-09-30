"""Quantised CDF tables for rANS (numpy only).

Tables are computed ONCE from floats at export time and then stored as integers in the model
file; encoder and decoder load the same integers, so libm differences (erf, exp) never matter.

Layout of a table set: cdf (n_tables, width + 1) uint32 with cdf[:, 0] = 0 and total 2**PREC;
symbol j of table t covers value j - half[t] for j < 2*half[t] + 1; symbol 2*half[t] + 1 is the
escape (value outside [-half, half], sent as a varint side stream). Unused columns have zero width.
"""
from __future__ import annotations

import math

import numpy as np

from .rans import PREC

TOTAL = 1 << PREC
SCALE_MIN, SCALE_MAX, N_SCALES = 0.11, 256.0, 64
TAIL_SIGMAS = 6.0
HALF_MAX = 1023


def scale_table() -> np.ndarray:
    """Log-spaced Gaussian scales (CompressAI-style). Index 0 is SCALE_MIN."""
    return np.exp(np.linspace(math.log(SCALE_MIN), math.log(SCALE_MAX), N_SCALES))


def quantize_pmf(p: np.ndarray, total: int = TOTAL) -> np.ndarray:
    """Float pmf -> integer frequencies >= 1 summing to `total` (deterministic)."""
    p = np.asarray(p, np.float64)
    p = p / p.sum()
    n = p.size
    f = np.floor(p * (total - n)).astype(np.int64) + 1
    f[int(np.argmax(f))] += total - int(f.sum())
    assert f.min() >= 1 and f.sum() == total
    return f


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc)(-x / math.sqrt(2.0))


def gaussian_tables(scales=None):
    """Discretised zero-mean Gaussians, one table per scale. Returns (cdf, half)."""
    scales = scale_table() if scales is None else np.asarray(scales, np.float64)
    half = np.minimum(np.ceil(TAIL_SIGMAS * scales).astype(np.int64), HALF_MAX)
    width = int(2 * half.max() + 2)
    cdf = np.full((len(scales), width + 1), TOTAL, np.uint32)
    cdf[:, 0] = 0
    for t, (s, k) in enumerate(zip(scales, half)):
        v = np.arange(-k, k + 1, dtype=np.float64)
        p = _phi((v + 0.5) / s) - _phi((v - 0.5) / s)
        tail = 2 * _phi(-(k + 0.5) / s)
        f = quantize_pmf(np.append(p, max(tail, 1e-12)))
        cdf[t, 1:f.size + 1] = np.cumsum(f)
    return cdf, half


def pmf_tables(pmfs: np.ndarray):
    """Learned discrete priors: pmfs (n_tables, 2*half + 2) incl. escape as last entry. Returns (cdf, half)."""
    pmfs = np.asarray(pmfs, np.float64)
    n, w = pmfs.shape
    half = np.full(n, (w - 2) // 2, np.int64)
    cdf = np.zeros((n, w + 1), np.uint32)
    for t in range(n):
        cdf[t, 1:] = np.cumsum(quantize_pmf(pmfs[t]))
    return cdf, half


def inverse(cdf: np.ndarray) -> np.ndarray:
    """inv[t, slot] = symbol whose interval contains slot (uint16; zero-width columns never hit)."""
    n, w1 = cdf.shape
    inv = np.empty((n, TOTAL), np.uint16)
    for t in range(n):
        freq = np.diff(cdf[t].astype(np.int64))
        inv[t] = np.repeat(np.arange(w1 - 1, dtype=np.uint16), freq)
    return inv
