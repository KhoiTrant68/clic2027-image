"""Shared subtractive dither (Proposition 1) with bit-exact integer bookkeeping (numpy only).

The dither u ~ U(-1/2, 1/2) is drawn from PCG64.random_raw: raw bit-generator output is fully
specified by the PCG64 algorithm, so encoder and decoder get the same bits on any machine and
numpy version. u has 16 fractional bits: u = (u16 + 1/2) / 2**16 - 1/2, exactly representable.

Coding (in units of the quantisation step, i.e. on the gain-scaled latent y_r):
  encoder: q = round(y_r + u);  decoder: y_hat_r = q - u  ->  y_hat_r - y_r ~ U(-1/2, 1/2), independent of y.
  entropy model: q ~ N(mu + u, sigma^2) discretised; with c = mu + u, the symbol is v = q - floor(c)
  and the table is chosen by (scale index, floor(frac(c) * n_offsets)).
mu comes from the integer hyper-network with MU_FRAC fractional bits, so c is computed exactly in
units of 2**-17 (u needs 17 bits once the 1/2 offset is included).
"""
from __future__ import annotations

import numpy as np

U_BITS = 16
C_BITS = U_BITS + 1


def dither_u16(n: int, seed: int) -> np.ndarray:
    """n raw 16-bit draws (int64 in [0, 2**16)), identical on every machine for a given seed."""
    raw = np.random.PCG64(seed).random_raw(n)  # uint64
    return (raw >> np.uint64(64 - U_BITS)).astype(np.int64)


def u_float(u16: np.ndarray) -> np.ndarray:
    """Exact dyadic value of u in (-1/2, 1/2)."""
    return (u16.astype(np.float64) + 0.5) / 2 ** U_BITS - 0.5


def centre_c17(mu_int: np.ndarray, u16: np.ndarray, mu_frac: int) -> np.ndarray:
    """c = mu + u in units of 2**-17, as int64."""
    assert mu_frac <= C_BITS
    return mu_int.astype(np.int64) * (1 << (C_BITS - mu_frac)) + 2 * u16 + 1 - (1 << U_BITS)


def table_and_base(c17: np.ndarray, scale_idx: np.ndarray, n_offsets: int):
    """(table index, base = floor(c)) for each element; symbol value v = q - base."""
    base = np.floor_divide(c17, 1 << C_BITS)
    frac = c17 - base * (1 << C_BITS)  # in [0, 2**17)
    off = (frac * n_offsets) >> C_BITS
    return scale_idx.astype(np.int64) * n_offsets + off, base
