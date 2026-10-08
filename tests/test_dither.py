"""numpy-only tests for subtractive-dither coding (Proposition 1 in the bitstream).

    pytest            # or: python tests/test_dither.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clic27.entropy import coding, dither, tables  # noqa: E402

NOFF, MU_FRAC = 8, 4
CDF, HALF = tables.gaussian_tables(n_offsets=NOFF)
INV = tables.inverse(CDF)


def _phi(x):
    return 0.5 * np.vectorize(math.erfc)(-x / math.sqrt(2.0))


def _setup(n, seed=0):
    rng = np.random.default_rng(seed)
    sidx = rng.integers(0, 40, n)
    sigma = tables.scale_table()[sidx]
    mu_int = rng.integers(-200, 200, n)
    mu = mu_int / 2 ** MU_FRAC
    y = rng.normal(mu, sigma)
    return y, sidx, sigma, mu_int, mu


def _encode(y, sidx, mu_int, seed):
    u16 = dither.dither_u16(y.size, seed)
    q = np.round(y + dither.u_float(u16)).astype(np.int64)
    t, base = dither.table_and_base(dither.centre_c17(mu_int, u16, MU_FRAC), sidx, NOFF)
    return coding.encode_values(q - base, t, CDF, HALF), q


def _decode(blob, sidx, mu_int, seed):
    u16 = dither.dither_u16(sidx.size, seed)
    t, base = dither.table_and_base(dither.centre_c17(mu_int, u16, MU_FRAC), sidx, NOFF)
    q = coding.decode_values(blob, t, CDF, HALF, INV) + base
    return q - dither.u_float(u16)


def test_dither_is_deterministic_and_uniform():
    a, b = dither.dither_u16(100000, 7), dither.dither_u16(100000, 7)
    assert np.array_equal(a, b) and not np.array_equal(a, dither.dither_u16(100000, 8))
    u = dither.u_float(a)
    assert u.min() > -0.5 and u.max() < 0.5 and abs(u.mean()) < 0.005 and abs(u.var() - 1 / 12) < 0.002


def test_centre_matches_float():
    rng = np.random.default_rng(1)
    mu_int = rng.integers(-5000, 5000, 1000)
    u16 = dither.dither_u16(1000, 3)
    c = dither.centre_c17(mu_int, u16, MU_FRAC) / 2 ** dither.C_BITS
    assert np.array_equal(c, mu_int / 2 ** MU_FRAC + dither.u_float(u16))  # dyadic: exact in float64


def test_roundtrip_and_error_is_uniform_independent():
    y, sidx, sigma, mu_int, mu = _setup(60000)
    blob, q = _encode(y, sidx, mu_int, seed=11)
    y_hat = _decode(blob, sidx, mu_int, seed=11)
    u = dither.u_float(dither.dither_u16(y.size, 11))
    assert np.array_equal(y_hat, q - u)
    e = y_hat - y
    assert np.all(np.abs(e) <= 0.5) and abs(e.var() - 1 / 12) < 0.002
    assert abs(np.corrcoef(e, y)[0, 1]) < 0.02  # Prop. 1: error independent of the source


def test_rate_close_to_dithered_ideal():
    y, sidx, sigma, mu_int, mu = _setup(60000, seed=2)
    blob, q = _encode(y, sidx, mu_int, seed=5)
    u = dither.u_float(dither.dither_u16(y.size, 5))
    # exact code length of the dithered quantiser under the true model: P(q|u) = F(q-u+1/2) - F(q-u-1/2)
    p = _phi((q - u + 0.5 - mu) / sigma) - _phi((q - u - 0.5 - mu) / sigma)
    ideal = -np.log2(np.maximum(p, 1e-12)).sum() / 8
    assert len(blob) < ideal * 1.03 + 200, (len(blob), ideal)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
