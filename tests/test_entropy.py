"""numpy-only tests for rANS, CDF tables and value coding (run locally, no torch).

    pytest            # or: python tests/test_entropy.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clic27.entropy import coding, rans, tables  # noqa: E402


def _tables():
    cdf, half = tables.gaussian_tables()
    return cdf, half, tables.inverse(cdf)


CDF, HALF, INV = _tables()


def test_tables_valid():
    assert CDF.shape[0] == tables.N_SCALES
    assert (CDF[:, 0] == 0).all() and (CDF[:, -1] == tables.TOTAL).all()
    for t in range(CDF.shape[0]):
        f = np.diff(CDF[t].astype(np.int64))
        n_sym = 2 * HALF[t] + 2
        assert (f[:n_sym] >= 1).all() and (f[n_sym:] == 0).all()


def test_rans_roundtrip_random_tables():
    rng = np.random.default_rng(0)
    for n in [0, 1, 63, 64, 65, 1000, 20001]:
        t = rng.integers(0, CDF.shape[0], n)
        s = np.array([rng.integers(0, 2 * HALF[i] + 2) for i in t], np.int64)
        blob = rans.encode(s, t, CDF)
        assert np.array_equal(rans.decode(blob, t, CDF, INV), s)


def test_values_roundtrip_with_escapes_and_rate():
    rng = np.random.default_rng(1)
    n = 50000
    t = rng.integers(0, 40, n)
    sc = tables.scale_table()[t]
    q = np.round(rng.normal(0, sc)).astype(np.int64)
    q[::997] = 10 ** 6 * np.sign(rng.normal(size=q[::997].size)).astype(np.int64) + 5  # force escapes
    blob = coding.encode_values(q, t, CDF, HALF)
    assert np.array_equal(coding.decode_values(blob, t, CDF, HALF, INV), q)
    # without escapes, the size should be close to the ideal code length
    q2 = np.round(rng.normal(0, sc)).astype(np.int64)
    q2 = np.clip(q2, -HALF[t], HALF[t])
    k = HALF[t]
    f = (CDF[t, q2 + k + 1] - CDF[t, q2 + k]).astype(np.float64)
    ideal_bytes = -np.log2(f / tables.TOTAL).sum() / 8
    real = len(coding.encode_values(q2, t, CDF, HALF))
    assert real < ideal_bytes * 1.01 + 4 * rans.lanes_for(q2.size) + 16, (real, ideal_bytes)


def test_corruption_detected():
    rng = np.random.default_rng(2)
    t = rng.integers(0, 64, 5000)
    s = np.array([rng.integers(0, 2 * HALF[i] + 2) for i in t], np.int64)
    blob = bytearray(rans.encode(s, t, CDF))
    blob[len(blob) // 2] ^= 0xFF
    try:
        out = rans.decode(bytes(blob), t, CDF, INV)
        assert not np.array_equal(out, s)
    except ValueError:
        pass


def test_quantize_pmf():
    f = tables.quantize_pmf([1e-30, 0.5, 0.5])
    assert f.sum() == tables.TOTAL and f.min() >= 1
    assert abs(math.log2(f[1] / f[2])) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
