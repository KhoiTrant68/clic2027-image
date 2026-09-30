"""Interleaved rANS in numpy: integer-only, so bit-exact on every machine.

Each symbol i is coded with its own table t[i] from a shared set of quantised CDFs
(`cdf[t, s]`, totals 2**PREC). Symbols are spread round-robin over N_LANES independent rANS
states, so encode/decode vectorise over lanes (one numpy step per N_LANES symbols).

State x in [L, L << 16) with L = 2**16; renormalisation moves exactly one 16-bit word, because
PREC <= 16. Stream: [u32 N_LANES final states][u16 words].

Tables must be built once (float -> int) and then shipped as integers; see `tables.py`.
"""
from __future__ import annotations

import numpy as np

PREC = 16
N_LANES = 32  # 4 bytes of state per lane per stream: 32 lanes = 128 B (~0.5% of a 0.075-bpp 2K image)
L = 1 << 16
MASK = (1 << PREC) - 1


def encode(symbols: np.ndarray, tables: np.ndarray, cdf: np.ndarray, n_lanes: int = N_LANES) -> bytes:
    """symbols[i] in [0, n_sym(tables[i])); cdf: (n_tables, max_sym + 1) uint32, cdf[:, 0] = 0, row total 2**PREC."""
    s = np.asarray(symbols, np.int64).ravel()
    t = np.asarray(tables, np.int64).ravel()
    n = s.size
    start = cdf[t, s].astype(np.uint64)
    freq = (cdf[t, s + 1] - cdf[t, s]).astype(np.uint64)
    if n and freq.min() == 0:
        raise ValueError("symbol with zero frequency")
    steps = -(-n // n_lanes)
    x = np.full(n_lanes, L, np.uint64)
    words = []  # one chunk per step, steps visited backwards; chunks are reversed at the end
    for k in range(steps - 1, -1, -1):
        lo, hi = k * n_lanes, min(n, (k + 1) * n_lanes)
        m = hi - lo
        xs, st, fr = x[:m], start[lo:hi], freq[lo:hi]
        x_max = ((L >> PREC) << 16) * fr
        ren = xs >= x_max
        if ren.any():
            words.append((xs[ren] & 0xFFFF).astype(np.uint16))  # ascending lanes = decoder read order
            xs[ren] >>= 16
        x[:m] = (xs // fr << PREC) + xs % fr + st
    body = np.concatenate(words[::-1]) if words else np.zeros(0, np.uint16)
    return x.astype("<u4").tobytes() + body.astype("<u2").tobytes()


def decode(data: bytes, tables: np.ndarray, cdf: np.ndarray, inv: np.ndarray, n_lanes: int = N_LANES) -> np.ndarray:
    """inv[t, slot] = symbol s with cdf[t, s] <= slot < cdf[t, s+1] (see `tables.inverse`)."""
    t = np.asarray(tables, np.int64).ravel()
    n = t.size
    x = np.frombuffer(data[:4 * n_lanes], "<u4").astype(np.uint64)
    words = np.frombuffer(data[4 * n_lanes:], "<u2").astype(np.uint64)
    out = np.empty(n, np.int64)
    pos = 0
    for k in range(-(-n // n_lanes)):
        lo, hi = k * n_lanes, min(n, (k + 1) * n_lanes)
        m = hi - lo
        xs, tt = x[:m], t[lo:hi]
        slot = xs & MASK
        s = inv[tt, slot].astype(np.int64)
        st = cdf[tt, s].astype(np.uint64)
        fr = (cdf[tt, s + 1] - cdf[tt, s]).astype(np.uint64)
        xs = fr * (xs >> PREC) + slot - st
        ren = xs < L
        r = int(ren.sum())
        if r:
            xs[ren] = (xs[ren] << 16) | words[pos:pos + r]
            pos += r
        x[:m] = xs
        out[lo:hi] = s
    if pos != words.size or not np.all(x == L):  # every lane must return to the initial state
        raise ValueError("rANS integrity check failed (corrupt stream or wrong tables)")
    return out


# ---------------------------------------------------------------- small helpers for side information
def put_varint(v: int, out: bytearray):
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return


def get_varint(buf: bytes, pos: int):
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, pos


def zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 63) if v < 0 else v << 1


def unzigzag(u: int) -> int:
    return (u >> 1) ^ -(u & 1)
