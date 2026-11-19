"""Value-level coding on top of rANS: integers per element, table per element, escape side stream.

    blob = encode_values(q, table_idx, cdf, half)       # q: integer values, table_idx: which table each uses
    q = decode_values(blob, table_idx, cdf, half, inv)

Numpy only. Blob layout: varint(len(rans)) | rans | varint(n_escapes) | zigzag varints.
"""
from __future__ import annotations

import numpy as np

from . import rans


def encode_values(q: np.ndarray, t: np.ndarray, cdf: np.ndarray, half: np.ndarray) -> bytes:
    q = np.asarray(q, np.int64).ravel()
    t = np.asarray(t, np.int64).ravel()
    k = half[t]
    esc = np.abs(q) > k
    sym = np.where(esc, 2 * k + 1, q + k)
    body = rans.encode(sym, t, cdf)
    out = bytearray()
    rans.put_varint(len(body), out)
    out += body
    ev = q[esc]
    rans.put_varint(ev.size, out)
    for v in ev.tolist():
        rans.put_varint(rans.zigzag(int(v)), out)
    return bytes(out)


def decode_values(blob: bytes, t: np.ndarray, cdf: np.ndarray, half: np.ndarray, inv: np.ndarray):
    t = np.asarray(t, np.int64).ravel()
    n_body, pos = rans.get_varint(blob, 0)
    sym = rans.decode(blob[pos:pos + n_body], t, cdf, inv)
    pos += n_body
    k = half[t]
    q = sym - k
    esc = sym == 2 * k + 1
    n_esc, pos = rans.get_varint(blob, pos)
    if n_esc != int(esc.sum()):
        raise ValueError("escape count mismatch")
    vals = []
    for _ in range(n_esc):
        u, pos = rans.get_varint(blob, pos)
        vals.append(rans.unzigzag(u))
    q[esc] = np.asarray(vals, np.int64)
    if pos != len(blob):
        raise ValueError("trailing bytes in blob")
    return q
