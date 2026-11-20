"""C rANS via ctypes, with the numpy implementation (clic27.entropy.rans) as a byte-identical fallback.

GC streams use ONE lane per blob: the C coder is fast enough without interleaving and every extra lane costs 4
bytes. The numpy fallback is far too slow for 2K images with one lane; it exists for tests and for checking
that the C library and numpy produce the same bytes.

    python -m clic27.gc.rans_c build     # compile _rans.so next to this file (gcc or cc)
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from clic27.entropy import rans

HERE = Path(__file__).resolve().parent
SRC = HERE / "_rans.c"
LIB = HERE / "_rans.so"
LANES = 1
_lib = None


def build(out: Path = LIB) -> Path:
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("no C compiler found (set CC)")
    subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", str(out), str(SRC)], check=True)
    return out


def lib():
    """Load (building if needed and possible) the C library; None if unavailable."""
    global _lib
    if _lib is None:
        if not LIB.exists() and os.environ.get("GC_NO_BUILD") != "1":
            try:
                build()
            except Exception as e:  # noqa: BLE001
                print(f"[rans_c] C build failed ({e}); using numpy rANS", file=sys.stderr)
        if LIB.exists():
            try:
                L = ctypes.CDLL(str(LIB))
            except OSError as e:  # e.g. built against a newer glibc than this machine has
                print(f"[rans_c] cannot load {LIB} ({e}); delete it to rebuild here", file=sys.stderr)
                return None
            i64p, u32p, u8p = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.int64, np.uint32, np.uint8))
            L.rans_encode.argtypes = [i64p, i64p, ctypes.c_int64, u32p, ctypes.c_int64, ctypes.c_int64, u8p]
            L.rans_encode.restype = ctypes.c_int64
            L.rans_decode.argtypes = [u8p, ctypes.c_int64, i64p, ctypes.c_int64, u32p, ctypes.c_int64, ctypes.c_int64,
                                      i64p]
            L.rans_decode.restype = ctypes.c_int64
            _lib = L
        else:
            _lib = False
    return _lib or None


def encode(sym: np.ndarray, tab: np.ndarray, cdf: np.ndarray, lanes: int = LANES) -> bytes:
    L = lib()
    sym = np.ascontiguousarray(sym, np.int64).ravel()
    tab = np.ascontiguousarray(tab, np.int64).ravel()
    if L is None:
        return rans.encode(sym, tab, cdf, n_lanes=lanes)
    cdf = np.ascontiguousarray(cdf, np.uint32)
    out = np.empty(4 * lanes + 2 * sym.size + 16, np.uint8)
    n = L.rans_encode(sym, tab, sym.size, cdf, cdf.shape[1], lanes, out)
    if n < 0:
        raise ValueError("symbol with zero frequency")
    return out[:n].tobytes()


def decode(data: bytes, tab: np.ndarray, cdf: np.ndarray, lanes: int = LANES) -> np.ndarray:
    L = lib()
    tab = np.ascontiguousarray(tab, np.int64).ravel()
    if L is None:
        return rans.decode(data, tab, cdf, None, n_lanes=lanes)
    cdf = np.ascontiguousarray(cdf, np.uint32)
    buf = np.frombuffer(data, np.uint8).copy() if len(data) else np.zeros(1, np.uint8)
    out = np.empty(tab.size, np.int64)
    if L.rans_decode(buf, len(data), tab, tab.size, cdf, cdf.shape[1], lanes, out) != 0:
        raise ValueError("rANS integrity check failed (corrupt stream or wrong tables)")
    return out


def encode_values(q: np.ndarray, t: np.ndarray, cdf: np.ndarray, half: np.ndarray) -> bytes:
    """Same blob layout as clic27.entropy.coding.encode_values (varint body length | body | escapes)."""
    q = np.asarray(q, np.int64).ravel()
    t = np.asarray(t, np.int64).ravel()
    k = half[t]
    esc = np.abs(q) > k
    sym = np.where(esc, 2 * k + 1, q + k)
    body = encode(sym, t, cdf)
    out = bytearray()
    rans.put_varint(len(body), out)
    out += body
    ev = q[esc]
    rans.put_varint(ev.size, out)
    for v in ev.tolist():
        rans.put_varint(rans.zigzag(int(v)), out)
    return bytes(out)


def decode_values(blob: bytes, t: np.ndarray, cdf: np.ndarray, half: np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.int64).ravel()
    n_body, pos = rans.get_varint(blob, 0)
    sym = decode(blob[pos:pos + n_body], t, cdf)
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


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(build())
    else:
        print(__doc__)
