"""GC bitstream: compress / decompress one image.

Layout (little endian):
    b"GC" | version u8 | H u16 | W u16 | level u8 | beta u8 | qmap flag u8
    [qmap blob]  (flag 1: varint length + rANS blob of cell values over a fixed integer table)
    z blob, then 2 * len(slices) y blobs (anchor, non-anchor per slice), each as varint length + blob

Encoder and decoder share `_entropy_pass`, so they derive the entropy parameters with literally the same code;
only the leaf callback differs (quantise + encode vs decode). Integer nets run on `ent_dev` with cuDNN disabled:
with integers carried in float64 every partial sum is exact, so CPU and any CUDA GPU give identical tables.
"""
from __future__ import annotations

import struct
import time
from contextlib import nullcontext

import numpy as np
import torch

from clic27.entropy import rans
from clic27.entropy.intnet import exact_f32_context

from . import rans_c
from .model import FRAC, Y_CLAMP, Z_CLAMP, GCodec, checkerboard, pad64

MAGIC, VERSION = b"GC", 1
HDR = struct.Struct("<2sBHHBBB")


# ---------------------------------------------------------------- quality-map table (integers only)
def qmap_cdf(R: int) -> np.ndarray:
    """Fixed table over 2R+1 symbols peaked at k = 0 (built with integer arithmetic only)."""
    w = np.array([1 << max(0, 10 - 3 * abs(k)) for k in range(-R, R + 1)], np.int64)
    f = w * (rans.L // int(w.sum()))
    f[R] += rans.L - int(f.sum())
    cdf = np.zeros((1, 2 * R + 3), np.uint32)
    cdf[0, 1:2 * R + 2] = np.cumsum(f)
    cdf[0, 2 * R + 2] = rans.L
    return cdf


def _put_blob(out: bytearray, blob: bytes):
    rans.put_varint(len(blob), out)
    out += blob


def _get_blob(buf: bytes, pos: int):
    n, pos = rans.get_varint(buf, pos)
    return buf[pos:pos + n], pos + n


def _z_tables(model: GCodec, shape) -> np.ndarray:
    """Table index of every hyper-latent element: the channel's frozen scale index (integer buffer)."""
    idx = model.prior.idx.cpu().numpy().astype(np.int64)
    return np.broadcast_to(idx.reshape(1, -1, 1, 1), shape).ravel()


EXACT = {"mode": "f32"}  # "f32": exact fp32-split convs (fast on L4); "f64": plain float64 convs. Same results.


def _ent_ctx(dev: str):
    if EXACT["mode"] == "f32":
        return exact_f32_context()
    return torch.backends.cudnn.flags(enabled=False) if str(dev).startswith("cuda") else nullcontext()


# ---------------------------------------------------------------- shared entropy pass
@torch.no_grad()
def _entropy_pass(model: GCodec, z_q: torch.Tensor, cells: torch.Tensor, level: int, ent_dev, leaf):
    """z_q: (1, Z, hz, wz) integers. leaf(k, pass, idx, mu_int, mask) -> int64 y_int (= y_hat_s * 2**FRAC) at
    masked positions (numpy, flattened in C order over (c, h, w)). Returns y_hat_s (1, M, hl, wl) float32 (CPU)."""
    lv = torch.tensor([level], dtype=torch.long)
    with _ent_ctx(ent_dev):
        hyp = model.h_s.forward_int(torch.clamp(z_q, -Z_CLAMP, Z_CLAMP).to(ent_dev, torch.float64), lv)
        _, _, hl, wl = hyp.shape
        qint = model.qmap_latent(cells, hl, wl).to(ent_dev, torch.float64) * 2 ** FRAC
        ma = checkerboard(hl, wl, "cpu").bool()
        prev, outs = [], []
        for k, c in enumerate(model.slices):
            m_a = ma.expand(1, c, hl, wl).numpy().ravel()
            idx, mu = model.head.coding_params(model.ctx_a[k].forward_int(torch.cat([hyp, *prev, qint], 1), lv))
            y_int = np.zeros(c * hl * wl, np.int64)
            y_int[m_a] = leaf(k, 0, idx.numpy().ravel()[m_a], mu.numpy().ravel()[m_a], m_a)
            anchors = torch.from_numpy(y_int.reshape(1, c, hl, wl)).to(ent_dev, torch.float64)
            lim = Y_CLAMP * 2 ** FRAC
            ctx = torch.clamp(anchors, -lim, lim)
            idx, mu = model.head.coding_params(model.ctx_n[k].forward_int(torch.cat([hyp, *prev, ctx, qint], 1), lv))
            m_n = ~m_a
            y_int[m_n] = leaf(k, 1, idx.numpy().ravel()[m_n], mu.numpy().ravel()[m_n], m_n)
            full = torch.from_numpy(y_int.reshape(1, c, hl, wl))
            outs.append(full)
            prev.append(torch.clamp(full.to(ent_dev, torch.float64), -lim, lim))
    return torch.cat(outs, 1).float() / 2 ** FRAC


# ---------------------------------------------------------------- encoder
@torch.no_grad()
def compress(model: GCodec, x01: torch.Tensor, level: int, beta_idx: int, cells: torch.Tensor | None = None,
             ent_dev="cpu", y_s: torch.Tensor | None = None, return_recon_latent: bool = False):
    """x01: (1, 3, H, W) in [0, 1] on the model device. cells: (1, 1, ch, cw) ints in [-R, R] or None (all 0).
    y_s: optional pre-computed (e.g. refined) scaled latent. Returns bytes (and y_hat_s if asked)."""
    model.eval()
    dev = x01.device
    H, W = x01.shape[-2:]
    xp, _ = pad64(x01)
    ch, cw = model.cell_shape(*xp.shape[-2:])
    cells = torch.zeros(1, 1, ch, cw, dtype=torch.long) if cells is None else cells.long().cpu()
    assert cells.shape[-2:] == (ch, cw), (cells.shape, ch, cw)
    R = model.qmap_range
    cells = cells.clamp(-R, R)
    lv = torch.tensor([level], device=dev)
    if y_s is None:
        y_s = model.analysis(xp, lv, cells.to(dev))
    y_s = y_s.float()
    z_q = torch.round(model.h_a(y_s)).clamp(-Z_CLAMP, Z_CLAMP).cpu()

    out = bytearray(HDR.pack(MAGIC, VERSION, H, W, level, beta_idx, int(bool(cells.abs().sum()))))
    if cells.abs().sum():
        _put_blob(out, rans_c.encode(cells.numpy().ravel() + R, np.zeros(cells.numel(), np.int64), qmap_cdf(R)))
    cdf_y, half_y, _ = model.gauss.np_tables()
    _put_blob(out, rans_c.encode_values(z_q.numpy().astype(np.int64), _z_tables(model, z_q.shape), cdf_y, half_y))

    y_np = y_s.cpu().double().numpy()
    starts = np.cumsum([0] + model.slices)
    blobs = []

    def leaf(k, p, idx, mu_int, mask):
        yk = y_np[:, starts[k]:starts[k + 1]].ravel()[mask]
        q = np.round(yk - mu_int / 2 ** FRAC).astype(np.int64)
        blobs.append(rans_c.encode_values(q, idx, cdf_y, half_y))
        return q * (1 << FRAC) + mu_int

    y_hat_s = _entropy_pass(model, z_q, cells, level, ent_dev, leaf)
    for b in blobs:
        _put_blob(out, b)
    return (bytes(out), y_hat_s) if return_recon_latent else bytes(out)


# ---------------------------------------------------------------- decoder
def parse_header(buf: bytes):
    magic, ver, H, W, level, beta, flag = HDR.unpack_from(buf, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a GC v1 stream")
    return dict(H=H, W=W, level=level, beta=beta, qmap=flag), HDR.size


@torch.no_grad()
def decompress_latent(model: GCodec, buf: bytes, ent_dev="cpu"):
    """Entropy decoding only. Returns (y_hat_s, header, cells)."""
    model.eval()
    hd, pos = parse_header(buf)
    Hp, Wp = -(-hd["H"] // 64) * 64, -(-hd["W"] // 64) * 64
    ch, cw = model.cell_shape(Hp, Wp)
    R = model.qmap_range
    if hd["qmap"]:
        blob, pos = _get_blob(buf, pos)
        v = rans_c.decode(blob, np.zeros(ch * cw, np.int64), qmap_cdf(R))
        cells = torch.from_numpy(v.reshape(1, 1, ch, cw) - R)
    else:
        cells = torch.zeros(1, 1, ch, cw, dtype=torch.long)
    zshape = (1, model.prior.channels, Hp // 64, Wp // 64)
    cdf_y, half_y, _ = model.gauss.np_tables()
    blob, pos = _get_blob(buf, pos)
    z_q = torch.from_numpy(rans_c.decode_values(blob, _z_tables(model, zshape), cdf_y, half_y).reshape(zshape)).float()

    state = {"pos": pos}

    def leaf(k, p, idx, mu_int, mask):
        b, state["pos"] = _get_blob(buf, state["pos"])
        q = rans_c.decode_values(b, idx, cdf_y, half_y)
        return q * (1 << FRAC) + mu_int

    y_hat_s = _entropy_pass(model, z_q, cells, hd["level"], ent_dev, leaf)
    if state["pos"] != len(buf):
        raise ValueError("trailing bytes in GC stream")
    return y_hat_s, hd, cells


@torch.no_grad()
def decompress(model: GCodec, buf: bytes, dev="cuda", ent_dev="cpu", synth_dtype=torch.float16, timings=None):
    """Returns HxWx3 uint8 numpy."""
    t0 = time.perf_counter()
    y_hat_s, hd, cells = decompress_latent(model, buf, ent_dev)
    t1 = time.perf_counter()
    lv = torch.tensor([hd["level"]], device=dev)
    ctx = torch.autocast("cuda", dtype=synth_dtype) if (str(dev).startswith("cuda") and synth_dtype != torch.float32) \
        else nullcontext()
    with ctx:
        x = model.synthesize(y_hat_s.to(dev), lv, torch.tensor([hd["beta"]], device=dev), cells.to(dev))
    img = (x[0, :, :hd["H"], :hd["W"]].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    if timings is not None:
        timings.update(entropy=t1 - t0, synthesis=time.perf_counter() - t1)
    return img
