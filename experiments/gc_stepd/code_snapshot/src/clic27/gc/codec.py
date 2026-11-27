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
from clic27.entropy.intnet import exact_f32_context, exact_i8_context

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


# All modes give identical integers (exact arithmetic); they differ in speed and in what they touch:
#   "i8": CUDA int8 GEMMs + float64 depthwise; no global flag is changed, so the synthesis of another image can
#         run concurrently with cuDNN on (engine B). CPU tensors use plain float64.
#   "f32": exact fp32-split convs; turns cuDNN and TF32 off process-wide while active.
#   "f64": plain float64 convs (cuDNN off on CUDA).
EXACT = {"mode": "i8"}


def _ent_ctx(dev: str):
    if EXACT["mode"] == "i8":
        return exact_i8_context() if str(dev).startswith("cuda") else nullcontext()
    if EXACT["mode"] == "f32":
        return exact_f32_context()
    return torch.backends.cudnn.flags(enabled=False) if str(dev).startswith("cuda") else nullcontext()


# ---------------------------------------------------------------- shared entropy pass
class _Pos:
    """Flat positions (C order over (c, h, w)) of the anchors (pass 0) / non-anchors (pass 1) of a c-channel slice,
    cached per device. Gathering with them instead of boolean masks avoids a host sync per indexing op (engine B,
    plans/GC_v2_architecture_2026-11-23.md 4.5); the order equals boolean-mask order, so streams are unchanged."""

    def __init__(self, hl, wl):
        self.hl, self.wl, self.cache = hl, wl, {}

    def __call__(self, c, p, dev):
        key = (c, p, str(dev))
        if key not in self.cache:
            ma = checkerboard(self.hl, self.wl, "cpu").bool().expand(1, c, self.hl, self.wl).reshape(-1)
            t = torch.nonzero(ma if p == 0 else ~ma).squeeze(1).to(dev)
            if t.is_cuda:
                torch.cuda.current_stream(t.device).synchronize()  # ready before another thread's stream reads it
            self.cache[key] = t
        return self.cache[key]


_POS: dict = {}


def _pos(hl, wl) -> _Pos:
    if (hl, wl) not in _POS:
        _POS[(hl, wl)] = _Pos(hl, wl)
    return _POS[(hl, wl)]


def _gather(pos: torch.Tensor, *ts: torch.Tensor):
    """Elements of each t at pos, brought to the host in ONE transfer; returns float64 numpy arrays."""
    g = torch.stack([t.reshape(-1).index_select(0, pos).double() for t in ts])
    return list(g.cpu().numpy())


def _scatter(vals: np.ndarray, pos: torch.Tensor, out: torch.Tensor):
    out.view(-1).index_copy_(0, pos, torch.from_numpy(vals).to(out.device, out.dtype))


@torch.no_grad()
def _entropy_pass(model: GCodec, z_q: torch.Tensor, cells: torch.Tensor, level: int, ent_dev, leaf, mu_dev=None):
    """z_q: (1, Z, hz, wz) integers. leaf(k, pass, idx, mu) gets the parameters of the positions coded in this pass
    (numpy, C order over (c, h, w)):
    v1 / E1 / E2 (integer mean): idx int64, mu = mu_int (int64, FRAC fractional bits) -> returns int64 y_int
    (= y_hat_s * 2**FRAC).
    split path: idx int64, mu = float mean (float64) -> returns int64 residual symbols q; y_hat_s = q + mu.
    Returns y_hat_s (1, M, hl, wl) float32 (on ent_dev for the integer path, on mu_dev for the split path)."""
    if model.split:
        return _entropy_pass_split(model, z_q, cells, level, ent_dev, leaf, mu_dev or ent_dev)
    lv = torch.tensor([level], dtype=torch.long, device=ent_dev)  # on device: no H2D copy (host sync) per layer
    lim = Y_CLAMP * 2 ** FRAC
    with _ent_ctx(ent_dev):
        hyp = model.h_s.forward_int(torch.clamp(z_q, -Z_CLAMP, Z_CLAMP).to(ent_dev, torch.float64), lv)
        _, _, hl, wl = hyp.shape
        pos = _pos(hl, wl)
        qint = model.qmap_latent(cells, hl, wl).to(ent_dev, torch.float64) * 2 ** FRAC
        prev, outs = [], []
        for k, c in enumerate(model.slices):
            pa, pn = pos(c, 0, ent_dev), pos(c, 1, ent_dev)
            y_int = torch.zeros(1, c, hl, wl, dtype=torch.float64, device=ent_dev)
            idx, mu = model.head.coding_params(model.ctx_a[k].forward_int(torch.cat([hyp, *prev, qint], 1), lv),
                                               keep_device=True)
            idx, mu = _gather(pa, idx, mu)
            _scatter(leaf(k, 0, idx.astype(np.int64), mu.astype(np.int64)), pa, y_int)
            ctx = torch.clamp(y_int, -lim, lim)
            if model.e1:
                ctx = model.loc[k].forward_int(ctx, lv)
            idx, mu = model.head.coding_params(model.ctx_n[k].forward_int(torch.cat([hyp, *prev, ctx, qint], 1), lv),
                                               keep_device=True)
            idx, mu = _gather(pn, idx, mu)
            _scatter(leaf(k, 1, idx.astype(np.int64), mu.astype(np.int64)), pn, y_int)
            outs.append(y_int)
            prev.append(torch.clamp(y_int, -lim, lim))
    return torch.cat(outs, 1).float() / 2 ** FRAC


@torch.no_grad()
def _entropy_pass_split(model: GCodec, z_q, cells, level, ent_dev, leaf, mu_dev):
    """Split path. sigma: integer nets on ent_dev (exact on any device) over hyper features, residual symbols q and
    the quality map. mu: float nets on mu_dev over the reconstructions; mu never decides a CDF, so a few ulp of
    difference between the encoder machine and the L4 only moves y_hat by as much (stream decoding is unaffected)."""
    lv_i = torch.tensor([level], dtype=torch.long, device=ent_dev)
    lv_f = torch.tensor([level], dtype=torch.long, device=mu_dev)
    lim = Y_CLAMP * 2 ** FRAC
    same = str(ent_dev) == str(mu_dev)
    with _ent_ctx(ent_dev):
        hyp = model.h_s.forward_int(torch.clamp(z_q, -Z_CLAMP, Z_CLAMP).to(ent_dev, torch.float64), lv_i)
        _, _, hl, wl = hyp.shape
        pos = _pos(hl, wl)
        qlat = model.qmap_latent(cells, hl, wl)
        qint = qlat.to(ent_dev, torch.float64) * 2 ** FRAC
        hyp_f = (hyp / 2 ** FRAC).to(mu_dev, torch.float32)
        qlat_f = qlat.to(mu_dev, torch.float32)
        ma_f = checkerboard(hl, wl, mu_dev)
        Yp, Qp, outs = [], [], []
        for k, c in enumerate(model.slices):
            pa, pn = pos(c, 0, ent_dev), pos(c, 1, ent_dev)
            pa_f, pn_f = (pa, pn) if same else (pos(c, 0, mu_dev), pos(c, 1, mu_dev))
            q = torch.zeros(1, c, hl, wl, dtype=torch.float64, device=ent_dev)
            yk = torch.zeros(1, c, hl, wl, dtype=torch.float32, device=mu_dev)
            # ---- anchors
            mu_a = model.mua[k](torch.cat([hyp_f, *Yp, qlat_f], 1), lv_f)
            if model.split == "hyper":
                idx_a = idx_n = model.sigma_index(model.sa[k].forward_int(torch.cat([hyp, qint], 1), lv_i))
            else:
                idx_a = model.sigma_index(model.sa[k].forward_int(torch.cat([hyp, *Qp, qint], 1), lv_i))
            if same:
                ia, ma_np = _gather(pa, idx_a, mu_a)
            else:
                (ia,), (ma_np,) = _gather(pa, idx_a), _gather(pa_f, mu_a)
            qa = leaf(k, 0, ia.astype(np.int64), ma_np)
            _scatter(qa.astype(np.float64), pa, q)
            _scatter(qa + ma_np, pa_f, yk)
            # ---- non-anchors
            ctx_y = torch.clamp(yk * ma_f, -Y_CLAMP, Y_CLAMP)
            mu_n = model.mun[k](torch.cat([hyp_f, *Yp, model.mloc[k](ctx_y, lv_f), qlat_f], 1), lv_f)
            if model.split == "q":
                ctx_q = model.sloc[k].forward_int(torch.clamp(q * 2 ** FRAC, -lim, lim), lv_i)
                idx_n = model.sigma_index(model.sn[k].forward_int(torch.cat([hyp, *Qp, ctx_q, qint], 1), lv_i))
            if same:
                i_n, mn_np = _gather(pn, idx_n, mu_n)
            else:
                (i_n,), (mn_np,) = _gather(pn, idx_n), _gather(pn_f, mu_n)
            qn = leaf(k, 1, i_n.astype(np.int64), mn_np)
            _scatter(qn.astype(np.float64), pn, q)
            _scatter(qn + mn_np, pn_f, yk)
            outs.append(yk)
            Yp.append(torch.clamp(yk, -Y_CLAMP, Y_CLAMP))
            Qp.append(torch.clamp(q * 2 ** FRAC, -lim, lim))
    return torch.cat(outs, 1)


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

    hl, wl = y_np.shape[-2:]
    ma = checkerboard(hl, wl, "cpu").bool().numpy()

    def leaf(k, p, idx, mu):
        c = model.slices[k]
        m = np.broadcast_to(ma if p == 0 else ~ma, (1, c, hl, wl)).ravel()
        yk = y_np[:, starts[k]:starts[k + 1]].ravel()[m]
        if model.split:  # mu is the float mean; code the residual symbol
            q = np.round(yk - mu).astype(np.int64)
            blobs.append(rans_c.encode_values(q, idx, cdf_y, half_y))
            return q
        q = np.round(yk - mu / 2 ** FRAC).astype(np.int64)
        blobs.append(rans_c.encode_values(q, idx, cdf_y, half_y))
        return q * (1 << FRAC) + mu

    y_hat_s = _entropy_pass(model, z_q, cells, level, ent_dev, leaf, mu_dev=dev).float().cpu()
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
def decompress_latent(model: GCodec, buf: bytes, ent_dev="cpu", mu_dev=None, keep_device: bool = False):
    """Entropy decoding only. Returns (y_hat_s, header, cells); y_hat_s on the CPU unless keep_device."""
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

    def leaf(k, p, idx, mu):
        b, state["pos"] = _get_blob(buf, state["pos"])
        q = rans_c.decode_values(b, idx, cdf_y, half_y)
        return q if model.split else q * (1 << FRAC) + mu

    y_hat_s = _entropy_pass(model, z_q, cells, hd["level"], ent_dev, leaf, mu_dev=mu_dev or ent_dev)
    if state["pos"] != len(buf):
        raise ValueError("trailing bytes in GC stream")
    return (y_hat_s if keep_device else y_hat_s.cpu()), hd, cells


@torch.no_grad()
def decompress(model: GCodec, buf: bytes, dev="cuda", ent_dev="cpu", synth_dtype=torch.float16, timings=None):
    """Returns HxWx3 uint8 numpy."""
    t0 = time.perf_counter()
    y_hat_s, hd, cells = decompress_latent(model, buf, ent_dev, mu_dev=dev, keep_device=True)
    t1 = time.perf_counter()
    img = synthesize_image(model, y_hat_s, hd, cells, dev, synth_dtype)
    if timings is not None:
        timings.update(entropy=t1 - t0, synthesis=time.perf_counter() - t1)
    return img


@torch.no_grad()
def synthesize_image(model: GCodec, y_hat_s, hd, cells, dev="cuda", synth_dtype=torch.float16):
    """Decoded latent -> HxWx3 uint8 numpy."""
    lv = torch.tensor([hd["level"]], device=dev)
    ctx = torch.autocast("cuda", dtype=synth_dtype) if (str(dev).startswith("cuda") and synth_dtype != torch.float32) \
        else nullcontext()
    with ctx:
        x = model.synthesize(y_hat_s.to(dev), lv, torch.tensor([hd["beta"]], device=dev), cells.to(dev))
    return (x[0, :, :hd["H"], :hd["W"]].float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
