"""A0: does the entropy-model path give bit-identical results on different devices?

Run the same h_s (hyper-synthesis) on each device and save scale_idx. Then compare
with compare.py. Inputs and weights are generated with numpy (PCG64), so they are
bit-identical on every machine; any difference comes from the compute itself.

Modes:
  float32    h_s in fp32 (TF32 off) -> scale -> index in a 64-level log table (CompressAI style)
  float32tf  like float32 but TF32 on (the PyTorch default for conv on Ampere+/Ada, e.g. L4)
  bf16       h_s in bf16
  intsim     "integer network": int8 weights, integer activations, requantised with floor(x / 2^s).
             Computed in float64; every partial sum is an integer < 2^53, so the result is
             exact and independent of summation order -> it should match on every device.

Usage:
  python probe.py --device cuda --tag l4        # writes out/<tag>.npz
  python probe.py --device cpu  --tag cpu
  python compare.py out/*.npz
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Latent sizes to test: (channels of z, H_z, W_z). Hyper-latent is 1/4 of y spatially.
# y for a ~2K image on the codec over DC-AE f32 latents is roughly 64x96; z is 16x24.
SIZES = [(192, 16, 24), (192, 32, 48)]
N_Z, N_MID, M_Y = 192, 288, 320
SCALE_MIN, SCALE_MAX, LEVELS = 0.11, 256.0, 64
SCALE_TABLE = np.exp(np.linspace(np.log(SCALE_MIN), np.log(SCALE_MAX), LEVELS))


def make_weights(rng):
    """Ballé-2018-like h_s: 3 transposed-conv-ish layers, done as upsample + conv so the
    int path is easy to write. Weights ~ He init so activations stay O(1)."""
    shapes = [(N_Z, N_Z, 5, 5), (N_MID, N_Z, 5, 5), (M_Y, N_MID, 3, 3)]
    ws, bs = [], []
    for co, ci, kh, kw in shapes:
        std = np.sqrt(2.0 / (ci * kh * kw))
        # compute in float64, cast once: float32_array * np.float64 is float32 in numpy 1.x but
        # float64 in numpy 2.x (NEP 50), which made weights differ between torch 2.6 / 2.10 images
        ws.append((rng.standard_normal((co, ci, kh, kw)) * std).astype(np.float32))
        bs.append((rng.standard_normal(co) * 0.1).astype(np.float32))
    return ws, bs


def make_input(rng, c, h, w):
    # quantized hyper-latent: small integers, like a real ẑ at low rate
    return np.round(rng.laplace(0.0, 1.5, size=(1, c, h, w))).astype(np.float32)


def hs_float(z, ws, bs, dtype):
    x = z.to(dtype)
    for i, (w, b) in enumerate(zip(ws, bs)):
        if i < 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.conv2d(x, w.to(dtype), b.to(dtype), padding=w.shape[-1] // 2)
        x = F.relu(x) if i < 2 else x
    # softplus-ish positive scale, as in CompressAI's exp/abs parameterisations
    return F.softplus(x.float()) + SCALE_MIN


def scale_to_index(scales: torch.Tensor) -> np.ndarray:
    # CompressAI: index = number of table entries <= scale, clamped (bucketize on thresholds)
    table = torch.tensor(SCALE_TABLE[:-1], dtype=scales.dtype, device=scales.device)
    return torch.bucketize(scales, table).to(torch.uint8).cpu().numpy()


# ---------- integer simulation ----------
W_BITS, A_SHIFT = 7, 7  # int8 weights (scale 2^7); activations requantised by >> 7


def quantize_weights(ws, bs):
    qws = [np.clip(np.round(w * 2**W_BITS), -127, 127).astype(np.float64) for w in ws]
    # bias lives in the accumulator domain (input scale 1 * weight scale 2^W_BITS)
    qbs = [np.round(b * 2**W_BITS).astype(np.float64) for b in bs]
    return qws, qbs


def hs_int(z, qws, qbs):
    """Every value is an integer stored in float64 -> exact, order-independent arithmetic.
    Max |partial sum| ~ 127 * 2^15 * 288*25 < 2^53, so no rounding ever happens."""
    x = z.double()
    for i, (w, b) in enumerate(zip(qws, qbs)):
        if i < 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        acc = F.conv2d(x, w, b, padding=w.shape[-1] // 2)
        if i < 2:
            x = torch.clamp(torch.floor(torch.relu(acc) / 2**A_SHIFT), 0, 2**15 - 1)
        else:
            x = acc  # final layer: accumulator, scale 2^W_BITS
    # Map the integer accumulator to a table index with integer thresholds precomputed
    # from the float table: idx = #{thresholds <= acc}. Thresholds are integers too.
    thr = np.round(_inv_softplus(SCALE_TABLE[:-1] - SCALE_MIN) * 2**W_BITS)
    thr_t = torch.tensor(thr, dtype=torch.float64, device=x.device)
    return torch.bucketize(x, thr_t, right=True).to(torch.uint8).cpu().numpy()


def _inv_softplus(y):
    y = np.maximum(y, 1e-6)
    return np.log(np.expm1(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ws_np, bs_np = make_weights(rng)
    inputs = [make_input(rng, *s) for s in SIZES]
    qws_np, qbs_np = quantize_weights(ws_np, bs_np)

    ws = [torch.from_numpy(w).to(dev) for w in ws_np]
    bs = [torch.from_numpy(b).to(dev) for b in bs_np]
    qws = [torch.from_numpy(w).to(dev) for w in qws_np]
    qbs = [torch.from_numpy(b).to(dev) for b in qbs_np]

    res = {}
    for k, z_np in enumerate(inputs):
        z = torch.from_numpy(z_np).to(dev)
        for mode in ["float32", "float32tf", "bf16", "intsim"]:
            torch.backends.cudnn.allow_tf32 = mode == "float32tf"
            torch.backends.cuda.matmul.allow_tf32 = mode == "float32tf"
            t0 = time.time()
            try:
                with torch.no_grad():
                    if mode == "intsim":
                        res[f"s{k}_{mode}_idx"] = hs_int(z, qws, qbs)
                    else:
                        dtype = torch.bfloat16 if mode == "bf16" else torch.float32
                        sc = hs_float(z, ws, bs, dtype)
                        res[f"s{k}_{mode}_scale"] = sc.cpu().numpy()
                        res[f"s{k}_{mode}_idx"] = scale_to_index(sc)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
            except RuntimeError as e:  # e.g. bf16 conv unsupported on P100
                print(f"size {SIZES[k]} {mode:10s} SKIPPED: {str(e).splitlines()[0]}")
                continue
            print(f"size {SIZES[k]} {mode:10s} {time.time() - t0:.3f}s")

    meta = dict(
        tag=args.tag, device=str(dev), torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=np.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor(),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.tag}.npz", meta=np.array(str(meta)), **res)
    print("saved", out / f"{args.tag}.npz", meta)


if __name__ == "__main__":
    main()
