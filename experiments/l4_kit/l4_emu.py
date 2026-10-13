"""Helpers for emulate_l4.sh, a rough L4 stand-in on an L40 of our server (2026-10-14).

    python l4_emu.py hog --stop FILE [--ready FILE]   # stream memory until FILE exists (an MPS client with few SMs)
    python l4_emu.py calib --tag T                    # copy bandwidth, bf16 GEMM and 3x3 conv throughput seen here

Both chips are Ada (sm_89) and do the same work per SM and clock. L4: 58 SMs at <= 2040 MHz under a 72 W cap,
300 GB/s, 48 MB L2. L40: 142 SMs at <= 2490 MHz, 864 GB/s, 96 MB L2. So an L4 is ~0.34 of an L40 in compute at
full clock (less once it hits 72 W) and ~0.35 in bandwidth.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def cmd_hog(a):
    x = torch.zeros(2**28, device="cuda")  # 1 GiB
    y = torch.empty_like(x)
    if a.ready:
        Path(a.ready).touch()
    while not Path(a.stop).exists():
        for _ in range(50):
            y.copy_(x)
            x.copy_(y)
        torch.cuda.synchronize()


def cmd_calib(a):
    x = torch.zeros(2**28, device="cuda")
    y = torch.empty_like(x)
    copy = 2 * x.numel() * 4 / timed(lambda: y.copy_(x), 50) / 1e9
    del x, y
    m = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    gemm = 2 * 8192**3 / timed(lambda: m @ m, 100) / 1e12
    w = torch.randn(256, 256, 3, 3, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(1, 256, 1024, 1024, device="cuda", dtype=torch.bfloat16)
    conv = 2 * 256 * 256 * 9 * 1024 * 1024 / timed(lambda: F.conv2d(z, w, padding=1), 20) / 1e12
    print("CALIB " + json.dumps(dict(tag=a.tag, copy_gbs=round(copy), gemm_tflops=round(gemm, 1),
                                     conv_tflops=round(conv, 1), gpu=torch.cuda.get_device_name(0),
                                     sm_pct=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hog")
    h.add_argument("--stop", required=True)
    h.add_argument("--ready", default=None)
    c = sub.add_parser("calib")
    c.add_argument("--tag", default="")
    a = ap.parse_args()
    dict(hog=cmd_hog, calib=cmd_calib)[a.cmd](a)


if __name__ == "__main__":
    main()
