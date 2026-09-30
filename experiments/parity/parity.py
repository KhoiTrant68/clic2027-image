"""Parity and determinism checks for src/ratflow/{nn,entropy} (needs torch + GPU; run on Kaggle).

  dcae      our DCAE vs diffusers AutoencoderDC: encode/decode on a Kodak image and a tiled 2048x1360 input
  dit       our SanaDiT vs diffusers SanaTransformer2DModel (0.6B fp32, 1.6B fp16) on random inputs
  entropy   integer h_s: forward_int CPU == GPU bit for bit; hyperprior + Gaussian conditional round trip
            (compress on one device, decompress on the other) and rate vs. ideal
  summary   PASS/FAIL table -> parity_out/summary.md

python parity.py all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ratflow.entropy import tables  # noqa: E402
from ratflow.entropy.intnet import IntSequential, QConv2d, Up2, check_exact_range  # noqa: E402
from ratflow.entropy.models import DiscretePrior, GaussianConditional, ScaleMeanHead  # noqa: E402
from ratflow.nn.dcae import DCAE  # noqa: E402
from ratflow.nn.sana import SanaDiT  # noqa: E402

SANA = {"0.6B": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "1.6B": "Efficient-Large-Model/Sana_1600M_1024px_diffusers"}
KODAK = "http://r0k.us/graphics/kodak/kodak/kodim23.png"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def diff(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return dict(max_abs=float((a - b).abs().max()), rel=float((a - b).norm() / (b.norm() + 1e-12)),
                bit_exact=bool(torch.equal(a, b)))


def save(out, key, val):
    p = out / "report.json"
    rep = json.loads(p.read_text()) if p.exists() else {}
    rep[key] = val
    p.write_text(json.dumps(rep, indent=1))
    print(key, json.dumps(val))


def kodak_tensor():
    import urllib.request
    from io import BytesIO

    from PIL import Image
    img = np.asarray(Image.open(BytesIO(urllib.request.urlopen(KODAK).read())).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1)[None].float().div(127.5).sub(1)


# ---------------------------------------------------------------- DC-AE
def dcae(args):
    from diffusers import AutoencoderDC
    ref = AutoencoderDC.from_pretrained(SANA["1.6B"], subfolder="vae", torch_dtype=torch.float32).to(DEV).eval()
    ours = DCAE.from_pretrained(SANA["1.6B"], "vae").to(DEV)
    x = kodak_tensor().to(DEV)  # 1x3x512x768
    with torch.no_grad():
        zr, zo = ref.encode(x).latent, ours.encode(x)
        save(args.out, "dcae/encode", diff(zo, zr))
        save(args.out, "dcae/decode", diff(ours.decode(zr), ref.decode(zr).sample))
        torch.manual_seed(0)
        big = torch.rand(1, 3, 1360, 2048, device=DEV) * 2 - 1  # CLIC-sized, exercises tiling
        ref.enable_tiling()
        ours.enable_tiling()
        zr = ref.encode(big).latent
        save(args.out, "dcae/tiled_encode", diff(ours.encode(big), zr))
        save(args.out, "dcae/tiled_decode", diff(ours.decode(zr), ref.decode(zr).sample))
        if DEV == "cuda":
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                ours.decode(zr)
            torch.cuda.synchronize()
            save(args.out, "dcae/decode_2048x1360_fp32_s", (time.time() - t0) / 3)
    del ref, ours
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- SANA DiT
def dit(args):
    from diffusers import SanaTransformer2DModel
    for name, dtype in (("0.6B", torch.float32), ("1.6B", torch.float16)):
        torch.manual_seed(0)
        x = torch.randn(2, 32, 32, 32, device=DEV, dtype=dtype)
        t = torch.tensor([999.0, 250.0], device=DEV)
        ctx = torch.randn(2, 20, 2304, device=DEV, dtype=dtype)
        mask = torch.ones(2, 20, device=DEV)
        mask[1, 12:] = 0
        ref = SanaTransformer2DModel.from_pretrained(SANA[name], subfolder="transformer", torch_dtype=dtype).to(DEV).eval()
        with torch.no_grad():
            yr = ref(x, encoder_hidden_states=ctx, timestep=t, encoder_attention_mask=mask).sample.cpu()
        del ref
        gc.collect()
        torch.cuda.empty_cache()
        ours = SanaDiT.from_pretrained(SANA[name], "transformer", dtype=dtype).to(DEV)
        with torch.no_grad():
            yo = ours(x, t, ctx, mask).cpu()
        save(args.out, f"dit/{name}_{str(dtype).split('.')[-1]}", diff(yo, yr))
        del ours
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- entropy
def make_hs(M=320, N=192, mu_frac=4):
    """A Ballé-style hyper-synthesis with integer layers: z (int) -> [scale acc, mean acc]."""
    net = IntSequential(
        Up2(), QConv2d(N, N, 5, in_frac=0, out_frac=7),
        Up2(), QConv2d(N, 288, 5, in_frac=7, out_frac=7),
        QConv2d(288, 2 * M, 3, in_frac=7, act=None),
    )
    return net, ScaleMeanHead(net[-1].acc_frac, mu_frac)


def entropy(args):
    torch.manual_seed(0)
    M, N, mu_frac = 320, 192, 4
    hs, head = make_hs(M, N, mu_frac)
    for m in hs:
        if isinstance(m, QConv2d):
            torch.nn.init.normal_(m.weight, std=0.5 / (m.in_channels * m.kernel_size[0] ** 2) ** 0.5)
            torch.nn.init.normal_(m.bias, std=0.1)
    save(args.out, "entropy/acc_bound_log2", float(np.log2(check_exact_range(hs, 64))))
    prior = DiscretePrior(N)
    prior.update_tables()
    gcond = GaussianConditional()

    z = torch.round(torch.randn(1, N, 16, 24) * 3)  # hyper-latent of a ~2K image
    acc_cpu = hs.forward_int(z.double())
    acc_gpu = hs.cuda().forward_int(z.double().cuda()).cpu() if DEV == "cuda" else acc_cpu
    hs.cpu()
    save(args.out, "entropy/hs_int_cpu_vs_gpu", diff(acc_gpu, acc_cpu))

    # float (training) path agrees with the integer path up to float32 rounding
    with torch.no_grad():
        acc_f = hs(z.float()) * 2 ** hs[-1].acc_frac
    save(args.out, "entropy/hs_float_vs_int", diff(acc_f, acc_cpu))

    # full round trip: encoder on GPU-side acc, decoder on CPU-side acc (identical ints -> must decode)
    idx_e, mu_e = head.coding_params(acc_gpu)
    idx_d, mu_d = head.coding_params(acc_cpu)
    sigma = torch.from_numpy(tables.scale_table()[idx_e.numpy()])
    y = torch.randn(idx_e.shape, dtype=torch.float64) * sigma + mu_e.double() / 2 ** mu_frac
    zb = prior.compress(z)
    yb = gcond.compress(y.float(), idx_e, mu_e, mu_frac)
    z_dec = prior.decompress(zb, tuple(z.shape))
    y_dec, _ = gcond.decompress(yb, idx_d, mu_d, mu_frac)
    y_ref = torch.round(y - mu_e.double() / 2 ** mu_frac) + mu_e.double() / 2 ** mu_frac
    ok = torch.equal(z_dec, z) and torch.equal(y_dec.double(), y_ref)
    _, bits = GaussianConditional.bits(y.float(), sigma.float(), mu_e.float() / 2 ** mu_frac, training=False)
    save(args.out, "entropy/roundtrip", dict(ok=bool(ok), bytes_y=len(yb), bytes_z=len(zb),
                                              ideal_bytes_y=float(bits.sum() / 8), n_y=int(y.numel())))


# ---------------------------------------------------------------- summary
CHECKS = [  # key, field, threshold, meaning
    ("dcae/encode", "rel", 1e-5, "DC-AE encode khớp diffusers"),
    ("dcae/decode", "rel", 1e-5, "DC-AE decode khớp diffusers"),
    ("dcae/tiled_encode", "rel", 1e-5, "DC-AE encode theo tile khớp"),
    ("dcae/tiled_decode", "rel", 1e-5, "DC-AE decode theo tile khớp"),
    ("dit/0.6B_float32", "rel", 1e-4, "SANA 0.6B fp32 khớp"),
    ("dit/1.6B_float16", "rel", 5e-3, "SANA 1.6B fp16 khớp"),
    ("entropy/hs_int_cpu_vs_gpu", "bit_exact", True, "h_s số nguyên: CPU == GPU từng bit"),
    ("entropy/hs_float_vs_int", "rel", 1e-3, "đường float (train) sát đường số nguyên"),
    ("entropy/roundtrip", "ok", True, "nén/giải nén khớp hoàn toàn"),
]


def summary(args):
    rep = json.loads((args.out / "report.json").read_text())
    L = ["# Parity: ratflow.nn / ratflow.entropy so với diffusers và giữa các thiết bị", "",
         "| kiểm tra | giá trị | ngưỡng | kết quả |", "|---|---|---|---|"]
    for key, field, thr, desc in CHECKS:
        if key not in rep:
            L.append(f"| {desc} (`{key}`) | – | – | CHƯA CHẠY |")
            continue
        v = rep[key][field]
        ok = (v == thr) if isinstance(thr, bool) else (v <= thr)
        L.append(f"| {desc} (`{key}`) | {v:.3g} | {thr} | {'PASS' if ok else 'FAIL'} |" if not isinstance(v, bool)
                 else f"| {desc} (`{key}`) | {v} | {thr} | {'PASS' if ok else 'FAIL'} |")
    for k in ("dcae/decode_2048x1360_fp32_s", "entropy/acc_bound_log2", "entropy/roundtrip"):
        if k in rep:
            L.append(f"\n`{k}`: {rep[k]}")
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dcae", "dit", "entropy", "summary", "all"])
    ap.add_argument("--out", type=Path, default=Path("parity_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for s in (["dcae", "dit", "entropy", "summary"] if args.step == "all" else [args.step]):
        print(f"===== {s}")
        globals()[s](args)


if __name__ == "__main__":
    main()
