"""Is DiT-IC's entropy coding deterministic across devices? (2026-10-12)

DiT-IC decodes y in four checkerboard groups: float networks (h_s, the spatial context, adapters) give the scales and
means of each group; the scales are bucketed into CompressAI's 64-entry scale table (build_indexes), the index picks
the CDF for rANS, and each decoded group feeds the next. If the decoder's float scales differ from the encoder's
enough to cross a table boundary, rANS decodes garbage from there on. The CLIC server decodes on an L4; we encode
elsewhere. This script encodes the codec input of a few validation images on the CPU and on the GPU, decodes every
bitstream on both, and counts index / symbol mismatches; it also reports how close the scales come to a table
boundary (the margin a cross-device difference has to beat).

    python ditic_check.py --q q3 --valid ~/clic/work_full_gpu/bakeoff/valid --n 6
Run with the DiT-IC checkout importable (experiments/ditic_distill/common.py: DITIC_DIR).
"""
import argparse
import copy
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ditic_distill"))
from common import load_ditic  # noqa: E402

PICK = ["02809272", "8426ed22", "ebfd571f", "bb7344a2", "937476dd", "50fe4c3d", "a365e654", "2684452d"]


def record_decode(codec):
    """Wrap decompress_group_with_mask to keep each group's indexes, decoded symbols and scales."""
    codec.log = []
    orig = type(codec).decompress_group_with_mask

    def wrapped(self, gc, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        sq = self.sequeeze_with_mask(scales, mask)
        idx = gc.build_indexes(sq)
        out = orig(self, gc, scales, means, mask, decoder, cdf, cdf_lengths, offsets)
        sym = self.sequeeze_with_mask(out, mask) - self.sequeeze_with_mask(means, mask)
        self.log.append(dict(idx=idx.cpu().reshape(-1), sym=sym.round().cpu().reshape(-1), scale=sq.detach().cpu().reshape(-1)))
        return out

    codec.decompress_group_with_mask = types.MethodType(wrapped, codec)


def chain_decompress(c, strings, shape):
    """LatentCodec.decompress up to y_hat: only what decides the rANS indexes (z_hat, h_s, the four context groups
    with their LRP). Runs in the codec's own dtype. Returns y_hat and the four groups' scales."""
    from compressai.ans import RansDecoder
    z_hat = c.entropy_bottleneck.decompress(strings[1], shape)
    cdf = c.gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = c.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    offsets = c.gaussian_conditional.offset.reshape(-1).int().tolist()
    dec = RansDecoder()
    dec.set_stream(strings[0][0])
    B, C, H, W = z_hat.shape
    masks = c.get_mask_four_parts(B, C * 2, H * 4, W * 4, device=z_hat.device)
    base = c.h_s(z_hat)
    y_hat, scales, prev = 0, [], None
    for g, mask in enumerate(masks):
        if g:
            base = base * (1 - masks[g - 1]) + prev
        means, sc = c.adapter_out[g](c.g_c(c.adapter_in[g](base))).chunk(2, 1)
        yg = c.decompress_group_with_mask(c.gaussian_conditional, sc, means, mask, dec, cdf, cdf_lengths, offsets)
        yg = yg + 0.5 * torch.tanh(c.LRP[g](torch.cat([yg, base], dim=1)) * mask)
        prev = yg
        y_hat = y_hat + yg
        scales.append(sc)
    return y_hat, scales


def rest_decompress(c, y_hat, scales):
    """The rest of LatentCodec.decompress (inputs for the DiT; no influence on entropy decoding)."""
    B, C, H, W = y_hat.shape
    masks = c.get_mask_four_parts(B, C, H, W, device=y_hat.device)
    return c.g_s(y_hat), c.scale(c.scale_with_mask(*scales, *masks)), c.aux(y_hat), c.prompt(y_hat)


def margin(scales, table):
    """Smallest relative distance |log(s) - log(boundary)| of each scale to a table boundary. Scales at or below the
    lower bound (table[0]) are clamped to exactly that value on every device, so they are left out."""
    s = scales[scales > table[0]].double().log()
    b = torch.as_tensor(table[:-1], dtype=torch.float64).log()
    return (s[:, None] - b[None, :]).abs().min(1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q3")
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"],
                    help="precision of the latent codec (the networks that set the entropy parameters) on both devices")
    ap.add_argument("--no-tf32", action="store_true", help="disable TF32 convolutions on the GPU")
    ap.add_argument("--devs", default="cpu,cuda", help="the two devices compared, e.g. cpu,cuda or cuda:0,cuda:1")
    ap.add_argument("--time-split", action="store_true",
                    help="also time the GPU decode split into the entropy chain (codec dtype) and the rest (fp32)")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
    dt = torch.float64 if a.dtype == "fp64" else torch.float32
    torch.manual_seed(0)
    net = load_ditic(a.q, device="cpu")
    del net.DiT
    dev_a, dev_b = a.devs.split(",")  # "cpu"/"gpu" below mean device A / device B
    base = net.codec.eval().to(dt)
    codec_cpu = copy.deepcopy(base).to(dev_a).eval()
    codec_gpu = copy.deepcopy(base).to(dev_b).eval()
    table = base.gaussian_conditional.scale_table.tolist()
    print(f"device A = {dev_a}, device B = {dev_b}", flush=True)
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=896, tile_sample_stride_width=896)
    for name in PICK[:a.n]:
        p = next(a.valid.glob(f"{name}*.png"))
        x = np.asarray(Image.open(p).convert("RGB"))
        H, W = x.shape[:2]
        t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1)
        t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
        with torch.no_grad():
            latent2 = net.aux_codec((t + 1) / 2)
            latent = net.vae.encode(t).latent * net.vae.config.scaling_factor
            latent, latent2 = latent.to(dt), latent2.to(dt)
            t0 = time.time()
            enc = {"cpu": codec_cpu.compress(latent.to(dev_a), latent2.to(dev_a))}
            t_cpu = time.time() - t0
            enc["gpu"] = codec_gpu.compress(latent.to(dev_b), latent2.to(dev_b))
        same_bits = enc["cpu"]["strings"] == enc["gpu"]["strings"]
        nbytes = {k: sum(len(s[0]) for s in v["strings"]) for k, v in enc.items()}
        line = [f"{name} {H}x{W}: cpu/gpu bitstreams {'identical' if same_bits else 'DIFFER'} ({nbytes['cpu']} / {nbytes['gpu']} B)"]
        tdec = {}
        for src in ("cpu", "gpu"):
            res = {}
            for dev, codec in (("cpu", codec_cpu), ("gpu", codec_gpu)):
                record_decode(codec)
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    out = codec.decompress(enc[src]["strings"], enc[src]["shape"])
                torch.cuda.synchronize(); tdec[dev] = time.time() - t0
                res[dev] = (codec.log, [o.float().cpu() for o in out])
            lc, lg = res["cpu"][0], res["gpu"][0]
            idx_bad = sum(int((c["idx"] != g["idx"]).sum()) for c, g in zip(lc, lg))
            sym_bad = sum(int((c["sym"] != g["sym"]).sum()) for c, g in zip(lc, lg))
            n_sym = sum(c["idx"].numel() for c in lc)
            rel = torch.cat([((c["scale"] - g["scale"]).abs() / c["scale"].abs().clamp_min(1e-6)) for c, g in zip(lc, lg)])
            m = margin(torch.cat([c["scale"] for c in lc]), table)
            dmean = max(float((oc - og).abs().max()) for oc, og in zip(res["cpu"][1], res["gpu"][1]))
            line.append(f"  stream from {src}: index mismatches {idx_bad}/{n_sym}, symbol mismatches {sym_bad}, "
                        f"scale rel. diff max {float(rel.max()):.1e} (p99.9 {float(rel.quantile(0.999)):.1e}), "
                        f"scales within 1e-4 / 1e-5 of a boundary: {int((m < 1e-4).sum())} / {int((m < 1e-5).sum())}, "
                        f"max |decoded output cpu - gpu| {dmean:.2e} | codec decode cpu {tdec['cpu']:.2f} s gpu {tdec['gpu']:.2f} s")
        if a.time_split:
            c32 = copy.deepcopy(codec_gpu).float()
            with torch.no_grad():
                ref = codec_gpu.decompress(enc["gpu"]["strings"], enc["gpu"]["shape"])
                for _ in range(2):  # the second pass is the one reported
                    torch.cuda.synchronize(); t0 = time.time()
                    y_hat, scales = chain_decompress(codec_gpu, enc["gpu"]["strings"], enc["gpu"]["shape"])
                    torch.cuda.synchronize(); t1 = time.time()
                    out = rest_decompress(c32, y_hat.float(), [sc.float() for sc in scales])
                    torch.cuda.synchronize(); t2 = time.time()
            d = float((out[0] - ref[0].float()).abs().max())
            line.append(f"  GPU decode split: entropy chain ({a.dtype}) {t1 - t0:.3f} s + rest (fp32) {t2 - t1:.3f} s; "
                        f"mean output vs the all-{a.dtype} decode: max |diff| {d:.2e}")
            del c32
        print("\n".join(line), flush=True)
    print(f"dtype {a.dtype} | cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}",
          flush=True)


if __name__ == "__main__":
    main()
