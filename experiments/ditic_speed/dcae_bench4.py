# Round 4: replace diffusers' RMSNorm path in the DC-AE decoder ResBlocks (movedim + full fp32 copy) with an NCHW
# version: fp32-accumulated reduction over channels, then in-place scale and addcmul with the residual. Variants are
# timed interleaved per image (the GPUs are shared), min of 3; quality vs the bf16 baseline and vs the original.
import sys, time, glob, gc, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
exec(open("/tmp/dcae_bench.py").read().split("def main():")[0])  # load_net, to_u8, psnr, NAMES, CACHE, VALID
from clic27.eval import metrics as M
from qhat import Features
from diffusers.models.autoencoders.autoencoder_dc import ResBlock


def rms_scale(h, norm):
    """1 / RMS over channels, (N, 1, H, W) in h's dtype; the sum of squares accumulates in fp32."""
    ss = torch.linalg.vector_norm(h, ord=2, dim=1, keepdim=True, dtype=torch.float32)
    return torch.rsqrt(ss.square_().div_(h.shape[1]).add_(norm.eps)).to(h.dtype)


def res_forward(self, x):
    h = self.conv2(F.silu(self.conv1(x), inplace=True) if isinstance(self.nonlinearity, nn.SiLU) else self.nonlinearity(self.conv1(x)))
    n = self.norm
    h.mul_(rms_scale(h, n))
    out = torch.addcmul(x, h, n.weight.view(1, -1, 1, 1))
    return out.add_(n.bias.view(1, -1, 1, 1)) if n.bias is not None else out


def dec_forward(self, hidden_states):
    if self.in_shortcut:
        x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1,
                                            output_size=hidden_states.shape[1] * self.in_shortcut_repeats)
        hidden_states = self.conv_in(hidden_states) + x
    else:
        hidden_states = self.conv_in(hidden_states)
    for up_block in reversed(self.up_blocks):
        hidden_states = up_block(hidden_states)
    n = self.norm_out
    h = hidden_states * rms_scale(hidden_states, n)
    h = torch.addcmul(n.bias.view(1, -1, 1, 1), h, n.weight.view(1, -1, 1, 1)) if n.bias is not None else h * n.weight.view(1, -1, 1, 1)
    return self.conv_out(self.conv_act(h))


def patch(vae, on):
    for m in vae.decoder.modules():
        if isinstance(m, ResBlock):
            if on:
                m.forward = types.MethodType(res_forward, m)
            elif "forward" in m.__dict__:
                del m.forward
    if on:
        vae.decoder.forward = types.MethodType(dec_forward, vae.decoder)
    elif "forward" in vae.decoder.__dict__:
        del vae.decoder.forward


def main():
    lat = torch.load(CACHE)
    net = load_net()
    vae = net.vae
    del net.DiT, net.aux_codec
    gc.collect(); torch.cuda.empty_cache()
    vae.to(torch.bfloat16)
    feat = Features(M.device())
    variants = [("bf16 tile1280/ov128 (round 3)", False, False, 1280, 128),
                ("+ fast norm", True, False, 1280, 128),
                ("+ fast norm + channels_last", True, True, 1280, 128),
                ("+ fast norm tile1024/ov128", True, False, 1024, 128)]
    times = {v[0]: [] for v in variants}
    outs = {v[0]: {} for v in variants}
    peak = {v[0]: 0.0 for v in variants}
    for n, (z, H, W) in lat.items():
        for rep in range(4):
            for name, fast, cl, tile, ov in variants:
                patch(vae, fast)
                vae.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
                vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                                  tile_sample_stride_height=tile - ov, tile_sample_stride_width=tile - ov)
                zz = z.cuda().to(torch.bfloat16)
                if cl:
                    zz = zz.contiguous(memory_format=torch.channels_last)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    y = vae.decode(zz, return_dict=False)[0]
                torch.cuda.synchronize()
                if rep:
                    times[name].append((n, time.time() - t0))
                else:
                    outs[name][n] = to_u8(y, H, W)
                peak[name] = max(peak[name], torch.cuda.max_memory_allocated() / 2**30)
                del y
    base = variants[0][0]
    for name, *_ in variants:
        per = [min(t for m, t in times[name] if m == n) for n in lat]
        q = []
        for n, (z, H, W) in lat.items():
            gt = M.load(glob.glob(f"{VALID}/{n}*.png")[0])
            f = feat(gt, outs[name][n].permute(1, 2, 0).numpy())
            q.append((f["psnr"], f["lpips_alex"], f["dists"]))
        q = np.mean(q, 0)
        d = min(psnr(outs[name][n], outs[base][n]) for n in lat)
        print(f"{name:32s} " + " ".join(f"{t:5.2f}" for t in per) + f" | mean {np.mean(per):5.2f} s peak {peak[name]:4.1f} GB"
              f" | vs base {d:5.1f} dB | vs original PSNR {q[0]:.3f} LPIPS {q[1]:.4f} DISTS {q[2]:.4f}", flush=True)


main()
