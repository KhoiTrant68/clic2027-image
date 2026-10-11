# Round 6: (1) the DC-AE up blocks (nearest x2 then conv3x3 at the high resolution) computed exactly as four 2x2 convs
# at the low resolution, one per output phase, interleaved by pixel_shuffle: 16/36 of the FLOPs of that conv and no
# 2x-upsampled copy of the input; (2) full-frame decoding (no tiles). Whole decoder on 8 images, interleaved, min of 3.
import sys, time, glob, gc, math, copy, types
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
exec(open("/tmp/dcae_bench5.py").read().split("\ndef run" + "():")[0])  # decode, load_net, patch, Features, M, ...
from diffusers.models.autoencoders.autoencoder_dc import DCUpBlock2d

# R[p]: how the 3 taps of the high-res kernel fold onto the 2 low-res rows seen by output phase p
# (p = 0: rows a-1, a get W0, W1+W2; p = 1: rows a, a+1 get W0+W1, W2).
R = torch.tensor([[[1., 0., 0.], [0., 1., 1.]], [[1., 1., 0.], [0., 0., 1.]]])


def up_prepare(m):
    W = m.conv.weight.detach().float()
    K = torch.stack([torch.einsum("ik,ockl,jl->ocij", R[p].to(W.device), W, R[q].to(W.device))
                     for p in (0, 1) for q in (0, 1)], 1)  # (Co, 4, Ci, 2, 2): channel c*4 + p*2 + q, pixel_shuffle order
    m.sp_w = K.reshape(-1, W.shape[1], 2, 2).to(m.conv.weight.dtype)
    m.sp_b = m.conv.bias.detach().repeat_interleave(4) if m.conv.bias is not None else None


def up_forward(self, h):
    N, _, H, W = h.shape
    Y = F.conv2d(F.pad(h, (1, 1, 1, 1)), self.sp_w, self.sp_b)  # (N, 4 Co, H+1, W+1); row o sees rows o-1, o
    Y = Y.view(N, -1, 4, H + 1, W + 1)
    x = torch.stack([Y[:, :, 0, :H, :W], Y[:, :, 1, :H, 1:], Y[:, :, 2, 1:, :W], Y[:, :, 3, 1:, 1:]], 2)
    x = F.pixel_shuffle(x.reshape(N, -1, H, W), 2)
    if self.shortcut:
        y = h.repeat_interleave(self.repeats, dim=1, output_size=h.shape[1] * self.repeats)
        x.add_(F.pixel_shuffle(y, 2))
    return x


def subpixel(vae, on):
    for m in vae.decoder.modules():
        if isinstance(m, DCUpBlock2d) and m.interpolate and m.interpolation_mode == "nearest" and m.factor == 2:
            if on:
                if not hasattr(m, "sp_w"):
                    up_prepare(m)
                m.forward = types.MethodType(up_forward, m)
            elif "forward" in m.__dict__:
                del m.forward


def run6():
    net = load_net()
    variants = [("default now: bf16+norm, tile1280, crop", dict(bf16=True, fast=True, tile=1280, crop=True, sp=False)),
                ("+ sub-pixel up conv", dict(bf16=True, fast=True, tile=1280, crop=True, sp=True)),
                ("+ sub-pixel, full frame", dict(bf16=True, fast=True, tile=0, crop=True, sp=True)),
                ("full frame, no sub-pixel", dict(bf16=True, fast=True, tile=0, crop=True, sp=False))]
    streams = {}
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=896, tile_sample_stride_width=896)
    for n in NAMES8:
        p = glob.glob(f"{VALID}/{n}*.png")[0]
        x = np.asarray(Image.open(p).convert("RGB")); H, W = x.shape[:2]
        t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
        t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
        with torch.no_grad():
            o = net.compress(t)
        streams[n] = (o["strings"], o["shape"], H, W, x)
    del net.aux_codec, net.vae.encoder
    mods = {True: tuple(m.to(torch.bfloat16) for m in (net.DiT, net.prompter, net.vae))}
    gc.collect(); torch.cuda.empty_cache()
    feat = Features(M.device())
    res = {v[0]: {"t": {}, "q": {}, "peak": 0.0, "fail": []} for v in variants}
    outs = {v[0]: {} for v in variants}
    for n, (strings, shape, H, W, x) in streams.items():
        for rep in range(4):
            for name, opt in variants:
                if n in res[name]["fail"]:
                    continue
                subpixel(net.vae, opt["sp"])
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.no_grad():
                        y, t = decode(net, mods, strings, shape, H, W, opt)
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    res[name]["fail"].append(n)
                    gc.collect(); torch.cuda.empty_cache()
                    continue
                res[name]["peak"] = max(res[name]["peak"], torch.cuda.max_memory_allocated() / 2**30)
                if rep == 0:
                    u8 = to_u8(y, H, W)
                    outs[name][n] = u8
                    f = feat(x, u8.permute(1, 2, 0).numpy())
                    res[name]["q"][n] = (f["psnr"], f["lpips_alex"], f["dists"])
                else:
                    old = res[name]["t"].get(n)
                    res[name]["t"][n] = t if old is None or t["total"] < old["total"] else old
                del y
    base = variants[0][0]
    common = [n for n in NAMES8 if all(n in r["t"] for r in res.values())]
    print("images every variant decoded:", len(common), "| failures (over the 10 GB cap):",
          {k: v["fail"] for k, v in res.items() if v["fail"]}, flush=True)
    for name, r in res.items():
        ok = [n for n in NAMES8 if n in r["t"]]
        if not ok:
            print(f"{name:40s} no result", flush=True)
            continue
        ks = ("entropy", "dit", "dcae", "total")
        m_all = {k: np.mean([r["t"][n][k] for n in ok]) for k in ks}
        m_com = np.mean([r["t"][n]["total"] for n in common]) if common else float("nan")
        q = np.mean([r["q"][n] for n in ok], 0)
        d = min(psnr(outs[name][n], outs[base][n]) for n in ok if n in outs[base])
        print(f"{name:40s} n={len(ok)} " + "  ".join(f"{k} {m_all[k]:.3f}" for k in ks) +
              f" | total on common {m_com:.3f} | peak {r['peak']:.1f} GB | vs default {d:.1f} dB"
              f" | vs original PSNR {q[0]:.3f} LPIPS {q[1]:.4f} DISTS {q[2]:.4f}", flush=True)


run6()
