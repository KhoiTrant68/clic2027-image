# DC-AE decoder speed/precision bench for DiT-IC's decoder (run with cwd = the DiT-IC checkout, PYTHONPATH=.).
# 1) latents: DiT-IC q3 encode -> entropy decode -> one-step DiT for a few validation images (cached in /tmp).
# 2) decode the same latents with several settings; time = median of 3 runs after a warm-up, per image.
#    Quality = PSNR of each variant's uint8 output against the fp32 full-frame output (or fp32 tiled if that OOMs).
import sys, time, types, glob, os, math, gc
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from PIL import Image
from huggingface_hub import hf_hub_download

VALID = sys.argv[1]
NAMES = ["02809272", "0369d229", "07b9f93f", "8426ed22"]  # 2048x1360 portrait, landscape x2, 2048x2048
CACHE = "/tmp/dcae_latents.pt"


def load_net():
    import models
    class Unused(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
    for mod, attr, val in (("models.losses", "LPIPSWithDiscriminator", Unused), ("models.foundation_models", "aux_foundation_model", None)):
        sys.modules[mod] = types.ModuleType(mod); setattr(sys.modules[mod], attr, val)
    from models.DiT_IC import Codec
    cfg = yaml.safe_load(open("configs/inference_merge.yaml"))["model"]["params"]
    cfg.update(dit_path="Efficient-Large-Model/Sana_600M_1024px_diffusers", use_ema=True, use_merge=True, training=False)
    net = Codec("cuda", "Codec", **dict(cfg, codec_path=hf_hub_download("JunqiShi/DiT-IC", "q3_merge_ema.pt"))).cuda().eval()
    net.codec.update(force=True)
    return net


def latents(net):
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=896, tile_sample_stride_width=896)
    out = {}
    for n in NAMES:
        p = glob.glob(f"{VALID}/{n}*.png")[0]
        x = np.asarray(Image.open(p).convert("RGB")); H, W = x.shape[:2]
        t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
        t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
        with torch.no_grad():
            o = net.compress(t)
            mean, logs, res, prompt = net.codec.decompress(o["strings"], o["shape"])
            sc = torch.exp(0.5 * torch.clamp(logs, -30, 20))
            enc = net.prompter(prompt)
            ts = net.sched.base_scheduler.timesteps.expand(1) * net.DiT.config.timestep_scale
            pred = net.DiT(mean, encoder_hidden_states=enc, timestep=ts.cuda(), return_dict=False)[0]
            xd = net.sched.step(pred, sc, net.timesteps, mean, return_dict=True) + res
        out[n] = (xd.float().cpu() / net.vae.config.scaling_factor, H, W)
    return out


def to_u8(y, H, W):
    return y[0, :, :H, :W].float().clamp(-1, 1).add(1).mul(127.5).round().byte().cpu()


def psnr(a, b):
    m = (a.float() - b.float()).pow(2).mean().item()
    return 99.0 if m == 0 else 10 * math.log10(255 ** 2 / m)


def run(vae, lat, dtype, tile, cl, bench, overlap=None):
    torch.backends.cudnn.benchmark = bench
    dec = vae
    dec.to(dtype)
    if cl:
        dec.to(memory_format=torch.channels_last)
    if tile:
        stride = tile - overlap if overlap else tile * 7 // 8
        dec.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=stride, tile_sample_stride_width=stride)
    else:
        dec.disable_tiling()
    times, outs, peak = [], {}, 0
    for n, (z, H, W) in lat.items():
        z = z.cuda().to(dtype)
        if cl:
            z = z.contiguous(memory_format=torch.channels_last)
        ts = []
        torch.cuda.reset_peak_memory_stats()
        for r in range(4):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                y = dec.decode(z, return_dict=False)[0]
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        times.append(float(np.min(ts[1:])))  # min: the GPUs are shared, other jobs add noise
        outs[n] = to_u8(y, H, W)
        del y
    dec.to(torch.float32).to(memory_format=torch.contiguous_format)
    return times, outs, peak


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.path.exists(CACHE):
        lat = torch.load(CACHE)
        from diffusers import AutoencoderDC
        net = None
    else:
        net = load_net()
        lat = latents(net)
        torch.save(lat, CACHE)
    if net is None:
        net = load_net()
    vae = net.vae
    vae.decoder.requires_grad_(False)
    del net.DiT, net.aux_codec
    gc.collect(); torch.cuda.empty_cache()
    print("images:", {n: (H, W) for n, (_, H, W) in lat.items()}, flush=True)
    variants = [  # round 2: bf16 only (round 1: bf16 weights 2.7x faster than fp32, full frame over the 10 GB cap)
        ("fp32 tile1024 (now)", torch.float32, 1024, False, False, False, None),
        ("bf16 tile1024", torch.bfloat16, 1024, False, False, False, None),
        ("bf16 tile1024 +CL", torch.bfloat16, 1024, True, False, False, None),
        ("bf16 tile1024 +CL+bench", torch.bfloat16, 1024, True, True, False, None),
        ("bf16 tile1056/ov64 +CL+bench", torch.bfloat16, 1056, True, True, False, 64),
        ("bf16 tile1280/ov64 +CL+bench", torch.bfloat16, 1280, True, True, False, 64),
        ("bf16 tile1536/ov64 +CL+bench", torch.bfloat16, 1536, True, True, False, 64),
        ("fp16 tile1056/ov64 +CL+bench", torch.float16, 1056, True, True, False, 64),
    ]
    res = {}
    for name, dt, tile, cl, bench, tf32, ov in variants:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            times, outs, peak = run(vae, lat, dt, tile, cl, bench, ov)
        except (torch.OutOfMemoryError, RuntimeError) as e:  # over the cap; the NVML-less allocator raises RuntimeError
            print(f"{name:34s} OOM ({type(e).__name__})", flush=True)
            gc.collect(); torch.cuda.empty_cache(); vae.to(torch.float32).to(memory_format=torch.contiguous_format)
            continue
        res[name] = (times, outs, peak)
        print(f"{name:34s} " + " ".join(f"{t:5.2f}" for t in times) + f" | mean {np.mean(times):5.2f} s  peak {peak:4.1f} GB",
              flush=True)
        gc.collect(); torch.cuda.empty_cache()
    refname = "fp32 full frame" if "fp32 full frame" in res else "fp32 tile1024 (now)"
    ref = res[refname][1]
    print("min PSNR over the images vs", refname, flush=True)
    for name, (_, outs, _) in res.items():
        print(f"  {name:34s} {min(psnr(outs[n], ref[n]) for n in outs):5.1f} dB", flush=True)


main()
