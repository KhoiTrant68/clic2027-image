# Round 5: the whole DiT-IC decoder (entropy decode -> one-step DiT -> DC-AE), as in the bake-off (fp32, DC-AE tiled
# 1024/128) against the optimised path (DiT + DC-AE weights in bf16, NCHW RMSNorm, tile 1280/128 or full frame,
# latent cropped to the image + 2 latent rows/cols instead of the 256-px padding). The latent codec stays fp32
# (entropy decoding must stay deterministic). fp32 and bf16 are separate model copies. Interleaved per image,
# min of 3 timed runs.
import sys, time, glob, gc, math, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
exec(open("/tmp/dcae_bench4.py").read().split("\ndef main" + "():")[0])  # load_net, patch, to_u8, Features, M, VALID
from PIL import Image

NAMES8 = ["02809272", "0369d229", "07b9f93f", "8426ed22", "2684452d", "a365e654", "ebfd571f", "50fe4c3d"]


def decode(net, mods, strings, shape, H, W, opt):
    t = {}
    sync = torch.cuda.synchronize
    sync(); t0 = time.time()
    mean, logs, res, prompt = net.codec.decompress(strings, shape)
    sync(); t["entropy"] = time.time() - t0
    sc = torch.exp(0.5 * torch.clamp(logs, -30, 20))
    dt = torch.bfloat16 if opt["bf16"] else torch.float32
    dit, prompter, vae = mods[opt["bf16"]]
    enc = prompter(prompt.to(dt))
    ts = net.sched.base_scheduler.timesteps.expand(1) * dit.config.timestep_scale
    pred = dit(mean.to(dt), encoder_hidden_states=enc.to(dt), timestep=ts.cuda(), return_dict=False)[0]
    xd = net.sched.step(pred.float(), sc, net.timesteps, mean, return_dict=True) + res
    sync(); t["dit"] = time.time() - t0 - t["entropy"]
    z = xd / vae.config.scaling_factor
    if opt["crop"]:
        z = z[:, :, :min(z.shape[2], math.ceil(H / 32) + 2), :min(z.shape[3], math.ceil(W / 32) + 2)]
    patch(vae, opt["fast"])
    if opt["tile"]:
        vae.enable_tiling(tile_sample_min_height=opt["tile"], tile_sample_min_width=opt["tile"],
                          tile_sample_stride_height=opt["tile"] - 128, tile_sample_stride_width=opt["tile"] - 128)
    else:
        vae.disable_tiling()
    y = vae.decode(z.to(dt), return_dict=False)[0]
    sync(); t["dcae"] = time.time() - t0 - t["entropy"] - t["dit"]
    t["total"] = time.time() - t0
    return y, t


def run():
    net = load_net()
    variants = [("bake-off now: fp32, tile1024", dict(bf16=False, fast=False, tile=1024, crop=False)),
                ("bf16 + fast norm, tile1280", dict(bf16=True, fast=True, tile=1280, crop=False)),
                ("bf16 + fast norm, tile1280, crop", dict(bf16=True, fast=True, tile=1280, crop=True)),
                ("bf16 + fast norm, full frame, crop", dict(bf16=True, fast=True, tile=0, crop=True))]
    streams = {}
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,  # the encoder too: 10 GB cap
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
    mods = {False: (net.DiT, net.prompter, net.vae)}
    mods[True] = tuple(copy.deepcopy(m).to(torch.bfloat16) for m in mods[False])
    gc.collect(); torch.cuda.empty_cache()
    feat = Features(M.device())
    res = {v[0]: {"t": {}, "q": {}, "oom": False} for v in variants}
    for n, (strings, shape, H, W, x) in streams.items():
        for rep in range(4):
            for name, opt in variants:
                if res[name]["oom"]:
                    continue
                try:
                    with torch.no_grad():
                        y, t = decode(net, mods, strings, shape, H, W, opt)
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    print(f"{name}: {n} {type(e).__name__} (over the 10 GB cap?)", flush=True)
                    res[name]["oom"] = True
                    gc.collect(); torch.cuda.empty_cache()
                    continue
                if rep == 0:
                    f = feat(x, to_u8(y, H, W).permute(1, 2, 0).numpy())
                    res[name]["q"][n] = (f["psnr"], f["lpips_alex"], f["dists"])
                else:
                    old = res[name]["t"].get(n)
                    res[name]["t"][n] = t if old is None or t["total"] < old["total"] else old
                del y
    for name, r in res.items():
        if not r["t"]:
            print(f"{name:36s} no result", flush=True)
            continue
        ks = ("entropy", "dit", "dcae", "total")
        m = {k: np.mean([r["t"][n][k] for n in r["t"]]) for k in ks}
        q = np.mean([r["q"][n] for n in r["q"]], 0)
        print(f"{name:36s} n={len(r['t'])} " + "  ".join(f"{k} {m[k]:.3f}" for k in ks) +
              f" | vs original PSNR {q[0]:.3f} LPIPS {q[1]:.4f} DISTS {q[2]:.4f}", flush=True)


run()
