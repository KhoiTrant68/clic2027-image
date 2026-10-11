# Round 8: decode time the way CLIC counts it. `encode`: DiT-IC q3 bitstreams for all 30 validation images, saved.
# `decode <tile>`: a fresh process loads the decoder, then decodes every image ONCE, in order (no warm-up per image:
# each new image size costs its first-call overhead, as on the CLIC server). Reports model load, first image, total.
import sys, time, glob, gc, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
STREAMS = "/tmp/ditic_streams_q3.pt"
VALID = "/mnt/home/guess03/clic/work_full_gpu/bakeoff/valid"
sys.argv = [sys.argv[0], VALID] + sys.argv[1:]
exec(open("/tmp/dcae_bench5.py").read().split("\ndef run" + "():")[0])  # load_net, patch, decode, M, Features, ...
from PIL import Image


def encode():
    net = load_net()
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=896, tile_sample_stride_width=896)
    out = {}
    for p in sorted(glob.glob(f"{VALID}/*.png")):
        x = np.asarray(Image.open(p).convert("RGB")); H, W = x.shape[:2]
        t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
        t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
        with torch.no_grad():
            o = net.compress(t)
        out[p.split("/")[-1][:8]] = (o["strings"], o["shape"], H, W)
    torch.save(out, STREAMS)
    print("encoded", len(out), "images", sum(sum(len(s[0]) for s in v[0]) for v in out.values()), "bytes", flush=True)


def decode_b(net, mods, strings, shape, H, W, tile, bucket):
    """bench5's decode, plus: the cropped latent padded (replicate) up to a multiple of `bucket` latents, so the
    30 images map onto a few tensor shapes and only the first of each shape pays the first-call cost."""
    sync = torch.cuda.synchronize
    sync(); t0 = time.time()
    mean, logs, res, prompt = net.codec.decompress(strings, shape)
    sc = torch.exp(0.5 * torch.clamp(logs, -30, 20))
    dit, prompter, vae = mods[True]
    enc = prompter(prompt.to(torch.bfloat16))
    ts = net.sched.base_scheduler.timesteps.expand(1) * dit.config.timestep_scale
    pred = dit(mean.to(torch.bfloat16), encoder_hidden_states=enc, timestep=ts.cuda(), return_dict=False)[0]
    z = (net.sched.step(pred.float(), sc, net.timesteps, mean, return_dict=True) + res) / vae.config.scaling_factor
    z = z[:, :, :min(z.shape[2], math.ceil(H / 32) + 2), :min(z.shape[3], math.ceil(W / 32) + 2)]
    if bucket:
        z = F.pad(z, (0, (-z.shape[3]) % bucket, 0, (-z.shape[2]) % bucket), mode="replicate")
    patch(vae, True)
    if tile:
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=tile - 128, tile_sample_stride_width=tile - 128)
    else:
        vae.disable_tiling()
    y = vae.decode(z.to(torch.bfloat16), return_dict=False)[0]
    sync()
    return y, time.time() - t0, tuple(z.shape[-2:])


def decode_all(tile, bucket=0, twice=False):
    streams = torch.load(STREAMS)
    torch.cuda.synchronize(); t0 = time.time()
    net = load_net()
    del net.aux_codec, net.vae.encoder
    mods = {True: tuple(m.to(torch.bfloat16) for m in (net.DiT, net.prompter, net.vae))}
    gc.collect(); torch.cuda.synchronize()
    t_load = time.time() - t0  # includes building the fp32 net from the 4.3 GB merged checkpoint; a submission would ship bf16
    ts, warm, shapes = [], [], set()
    feat = Features(M.device())
    q = []
    for n, (strings, shape, H, W) in streams.items():
        with torch.no_grad():
            y, t, zs = decode_b(net, mods, strings, shape, H, W, tile, bucket)
            u8 = to_u8(y, H, W)  # the PNG write is left out; the copy to the CPU is in
            if twice:
                y, t2, _ = decode_b(net, mods, strings, shape, H, W, tile, bucket)
                warm.append(t2)
        shapes.add(zs)
        ts.append(t)
        if feat is not None and len(q) < 8:
            x = np.asarray(Image.open(glob.glob(f"{VALID}/{n}*.png")[0]).convert("RGB"))
            f = feat(x, u8.permute(1, 2, 0).numpy())
            q.append((f["lpips_alex"], f["dists"]))
        del y
    extra = f" | same image again: mean {np.mean(warm[1:]):.3f} s" if twice else ""
    extra += f" | first 8 images LPIPS {np.mean([a for a, _ in q]):.4f} DISTS {np.mean([b for _, b in q]):.4f}" if q else ""
    print(f"tile {tile} bucket {bucket}: load {t_load:.1f} s | {len(shapes)} latent shapes | first image {ts[0]:.2f} s | "
          f"images 2..30 mean {np.mean(ts[1:]):.3f} s | decode total {sum(ts):.1f} s for {len(ts)} images" + extra, flush=True)


if sys.argv[2] == "encode":
    encode()
else:
    decode_all(int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0, len(sys.argv) > 5 and sys.argv[5] == "twice")
