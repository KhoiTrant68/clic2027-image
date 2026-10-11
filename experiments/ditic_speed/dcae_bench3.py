# Round 3: (a) quality of each tiling against the ORIGINAL image (not against another tiling), (b) torch.profiler
# breakdown of one bf16 decode. Uses the latents cached by dcae_bench.py. cwd = DiT-IC checkout, PYTHONPATH=.
import sys, time, glob, gc
import numpy as np, torch
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
exec(open("/tmp/dcae_bench.py").read().split("def main():")[0])  # load_net, to_u8, psnr, NAMES, CACHE
from clic27.eval import metrics as M
from qhat import Features


def decode(vae, z, dtype, tile, overlap):
    if tile:
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=tile - overlap, tile_sample_stride_width=tile - overlap)
    else:
        vae.disable_tiling()
    with torch.no_grad():
        return vae.decode(z.cuda().to(dtype), return_dict=False)[0]


def main():
    lat = torch.load(CACHE)
    net = load_net()
    vae = net.vae
    del net.DiT, net.aux_codec
    gc.collect(); torch.cuda.empty_cache()
    feat = Features(M.device())
    variants = [("fp32 tile1024/ov128 (now)", torch.float32, 1024, 128),
                ("bf16 tile1024/ov128", torch.bfloat16, 1024, 128),
                ("bf16 tile1280/ov64", torch.bfloat16, 1280, 64),
                ("bf16 tile1280/ov128", torch.bfloat16, 1280, 128),
                ("bf16 tile1280/ov256", torch.bfloat16, 1280, 256),
                ("bf16 tile1536/ov256", torch.bfloat16, 1536, 256)]
    for name, dt, tile, ov in variants:
        vae.to(dt)
        rows, ts = [], []
        for n, (z, H, W) in lat.items():
            gt = M.load(glob.glob(f"{VALID}/{n}*.png")[0])
            decode(vae, z, dt, tile, ov)
            torch.cuda.synchronize(); t0 = time.time()
            y = decode(vae, z, dt, tile, ov)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
            rec = to_u8(y, H, W).permute(1, 2, 0).numpy()
            f = feat(gt, rec)
            rows.append((f["psnr"], f["lpips_alex"], f["dists"]))
            del y
        r = np.mean(rows, 0)
        print(f"{name:28s} t {np.mean(ts):5.2f} s | vs original: PSNR {r[0]:6.3f}  LPIPS {r[1]:.4f}  DISTS {r[2]:.4f}", flush=True)
        vae.to(torch.float32)
        gc.collect(); torch.cuda.empty_cache()
    # profiler: one bf16 tile1280/ov128 decode of the first image
    vae.to(torch.bfloat16)
    z, H, W = next(iter(lat.values()))
    decode(vae, z, torch.bfloat16, 1280, 128)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        decode(vae, z, torch.bfloat16, 1280, 128)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=60), flush=True)


main()
