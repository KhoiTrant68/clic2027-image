# Round 7: how much would a compiled DC-AE decoder gain (upper bound for shipping it AOT-compiled with AOTInductor,
# so no compile time at decode)? Eager (bf16, NCHW RMSNorm) vs torch.compile on one fixed tile: latent 40x40 = a
# 1280x1280 output, the default tile. Min of 5 runs after warm-up; compile time reported separately.
import sys, time, gc
import numpy as np, torch
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/src")
sys.path.insert(0, "/mnt/home/guess03/clic/clic2027-image/experiments/qhat")
exec(open("/tmp/dcae_bench4.py").read().split("\ndef main" + "():")[0])  # load_net, patch, psnr, ...


def timeit(fn, z, n=5):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            y = fn(z)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return min(ts), y


def main():
    lat = torch.load(CACHE)
    net = load_net()
    dec = net.vae.decoder.to(torch.bfloat16).eval()
    del net.DiT, net.aux_codec, net.vae.encoder
    gc.collect(); torch.cuda.empty_cache()
    patch(net.vae, True)
    z = next(iter(lat.values()))[0][:, :, :40, :40].cuda().to(torch.bfloat16).contiguous()
    with torch.no_grad():
        dec(z)
    t_eager, y_eager = timeit(dec, z)
    print(f"eager bf16 + NCHW norm, 1280x1280 tile: {t_eager:.3f} s", flush=True)
    for mode in ("default", "max-autotune-no-cudagraphs"):
        torch._dynamo.reset()
        comp = torch.compile(dec, mode=mode, dynamic=False)
        t0 = time.time()
        with torch.no_grad():
            comp(z)
        torch.cuda.synchronize()
        t_comp = time.time() - t0
        t, y = timeit(comp, z)
        print(f"torch.compile {mode:28s}: {t:.3f} s ({t_eager / t:.2f}x)  compile {t_comp:.0f} s  "
              f"vs eager {psnr(y.float().clamp(-1, 1).add(1).mul(127.5).round().byte(), y_eager.float().clamp(-1, 1).add(1).mul(127.5).round().byte()):.1f} dB",
              flush=True)


main()
