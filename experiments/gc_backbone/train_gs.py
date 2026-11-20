"""GC backbone ablation: train only the synthesis g_s of a frozen pretrained MSE codec with the GC GAN recipe.

    python train_gs.py --arm elic|mlicpp --quality Q --upstream REPO --weights DIR --data DIV2K --out DIR [--steps N]

The analysis transform, hyperprior and entropy model stay frozen, so the bitstream (and the rate) of every image is
exactly the one of the original checkpoint; only the decoder changes. Both arms use the same recipe as the GC `gan`
stage (src/clic27/gc/train.py): k_mse * MSE + k_lpips * LPIPS(alex) + k_gan * non-saturating GAN with the GC
conditional multi-scale PatchGAN (conditioned on y_hat), Adam 1e-4 (D: betas 0.5/0.999), grad clip 1, EMA 0.999,
bf16 autocast for g_s (D in fp32), lr x0.1 for the last 10 % of steps, random 256 crops of DIV2K train with random 0.5-1x downscale (built on the GPU, see GPUCrops), drawn from a fixed
pool of 32768 crops whose latents are computed once (same pool for both arms).
g_s starts from the MSE weights (HiFiC-style warm start). Not part of this ablation: rate levels, quality maps,
beta conditioning and the Q-hat v2 alignment stage of GC (identical for both backbones).

y_hat is captured with a forward pre-hook on g_s while running the upstream forward in eval mode
(round(y - mu) + mu, the same quantisation as compress/decompress); `--check` compares it with a real decode.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class _Captured(Exception):
    pass


class GPUCrops:
    """All training images decoded once and kept on the GPU as uint8; random crops built on the GPU.

    Same augmentation as clic27.gc.data.CropDataset (photo path): for images with min side > 2 * crop a random
    downscale s in [max(0.5, crop / min side), 1], then a random crop x crop window, random horizontal and vertical
    flips. The downscale is applied to the (crop / s)-sized window with area interpolation instead of resizing the
    whole image with PIL bicubic first; no CPU loader workers are used.
    """

    def __init__(self, folder, crop, dev, downscale=(0.5, 1.0)):
        import torch
        from PIL import Image
        files = sorted(Path(folder).glob('*.png'))
        if not files:
            raise SystemExit(f'no PNG in {folder}')
        self.imgs = []
        for f in files:
            with Image.open(f) as im:
                a = np.asarray(im.convert('RGB'), dtype=np.uint8)
            if min(a.shape[:2]) < crop:
                raise SystemExit(f'{f} smaller than the crop')
            self.imgs.append(torch.from_numpy(a.copy()).permute(2, 0, 1).to(dev))
        self.crop, self.lo, self.hi = crop, downscale[0], downscale[1]
        self.bytes = sum(t.numel() for t in self.imgs)

    def batch(self, B):
        import torch
        import torch.nn.functional as F
        out = []
        c = self.crop
        for _ in range(B):
            t = random.choice(self.imgs)
            H, W = t.shape[-2:]
            s = 1.0
            if min(H, W) > 2 * c:
                lo = max(self.lo, c / min(H, W))
                s = random.uniform(lo, max(lo, self.hi))
            w = min(H, W, max(c, round(c / s)))
            y0, x0 = random.randint(0, H - w), random.randint(0, W - w)
            v = t[:, y0:y0 + w, x0:x0 + w].float().div(255)[None]
            if w != c:
                v = F.interpolate(v, size=(c, c), mode='area')
            if random.random() < 0.5:
                v = v.flip(-1)
            if random.random() < 0.5:
                v = v.flip(-2)
            out.append(v)
        return torch.cat(out)


def load_probe(arm):
    """experiments/<arm>/probe.py under a unique module name (both runners are called probe.py)."""
    d = ROOT / 'experiments' / arm
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f'probe_{arm}', d / 'probe.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_codec(a):
    probe = load_probe(a.arm)
    ns = argparse.Namespace(upstream=a.upstream, weights=a.weights, legacy=True, device='cuda')
    model, meta = probe.load(ns, a.quality)
    model.eval().requires_grad_(False)
    return model, meta


def latent_fn(model):
    """x in [0,1] (B,3,H,W), H,W multiples of 64 -> y_hat exactly as fed to g_s by the upstream forward."""
    import torch
    box = {}

    def hook(_m, inp):
        box['y'] = inp[0].detach()
        raise _Captured

    handle = model.g_s.register_forward_pre_hook(hook)

    @torch.no_grad()
    def f(x):
        try:
            model(x)
        except _Captured:
            pass
        return box.pop('y')
    return f, handle


def check(a):
    """y_hat from the training path must reproduce the real decoder output (stream -> decompress)."""
    from PIL import Image
    torch, _ = load_probe(a.arm).settings()  # the runner's deterministic fp32 settings (entropy coding needs them)
    model, _ = build_codec(a)
    img = sorted(Path(a.check).glob('*.png'))[0]
    x = torch.from_numpy(np.array(Image.open(img).convert('RGB'))).permute(2, 0, 1)[None].float().div(255).cuda()
    h, w = x.shape[-2:]
    x = torch.nn.functional.pad(x, (0, (-w) % 64, 0, (-h) % 64))
    with torch.no_grad():
        enc = model.compress(x)
        dec = model.decompress(enc['strings'], enc['shape'])['x_hat'].clamp(0, 1)
        lat, handle = latent_fn(model)
        if a.tf32:  # measure what TF32 in the frozen encoder pass changes vs the real (fp32) decoder
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
            torch.use_deterministic_algorithms(False)
        y = lat(x)
        handle.remove()
        via = model.g_s(y).clamp(0, 1)
    diff = (dec - via).abs()
    res = dict(image=img.name, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
               psnr_dec=float(-10 * torch.log10(((dec - x) ** 2).mean())),
               psnr_via=float(-10 * torch.log10(((via - x) ** 2).mean())))
    print(json.dumps(res))
    if res['mean_abs'] > 1e-3:
        raise SystemExit('training-path y_hat does not match the real decoder')


def train(a):
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))
    from clic27.gc import config as C
    from clic27.gc.disc import MultiScaleD, d_loss, g_loss
    from clic27.gc.losses import LPIPS, psnr_t
    from half_vram import install

    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / 'vram.jsonl')
    torch.set_num_threads(2)  # shared server: keep the CPU footprint small (data pipeline runs on the GPU)
    seed = 1
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # Default fp32 without TF32, as in the runners, so the means that decide round(y - mu) + mu match the real
    # encoder exactly. --tf32 trades a measured, tiny mismatch (see --check --tf32) for speed.
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cudnn.benchmark = False  # its algorithm search asks for >19 GiB workspace on MLIC++ convs
    dev = 'cuda'
    Lw, st, T = C.DEFAULTS['loss'], C.DEFAULTS['stages']['gan'], C.DEFAULTS['train']

    model, meta = build_codec(a)
    lat, _ = latent_fn(model)
    gs = copy.deepcopy(model.g_s).train().requires_grad_(True)  # deepcopy also copies the hook; gs calls skip it below
    gs._forward_pre_hooks.clear()
    M = 320
    disc = MultiScaleD(M).to(dev)
    opt = torch.optim.Adam(gs.parameters(), lr=st['lr'])
    opt_d = torch.optim.Adam(disc.parameters(), lr=st['lr_d'], betas=(0.5, 0.999))
    ema = copy.deepcopy(gs).eval().requires_grad_(False)
    step = 0
    last = a.out / 'last.pt'
    if last.exists():
        s = torch.load(last, map_location='cpu')
        gs.load_state_dict(s['gs']); disc.load_state_dict(s['disc']); ema.load_state_dict(s['ema'])
        opt.load_state_dict(s['opt']); opt_d.load_state_dict(s['opt_d']); step = s['step']
        print(f'resume from step {step}', flush=True)
    lp = LPIPS().to(dev)
    # Latent cache: a fixed pool of crops (identical for every arm and quality: fixed seed, same images), encoded
    # once by the frozen codec in large batches. The MLIC++ entropy model is launch-bound at batch 8, so encoding
    # inside the loop made the MLIC++ arm ~3x slower than ELIC; training then only runs g_s, LPIPS and D.
    t_pool = time.time()
    random.seed(12345)
    data = GPUCrops(a.data, a.crop, dev)
    xs, ys = [], []
    with torch.no_grad():
        for i in range(0, a.pool, a.pool_batch):
            x = data.batch(min(a.pool_batch, a.pool - i)).mul(255).round()  # 8-bit, exactly what is stored
            ys.append(lat(x.div(255)).float().cpu())
            xs.append(x.to(torch.uint8).cpu())
    del data
    torch.cuda.empty_cache()
    xs, ys = torch.cat(xs).pin_memory(), torch.cat(ys).pin_memory()
    print(f'latent pool: {len(xs)} crops, x {xs.numel() / 2**30:.2f} GiB, y_hat {ys.numel() * 4 / 2**30:.2f} GiB, '
          f'{(time.time() - t_pool) / 60:.1f} min', flush=True)
    gen = torch.Generator().manual_seed(seed * 1000 + step)  # a resumed run continues with a different draw
    actx = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    write = lambda: torch.save(dict(gs=gs.state_dict(), disc=disc.state_dict(), ema=ema.state_dict(),
                                    opt=opt.state_dict(), opt_d=opt_d.state_dict(), step=step), last.with_suffix('.tmp'))
    (a.out / 'meta.json').write_text(json.dumps(dict(
        arm=a.arm, quality=a.quality, steps=a.steps, batch=a.batch, crop=a.crop, data=str(a.data), tf32=a.tf32,
        latent_pool=a.pool, pool_seed=12345,
        loss=dict(k_mse=Lw['k_mse'], k_lpips=Lw['k_lpips'], k_gan=Lw['k_gan']), lr=st['lr'], lr_d=st['lr_d'],
        ema=T['ema'], grad_clip=T['grad_clip'], codec=meta,
        gs_params=sum(p.numel() for p in gs.parameters())), indent=1))
    logf = (a.out / 'log.jsonl').open('a')
    t0, tl = time.time(), time.time()
    while step < a.steps:
        idx = torch.randint(len(xs), (a.batch,), generator=gen)
        x = xs[idx].to(dev, non_blocking=True).float().div(255)
        y = ys[idx].to(dev, non_blocking=True)
        B = x.shape[0]
        for g in opt.param_groups:
            g['lr'] = st['lr'] * (0.1 if step > 0.9 * a.steps else 1.0)
        with actx():
            xh = gs(y)
        xh = xh.float()
        mse = ((xh - x) ** 2).flatten(1).mean(1)
        lpv = lp(xh, x)
        disc.requires_grad_(False)
        fake = disc(xh, y)  # fp32: torch 2.0.1 has no bf16 CUDA nearest-upsample (used by the conditional D)
        disc.requires_grad_(True)
        gl = g_loss([f.float() for f in fake])
        loss = (Lw['k_mse'] * mse + Lw['k_lpips'] * lpv).mean() + Lw['k_gan'] * gl
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(gs.parameters(), T['grad_clip'])
        opt.step()
        d = min(T['ema'], (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), gs.parameters()):
                pe.lerp_(p.detach(), 1 - d)
            for be, b in zip(ema.buffers(), gs.buffers()):
                be.copy_(b)
        both = disc(torch.cat([x, xh.detach()]), torch.cat([y, y]))
        rl, fl = [b[:B].float() for b in both], [b[B:].float() for b in both]
        dl_ = d_loss(rl, fl)
        opt_d.zero_grad(set_to_none=True)
        dl_.backward()
        opt_d.step()
        step += 1
        if step % 100 == 0 or step == 1:
            rec = dict(step=step, loss=float(loss), mse=float(mse.mean()), psnr=float(psnr_t(xh.detach(), x).mean()),
                       lpips=float(lpv.mean()), g=float(gl), d=float(dl_), gnorm=float(gn),
                       d_real=float(sum(torch.sigmoid(r).mean() for r in rl) / len(rl)),
                       d_fake=float(sum(torch.sigmoid(f).mean() for f in fl) / len(fl)),
                       it_s=(100 if step > 1 else 1) / max(time.time() - tl, 1e-9), min=(time.time() - t0) / 60)
            tl = time.time()
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            print(json.dumps(rec), flush=True)
        if step % 2000 == 0 or step == a.steps:
            write(); last.with_suffix('.tmp').replace(last)
    torch.save(dict(g_s=ema.state_dict(), arm=a.arm, quality=a.quality, step=step), a.out / 'gs.pt')
    (a.out / 'done.json').write_text(json.dumps(dict(step=step, minutes=(time.time() - t0) / 60)))
    stop.set()
    print('DONE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=['elic', 'mlicpp'], required=True)
    p.add_argument('--quality', type=int, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--pool', type=int, default=32768, help='crops in the fixed latent pool')
    p.add_argument('--pool-batch', type=int, default=32)
    p.add_argument('--tf32', action='store_true', help='allow TF32 (frozen encoder pass and training)')
    p.add_argument('--check', help='folder of PNGs: verify the y_hat capture against a real decode and exit')
    a = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if a.check:
        check(a)
    else:
        train(a)


if __name__ == '__main__':
    main()
