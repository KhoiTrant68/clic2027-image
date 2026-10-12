"""Distil the DC-AE decoder of one DiT-IC checkpoint into the lighter student of common.py.

Per step: random latent crops (from prep.py, DIV2K) -> the shared frozen prefix (teacher's conv_in + EfficientViT
stages) -> teacher suffix gives the target, student suffix is trained with L1 + LPIPS(VGG) against it. No original
images are needed: the student learns the teacher's function on the latents DiT-IC actually produces.

    python train.py --q q3 --lat ~/clic/distill/lat_q3 --out ~/clic/distill/run_q3 --steps 30000
Resumes from <out>/last.pt. Second stage (sharper textures): --init <first stage best.pt> --gan 0.1 adds a PatchGAN
(spectral norm, hinge) that takes the teacher's outputs as real; --snap-every keeps the trained weights every N steps
(choose by Q-hat with eval.py, not by the distance to the teacher, which a GAN raises).
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TRAINED, STUDENT_CHANNELS, build_student, load_ditic, load_student_weights,  # noqa: E402
                    patch_resblocks, prefix, res_forward_fast, res_forward_train, suffix)

BF16 = torch.bfloat16
TRAINED_PREFIXES = tuple(f"up_blocks.{i}." for i in TRAINED) + ("norm_out.", "conv_out.")


def patch_disc(ch=64, n=3):
    """pix2pix PatchGAN with spectral norm instead of normalisation layers; 70x70 receptive field."""
    sn = torch.nn.utils.parametrizations.spectral_norm
    layers, c = [sn(torch.nn.Conv2d(3, ch, 4, 2, 1)), torch.nn.LeakyReLU(0.2, True)], ch
    for i in range(1, n + 1):
        c2 = min(ch * 2 ** i, 512)
        layers += [sn(torch.nn.Conv2d(c, c2, 4, 2 if i < n else 1, 1)), torch.nn.LeakyReLU(0.2, True)]
        c = c2
    layers += [sn(torch.nn.Conv2d(c, 1, 4, 1, 1))]
    return torch.nn.Sequential(*layers)


def latents(d):
    return [torch.load(p)["z"] for p in sorted(Path(d).glob("*.pt"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q3")
    ap.add_argument("--lat", type=Path, required=True, help="dir with train/ and clic_val/ from prep.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=12, help="latent crop (x32 px): 12 = 384x384")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--w-lpips", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", default=",".join(map(str, STUDENT_CHANNELS)),
                    help="student block_out_channels; the last three must be the teacher's (512,1024,1024)")
    ap.add_argument("--init", type=Path, default=None, help="start the student from this checkpoint (weights only)")
    ap.add_argument("--gan", type=float, default=0.0, help="weight of the generator hinge loss (0: no GAN)")
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--d-warmup", type=int, default=500, help="steps the discriminator trains before G gets its loss")
    ap.add_argument("--snap-every", type=int, default=0, help="save the trained weights every N steps (0: off)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"

    net = load_ditic(a.q)
    teacher = net.vae.decoder
    del net.DiT, net.aux_codec, net.vae.encoder
    teacher.to(BF16).eval().requires_grad_(False)
    patch_resblocks(teacher, res_forward_fast)
    channels = tuple(int(c) for c in a.channels.split(","))
    student = build_student(teacher, channels).to(dev)
    patch_resblocks(student, res_forward_train)
    student.requires_grad_(False)
    train_mods = [student.up_blocks[i] for i in TRAINED] + [student.norm_out, student.conv_out]
    params = [p for m in train_mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in params)
    n_s = sum(p.numel() for p in student.parameters())
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"teacher decoder {n_t / 1e6:.1f} M params, student {n_s / 1e6:.1f} M ({n_tr / 1e6:.1f} M trained), "
          f"channels {channels}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        f = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, f))))

    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)

    train = latents(a.lat / "train")
    val = latents(a.lat / "clic_val")
    vc = 16
    val_z = torch.cat([z[:, :, (z.shape[2] - vc) // 2:(z.shape[2] + vc) // 2, (z.shape[3] - vc) // 2:(z.shape[3] + vc) // 2]
                       for z in val]).to(dev).to(BF16)
    print(f"{len(train)} train latents, {len(val)} CLIC val crops of {vc * 32} px", flush=True)
    with torch.no_grad():
        val_h = torch.cat([prefix(teacher, val_z[i:i + 6]) for i in range(0, len(val_z), 6)])
        val_t = torch.cat([suffix(teacher, val_h[i:i + 6]).float().clamp(-1, 1) for i in range(0, len(val_h), 6)])

    disc = patch_disc().to(dev) if a.gan else None
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.d_lr, betas=(0.5, 0.9)) if a.gan else None
    step, best = 0, float("inf")
    ck = a.out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        student.load_state_dict(s["student"])
        opt.load_state_dict(s["opt"])
        if disc is not None and "disc" in s:
            disc.load_state_dict(s["disc"])
            opt_d.load_state_dict(s["opt_d"])
        step, best = s["step"], s.get("best", best)
        print(f"resumed at step {step}", flush=True)
    elif a.init:
        s = torch.load(a.init, map_location="cpu", weights_only=False)
        load_student_weights(student, s)
        print(f"student initialised from {a.init} (step {s.get('step')})", flush=True)

    def evaluate():
        student.eval()
        l1s, lps, mses = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=BF16):
            for i in range(0, len(val_h), 6):
                o = suffix(student, val_h[i:i + 6]).float().clamp(-1, 1)
                t = val_t[i:i + 6]
                l1s.append((o - t).abs().mean(dim=(1, 2, 3)))
                mses.append((o - t).pow(2).mean(dim=(1, 2, 3)))
                lps.append(lp(o, t).flatten())
        student.train()
        l1, lpv = torch.cat(l1s).mean().item(), torch.cat(lps).mean().item()
        psnr = torch.cat(mses).clamp_min(1e-12).reciprocal().mul(4).log10().mul(10).mean().item()  # [-1, 1] range
        return l1, lpv, psnr

    def save(path, extra=None):
        d = dict(student=student.state_dict(), opt=opt.state_dict(), step=step, best=best, q=a.q,
                 channels=channels, **(extra or {}))
        if disc is not None:
            d.update(disc=disc.state_dict(), opt_d=opt_d.state_dict())
        torch.save(d, path)

    student.train()
    t0, acc = time.time(), [0.0, 0.0, 0.0, 0.0, 0.0]
    while step < a.steps:
        zs = []
        for _ in range(a.bs):
            z = random.choice(train)
            y = random.randint(0, z.shape[2] - a.crop)
            x = random.randint(0, z.shape[3] - a.crop)
            zs.append(z[:, :, y:y + a.crop, x:x + a.crop])
        z = torch.cat(zs).to(dev).to(BF16)
        with torch.no_grad():
            h = prefix(teacher, z)
            tgt = suffix(teacher, h).float().clamp(-1, 1)
        with torch.autocast("cuda", dtype=BF16):
            out = suffix(student, h).float()
            lpl = lp(out, tgt).float().mean()  # VGG in bf16: fp32 activations at 8 x 384^2 exceed the 10 GB cap
        l1 = (out - tgt).abs().mean()
        loss = l1 + a.w_lpips * lpl
        g_adv = torch.zeros((), device=dev)
        if disc is not None:
            if step >= a.d_warmup:
                disc.requires_grad_(False)
                with torch.autocast("cuda", dtype=BF16):
                    g_adv = -disc(out).float().mean()
                loss = loss + a.gan * g_adv
            disc.requires_grad_(True)
            with torch.autocast("cuda", dtype=BF16):
                d_loss = (torch.relu(1 - disc(tgt)).float().mean() + torch.relu(1 + disc(out.detach())).float().mean())
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            acc[4] += d_loss.item()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        acc[0] += loss.item(); acc[1] += l1.item(); acc[2] += lpl.item(); acc[3] += g_adv.item()
        if step % 100 == 0:
            dt = (time.time() - t0) / 100
            gan = f" g_adv {acc[3] / 100:+.3f} d {acc[4] / 100:.3f}" if disc is not None else ""
            print(f"step {step} loss {acc[0] / 100:.4f} l1 {acc[1] / 100:.4f} lpips {acc[2] / 100:.4f}{gan} "
                  f"lr {lr_at(step):.2e} {dt:.3f} s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            t0, acc = time.time(), [0.0, 0.0, 0.0, 0.0, 0.0]
        if a.snap_every and step % a.snap_every == 0:
            torch.save(dict(trained={k: v for k, v in student.state_dict().items() if k.startswith(TRAINED_PREFIXES)},
                            step=step, q=a.q, channels=channels), a.out / f"snap_{step}.pt")
        if step % a.eval_every == 0 or step == a.steps:
            l1v, lpv, ps = evaluate()
            score = l1v + a.w_lpips * lpv
            tag = ""
            if score < best:
                best = score
                save(a.out / "best.pt", dict(val=dict(l1=l1v, lpips=lpv)))
                tag = " (best)"
            save(ck)
            print(f"== eval step {step}: CLIC val vs teacher L1 {l1v:.4f} LPIPS {lpv:.4f} ~PSNR {ps:.1f} dB{tag}", flush=True)


if __name__ == "__main__":
    main()
