"""Train S1 (variable-rate latent codec) on cached DC-AE latents.

    python train_s1.py selftest                       # random model: GPU-encode / CPU-decode bit-exactness, bpp sanity
    python train_s1.py train --latents latents --out s1_out --hours 10
    (resumes from s1_out/last.pt if it exists)

Loss per sample: bpp + lambda[r] * MSE(latent_hat, latent), r ~ U{0..n_rates-1}, latents scaled by the DC-AE
scaling factor (~unit variance). bpp counts y and z bits over the crop's pixels (crop latent h*w*32*32).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ratflow.codec.latent_codec import LatentCodec  # noqa: E402

SCALING = 0.41407  # DC-AE f32c32 (SANA) scaling factor
# run 1 used [0.03 .. 4.0] -> Kodak 0.0155-0.12 bpp, saturating near the DC-AE ceiling above ~0.1 bpp;
# run 2 shifts down to cover ~0.010-0.08 bpp (CVPR range plus CLIC 0.075; higher rates go to the residual branch)
DEFAULT_LAMBDAS = [0.015, 0.0256, 0.0438, 0.0748, 0.128, 0.219, 0.374, 0.6]


# ---------------------------------------------------------------- data
class LatentCrops:
    """Random crops of cached full-image latents (memory-mapped)."""

    def __init__(self, root: Path, crop: int, seed: int = 0, holdout: int = 16):
        files = sorted(Path(root).glob("*.npy"))
        assert files, f"no .npy latents in {root}"
        self.arrs = [np.load(f, mmap_mode="r") for f in files]
        self.arrs = [a for a in self.arrs if min(a.shape[1:]) >= crop]
        self.val, self.arrs = self.arrs[:holdout], self.arrs[holdout:]
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def batch(self, n: int) -> torch.Tensor:
        out = np.empty((n, 32, self.crop, self.crop), np.float32)
        for i in range(n):
            a = self.arrs[self.rng.integers(len(self.arrs))]
            y = self.rng.integers(a.shape[1] - self.crop + 1)
            x = self.rng.integers(a.shape[2] - self.crop + 1)
            out[i] = a[:, y:y + self.crop, x:x + self.crop]
        return torch.from_numpy(out) * SCALING

    def val_items(self, n: int = 8):
        return [torch.from_numpy(np.asarray(a, np.float32))[None] * SCALING for a in self.val[:n]]


# ---------------------------------------------------------------- selftest
def selftest(args):
    """No training data needed: checks the codec end to end on random latents."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = LatentCodec().to(dev)
    m.prepare_coding()
    lat = torch.randn(1, 32, 43, 64, device=dev)  # a 1360x2048 image
    res = {}
    for r in (0, m.n_rates - 1):
        blob = m.compress(lat, r, seed=123)
        m_cpu = LatentCodec()
        m_cpu.load_state_dict(m.state_dict())  # decoder = fresh object loaded from the checkpoint, on CPU
        out_cpu = m_cpu.decompress(blob)
        out_dev = m.decompress(blob, device=dev).cpu()
        with torch.no_grad():
            est = m(lat, torch.tensor([r], device=dev))
            # what the encoder expects the decoder to reconstruct: q - u with the shared dither, through g_s
            from ratflow.entropy import dither
            x, (h, w) = m._pad(lat)
            gain = m.gain(torch.tensor([r], device=dev))
            y_r = (m.g_a(x) * gain).double().cpu().numpy()
            u = dither.u_float(dither.dither_u16(y_r.size, 123)).reshape(y_r.shape)
            y_hat = torch.from_numpy(np.round(y_r + u) - u).float().to(dev)
            expect = m.g_s(y_hat / gain)[..., :h, :w].cpu()
        bits_est = float(est["bits_y"] + est["bits_z"])
        res[r] = dict(bytes=len(blob), est_bytes=bits_est / 8, shape_ok=tuple(out_cpu.shape) == tuple(lat.shape),
                      decode_equals_encoder_view=bool(torch.equal(out_dev, expect)),
                      cpu_vs_dev_max_abs=float((out_cpu - out_dev).abs().max()))
        print(r, res[r])
    ok = all(v["shape_ok"] and v["decode_equals_encoder_view"] and v["cpu_vs_dev_max_abs"] < 1e-3
             for v in res.values())
    print("SELFTEST", "PASS" if ok else "FAIL")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "selftest.json").write_text(json.dumps(dict(ok=ok, **{str(k): v for k, v in res.items()}), indent=1))


# ---------------------------------------------------------------- train
def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lambdas = torch.tensor(args.lambdas, device=dev)
    data = LatentCrops(Path(args.latents), args.crop, seed=args.seed)
    m = LatentCodec(n_rates=len(args.lambdas), N=args.N, M=args.M, Z=args.Z, y_stride=args.y_stride).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.0)
    step, log = 0, []
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step, log = s["step"], s["log"]
        print("resumed at step", step)
    px = (args.crop * 32) ** 2
    t_end = time.time() + args.hours * 3600
    acc = {k: torch.zeros(len(args.lambdas), device=dev) for k in ("bpp", "mse", "n")}
    while step < args.steps and time.time() < t_end:
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        lat = data.batch(args.batch).to(dev, non_blocking=True)
        r = torch.randint(0, len(args.lambdas), (args.batch,), device=dev)
        o = m(lat, r)
        bpp = (o["bits_y"] + o["bits_z"]) / px
        mse = (o["lat_hat"] - lat).pow(2).flatten(1).mean(1)
        loss = (bpp + lambdas[r] * mse).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            acc["bpp"].index_add_(0, r, bpp.detach())
            acc["mse"].index_add_(0, r, mse.detach())
            acc["n"].index_add_(0, r, torch.ones_like(bpp))
        if step % args.log_every == 0:
            n = acc["n"].clamp_min(1)
            row = dict(step=step, lr=lr, bpp=(acc["bpp"] / n).tolist(), mse=(acc["mse"] / n).tolist())
            log.append(row)
            print(f"step {step}  lr {lr:.2e}  bpp " + " ".join(f"{v:.4f}" for v in row["bpp"])
                  + "  mse " + " ".join(f"{v:.3f}" for v in row["mse"]), flush=True)
            acc = {k: torch.zeros_like(v) for k, v in acc.items()}
        if step % args.save_every == 0:
            save(m, opt, step, log, args, out)
    save(m, opt, step, log, args, out)
    print("done at step", step)


def save(m, opt, step, log, args, out: Path):
    m.prepare_coding()
    s = dict(model=m.state_dict(), opt=opt.state_dict(), step=step, log=log, config=m.config,
             lambdas=list(args.lambdas), scaling=SCALING)
    torch.save(s, out / "last.pt.tmp")
    (out / "last.pt.tmp").replace(out / "last.pt")
    (out / "log.json").write_text(json.dumps(dict(config=m.config, lambdas=list(args.lambdas), log=log), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["selftest", "train"])
    ap.add_argument("--latents", default="latents")
    ap.add_argument("--out", default="s1_out")
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--N", type=int, default=192)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--Z", type=int, default=96)
    ap.add_argument("--y-stride", type=int, default=2)
    ap.add_argument("--crop", type=int, default=16, help="latent crop (16 = 512x512 pixels)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    globals()[args.step](args)


if __name__ == "__main__":
    main()
