# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "torch"]
# ///
"""Go/no-go 1 (criterion of 27/9): does knowing the EXACT quantization-noise model help a decoder?

On real DC-AE latents (cached by `measure_quant_noise.py encode`), latents are quantized by the
KLT-2x2 proxy transform coder with a subtractive dither, so the channel is  y_hat = y + e,
e ~ Unif(-Delta/2, Delta/2) iid in the KLT coefficient domain, mapped back to the latent domain.
The same small residual CNN (conditioned on log Delta) is trained three times:

    exact      trained on the true channel (dithered uniform noise)            -> test on true channel
    gaussian   trained on the OSCAR-style assumption y + N(0, Delta^2/12)       -> test on true channel
    nodither   reference: deterministic rounding (what StableCodec/AEIC do),
               trained and tested on its own channel (different rate, see below)

Decision (plan, go/no-go 1 (a)): exact beats gaussian by >= 5% latent MSE at >= 2 of the 4 rates.

Rates: Delta is chosen per target bpp from the zeroth-order entropy of the undithered indices on the
TRAIN latents (as in measure_quant_noise.py). The dithered arms use the same Delta; their true rate
H(Q|U) is not estimated here (the reported dithered H(Q) is an upper bound). The exact-vs-gaussian
comparison is at identical bitstreams, hence identical rate.

Usage (train on e.g. DIV2K-train latents, test on Kodak and CLIC latents):
    python measure_quant_noise.py encode --images /data/DIV2K_train_HR --out runs/div2k --device cuda
    python measure_quant_noise.py encode --images /data/kodak --out runs/kodak --device cuda
    python denoiser_gonogo.py --train runs/div2k --test runs/kodak runs/clic --out runs/gonogo1 --device cuda
    python denoiser_gonogo.py --synthetic --out runs/gonogo1_smoke --device cpu --steps 200   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from measure_quant_noise import TransformCoder, bpp_for, entropy_bits, find_delta, synthetic_latents

ARMS = ("exact", "gaussian", "nodither")


# ----------------------------------------------------------------------------- data
def load_latents(run_dir: Path):
    files = sorted((run_dir / "latents").glob("*.npy"))
    assert files, f"no latents in {run_dir / 'latents'}; run `measure_quant_noise.py encode` first"
    cfg = json.loads((run_dir / "encode_config.json").read_text())
    return [np.load(f) for f in files], cfg["downsample"]


class TorchKLT:
    """The numpy TransformCoder (f=2, PCA) as torch ops on (B, C, H, W) latents."""

    def __init__(self, tc: TransformCoder, device):
        assert tc.f == 2
        self.A = torch.tensor(tc.A, dtype=torch.float32, device=device)
        self.A_inv = torch.tensor(tc.A_inv, dtype=torch.float32, device=device)
        self.mu = torch.tensor(tc.mu, dtype=torch.float32, device=device)

    def forward(self, z):  # (B, C, H, W) -> (B, H/2, W/2, 4C); same ordering as measure_quant_noise.to_vectors
        v = F.pixel_unshuffle(z, 2).permute(0, 2, 3, 1)
        return (v - self.mu) @ self.A

    def inverse(self, y):
        v = y @ self.A_inv + self.mu
        return F.pixel_shuffle(v.permute(0, 3, 1, 2), 2)


def channel(z, klt: TorchKLT, delta, arm, gen):
    """Decoder input z_hat for a batch; delta is a (B,) tensor."""
    y = klt.forward(z)
    d = delta.view(-1, 1, 1, 1)
    if arm == "gaussian_train":  # the assumption, used only to TRAIN the gaussian arm
        yh = y + torch.randn(y.shape, generator=gen, device=y.device) * d / np.sqrt(12)
    elif arm in ("exact", "gaussian"):  # true dithered channel
        u = (torch.rand(y.shape, generator=gen, device=y.device) - 0.5) * d
        yh = d * torch.round((y + u) / d) - u
    elif arm == "nodither":
        yh = d * torch.round(y / d)
    else:
        raise ValueError(arm)
    return klt.inverse(yh)


# ----------------------------------------------------------------------------- model
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class Denoiser(nn.Module):
    """Residual CNN, input = z_hat plus a constant log(Delta) channel."""

    def __init__(self, c, width=128, blocks=8):
        super().__init__()
        self.inp = nn.Conv2d(c + 1, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.out = nn.Conv2d(width, c, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, zh, log_delta):
        cond = log_delta.view(-1, 1, 1, 1).expand(-1, 1, *zh.shape[2:])
        return zh + self.out(F.silu(self.blocks(self.inp(torch.cat([zh, cond], 1)))))


# ----------------------------------------------------------------------------- train / eval
def random_crops(latents, n, size, rng):
    out = []
    for i in rng.integers(len(latents), size=n):
        z = latents[i]
        h0 = 2 * rng.integers((z.shape[1] - size) // 2 + 1)
        w0 = 2 * rng.integers((z.shape[2] - size) // 2 + 1)
        z = z[:, h0:h0 + size, w0:w0 + size]
        if rng.random() < 0.5:
            z = z[:, :, ::-1]
        out.append(np.ascontiguousarray(z))
    return np.stack(out)


def train_arm(arm, latents, klt, deltas, args, device):
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    model = Denoiser(latents[0].shape[0], args.width, args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    log_d = np.log(np.array(deltas))
    train_channel = "gaussian_train" if arm == "gaussian" else arm
    t0 = time.time()
    for it in range(args.steps):
        z = torch.from_numpy(random_crops(latents, args.batch, args.crop, rng)).to(device)
        # log-uniform Delta over the tested range (slightly widened) so the model interpolates between rates
        ld = torch.tensor(rng.uniform(log_d.min() - 0.1, log_d.max() + 0.1, args.batch), dtype=torch.float32, device=device)
        with torch.no_grad():
            zh = channel(z, klt, ld.exp(), train_channel, gen)
        loss = F.mse_loss(model(zh, ld), z)
        model.last_losses = (getattr(model, "last_losses", []) + [loss.item()])[-500:]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, args.steps // 10) == 0:
            print(f"    [{arm}] step {it + 1}/{args.steps} loss {loss.item():.5f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def evaluate(models, latents, klt, deltas, args, device):
    """Per-element latent MSE on full test latents, averaged over `n_draws` dither draws."""
    res = {}
    for delta in deltas:
        row = {a: [] for a in ("raw_dither", "raw_nodither", *ARMS)}
        for draw in range(args.n_draws):
            gen = torch.Generator(device=device).manual_seed(10_000 + draw)
            for z_np in latents:
                z = torch.from_numpy(z_np[None]).to(device)
                d = torch.full((1,), delta, device=device)
                ld = d.log()
                zh_d = channel(z, klt, d, "exact", gen)
                zh_n = channel(z, klt, d, "nodither", gen)
                row["raw_dither"].append(F.mse_loss(zh_d, z).item())
                row["raw_nodither"].append(F.mse_loss(zh_n, z).item())
                row["exact"].append(F.mse_loss(models["exact"](zh_d, ld), z).item())
                row["gaussian"].append(F.mse_loss(models["gaussian"](zh_d, ld), z).item())
                row["nodither"].append(F.mse_loss(models["nodither"](zh_n, ld), z).item())
        res[delta] = {k: float(np.mean(v)) for k, v in row.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path)
    ap.add_argument("--test", type=Path, nargs="*", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bpp", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crop", type=int, default=16, help="latent crop size (even)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_draws", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=8, help="training images held out as an in-distribution test set")
    ap.add_argument("--min-train", type=int, default=200, help="refuse to run with fewer training images (memorisation)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.synthetic:
        train_lat = synthetic_latents(n=24, c=32, h=32, w=32, seed=0)
        tests = {"synthetic_test": (synthetic_latents(n=6, c=32, h=32, w=48, seed=1), 32)}
        down = 32
    else:
        train_lat, down = load_latents(args.train)
        tests = {p.name: load_latents(p) for p in args.test}
    crop_ok = [z for z in train_lat if z.shape[1] >= args.crop and z.shape[2] >= args.crop]
    assert crop_ok, "no training latent is large enough for --crop"
    train_lat = crop_ok
    if not args.synthetic:
        assert len(train_lat) - args.holdout >= args.min_train, (
            f"only {len(train_lat)} training latents: a 2.4M-parameter denoiser memorises them and does worse than "
            f"identity on new images; need >= {args.min_train + args.holdout} (or lower --min-train for a smoke run)")
    if args.holdout:
        tests = {"train_holdout": (train_lat[-args.holdout:], down), **tests}
        train_lat = train_lat[:-args.holdout]

    # transform coder + step sizes, fitted on TRAIN latents only
    tc = TransformCoder(train_lat, 2, "pca")
    ys = [tc.forward(z) for z in train_lat]
    pixels = sum(z.shape[1] * z.shape[2] for z in train_lat) * down**2
    deltas, rate_info = [], []
    for b in args.bpp:
        dlt = find_delta(ys, b, pixels)
        rng = np.random.default_rng(0)
        qd = np.concatenate([np.round((y + rng.uniform(-dlt / 2, dlt / 2, y.shape)) / dlt) for y in ys])
        deltas.append(dlt)
        rate_info.append({"target_bpp": b, "delta": dlt, "bpp_nodither": bpp_for(ys, dlt, pixels),
                          "bpp_dither_marginal_upper_bound": entropy_bits(qd) / pixels})
        print(f"target {b:.3f} bpp -> Δ={dlt:.3f}  (dithered H(Q) upper bound {rate_info[-1]['bpp_dither_marginal_upper_bound']:.4f} bpp)")
    klt = TorchKLT(tc, device)

    models = {}
    for arm in ARMS:
        print(f"training {arm} ...", flush=True)
        models[arm] = train_arm(arm, train_lat, klt, deltas, args, device)

    results = {"args": {k: str(v) for k, v in vars(args).items()}, "rates": rate_info, "tests": {}}
    lines = ["# Go/no-go 1 (a): exact noise model vs Gaussian assumption", "",
             f"Train latents: {len(train_lat)}; threshold: exact must beat gaussian by ≥ {args.threshold:.0%} "
             "latent MSE at ≥ 2 of the rates (per test set).", ""]
    lines += ["Final train loss (mean of last 500 steps): " + ", ".join(
        f"{a} {np.mean(getattr(m, 'last_losses', [float('nan')])):.4f}" for a, m in models.items()), ""]
    for name, (lat, _) in tests.items():
        invalid = 0
        lat = [z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2] for z in lat]
        ev = evaluate(models, lat, klt, deltas, args, device)
        wins = 0
        lines += [f"## {name} ({len(lat)} latents)", "",
                  "| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |",
                  "|---|---|---|---|---|---|---|---|"]
        for info, delta in zip(rate_info, deltas):
            e = ev[delta]
            gain = 1 - e["exact"] / e["gaussian"]
            valid = e["exact"] < e["raw_dither"]
            invalid += not valid
            wins += (gain >= args.threshold) and valid
            lines.append(f"| {info['target_bpp']:.3f} | {delta:.3f} | {e['raw_dither']:.4f} | {e['exact']:.4f} | "
                         f"{e['gaussian']:.4f} | {gain:+.1%} | {e['raw_nodither']:.4f} | {e['nodither']:.4f} |")
        passed = wins >= 2
        verdict = "INVALID (denoiser worse than its input)" if invalid >= 2 else ("PASS" if passed else "FAIL")
        lines += ["", f"**{name}: {wins}/{len(deltas)} valid rates reach the threshold -> {verdict}**", ""]
        results["tests"][name] = {"per_delta": {str(k): v for k, v in ev.items()}, "wins": int(wins), "pass": bool(passed),
                                  "invalid_rates": int(invalid), "verdict": verdict}
    lines += ["Note: the no-dither arm uses the same Δ, so its true rate is lower than the dithered arms'; "
              "compare it only together with the rate table in results.json."]
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
