# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "torch", "diffusers", "transformers", "accelerate", "pillow", "matplotlib"]
# ///
"""Go/no-go 1: is quantization error on generative-model latents far from Gaussian?

Stage 1 (GPU/TPU/CPU): encode images with the backbone's autoencoder and cache normalized latents
(-> <out>/latents/*.npy). Default: SANA's DC-AE f32c32 (AutoencoderDC, z = latent * scaling_factor,
32x downsampling). A KL VAE (e.g. SD3.5, z = (mean - shift_factor) * scaling_factor, 8x) also works.
Stage 2 (CPU): quantize with a simple transform coder and measure the error, with and without a
    subtractive dither, at target rates chosen by bisection on the step size.

Transform coders (no learned codec exists yet, so these are proxies for the S1 analysis transform):
    direct  per-pixel 16-dim vectors, per-channel standardization         (f = 1, basis = std)
    klt2    2x2 space-to-depth -> 64-dim vectors, PCA/KLT basis             (f = 2, basis = pca)
Rate = empirical zeroth-order entropy of the (undithered) quantization indices per coefficient,
converted to bits per image pixel (downsampling factor read from encode_config.json). This ignores context models / hyperpriors,
so it OVERestimates the rate of a learned codec; the same step is used for the dithered variant.

Error statistics, in the coefficient domain (what is quantized) and in the latent domain
(e_z = inverse(y_hat) - z, what a latent bridge actually has to remove):
    excess kurtosis / skewness / KS distance to a fitted Gaussian (per coefficient or channel),
    corr(e, y) and corr(e^2, y^2)          signal dependence and heteroscedasticity,
    zero fraction                          dead zone: e = -y wherever the index is 0,
    mean |off-diagonal| channel corr       cross-channel structure (latent domain),
    lag-1 spatial autocorrelation          spatial structure (latent domain),
    Spearman(patch error energy, patch latent variance)   content dependence (latent domain).
A Gaussian reference (iid N(0, Delta^2/12) in the coefficient domain, the "diffusion noise"
assumption) is pushed through the same pipeline to give the null values of every statistic.

Usage
    uv run measure_quant_noise.py encode  --images /data/kodak --out runs/kodak --device cuda
    uv run measure_quant_noise.py analyze --out runs/kodak
    uv run measure_quant_noise.py analyze --out runs/synthetic --synthetic      # pipeline smoke test
The default SANA repo is not gated. For a gated KL VAE (SD3.5) accept the license and log in first,
e.g. --vae stabilityai/stable-diffusion-3.5-medium. --device accepts cuda, cpu or xla (torch_xla).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_DOWNSAMPLE = 8  # only used for --synthetic


# ============================================================================= stage 1: encode
def get_device(name: str):
    import torch
    if name == "xla":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device(name)


def encode(args):
    import torch
    from diffusers import AutoencoderDC, AutoencoderKL
    from PIL import Image

    device = get_device(args.device)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    cfg = AutoencoderKL.load_config(args.vae, subfolder=args.vae_subfolder)
    is_dc = cfg.get("_class_name") == "AutoencoderDC"
    cls = AutoencoderDC if is_dc else AutoencoderKL
    vae = cls.from_pretrained(args.vae, subfolder=args.vae_subfolder, torch_dtype=dtype).to(device).eval()
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    scale = vae.config.scaling_factor
    down = getattr(vae, "spatial_compression_ratio", None) or 2 ** (len(vae.config.block_out_channels) - 1)
    out = Path(args.out) / "latents"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    print(f"{len(paths)} images, {cls.__name__} down={down} shift={shift} scale={scale}, device={device}")
    m = 2 * down  # crop so that klt2 (2x2 on the latent) tiles exactly
    for p in paths:
        img = Image.open(p).convert("RGB")
        if args.max_side and max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side), Image.BICUBIC)
        w, h = (img.size[0] // m) * m, (img.size[1] // m) * m
        img = img.crop(((img.size[0] - w) // 2, (img.size[1] - h) // 2, (img.size[0] - w) // 2 + w, (img.size[1] - h) // 2 + h))
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.no_grad():
            enc = vae.encode(x.to(device, dtype))
            mean = enc.latent if is_dc else enc.latent_dist.mean
        z = ((mean.float() - shift) * scale)[0].cpu().numpy()
        np.save(out / f"{p.stem}.npy", z.astype(np.float32))
        print(f"  {p.name}: {tuple(img.size)} -> latent {z.shape}")
    (Path(args.out) / "encode_config.json").write_text(json.dumps(
        {"vae": args.vae, "subfolder": args.vae_subfolder, "class": cls.__name__, "downsample": down,
         "shift_factor": shift, "scaling_factor": scale,
         "n_images": len(paths), "max_side": args.max_side}, indent=2))


# ============================================================================= stage 2: analyze
def synthetic_latents(n=8, c=16, h=64, w=96, seed=0):
    """Smooth, channel-correlated, heavy-ish tailed fields: only to smoke-test the pipeline."""
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((c, c)) / np.sqrt(c) + np.eye(c)
    ky, kx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    filt = 1.0 / (1e-3 + (ky**2 + kx**2)) ** 0.6
    out = []
    for _ in range(n):
        f = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((c, h, w))) * filt))
        f = np.einsum("dc,chw->dhw", mix, f / f.std())
        out.append((np.sign(f) * np.abs(f) ** 1.2).astype(np.float32))
    return out


def to_vectors(z, f):
    c, h, w = z.shape
    return z.reshape(c, h // f, f, w // f, f).transpose(1, 3, 0, 2, 4).reshape(-1, c * f * f)


def from_vectors(v, shape, f):
    c, h, w = shape
    return v.reshape(h // f, w // f, c, f, f).transpose(2, 0, 3, 1, 4).reshape(c, h, w)


class TransformCoder:
    def __init__(self, latents, f, basis):
        self.f = f
        x = np.concatenate([to_vectors(z, f) for z in latents]).astype(np.float64)
        self.mu = x.mean(0)
        if basis == "pca":
            evals, evecs = np.linalg.eigh(np.cov(x - self.mu, rowvar=False))
            order = np.argsort(evals)[::-1]
            self.A, self.A_inv = evecs[:, order], evecs[:, order].T
            self.var = evals[order]
        else:
            s = x.std(0)
            self.A, self.A_inv = np.diag(1 / s), np.diag(s)
            self.var = np.ones_like(s)

    def forward(self, z):
        return (to_vectors(z, self.f) - self.mu) @ self.A

    def inverse(self, y, shape):
        return from_vectors(y @ self.A_inv + self.mu, shape, self.f)


def quantize(y, delta, mode, rng):
    if mode == "no_dither":
        return delta * np.round(y / delta), np.round(y / delta)
    if mode == "dither":
        u = rng.uniform(-delta / 2, delta / 2, y.shape)
        return delta * np.round((y + u) / delta) - u, None
    if mode == "gauss_ref":
        return y + rng.normal(0, delta / np.sqrt(12), y.shape), None
    raise ValueError(mode)


def entropy_bits(q):
    bits = 0.0
    for j in range(q.shape[1]):
        _, cnt = np.unique(q[:, j], return_counts=True)
        p = cnt / cnt.sum()
        bits += -(cnt * np.log2(p)).sum()
    return bits


def bpp_for(ys, delta, pixels):
    q = np.concatenate([np.round(y / delta) for y in ys])
    return entropy_bits(q) / pixels


def find_delta(ys, target_bpp, pixels):
    lo, hi = np.log(1e-3), np.log(1e3)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bpp_for(ys, np.exp(mid), pixels) > target_bpp:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def _safe_corr(a, b):
    sa, sb = a.std(), b.std()
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.0


def column_stats(e, y, rng, max_cols=64, ks_n=20000):
    """Per-column statistics of the error e given the signal y; summarized over columns."""
    cols = [j for j in range(min(e.shape[1], max_cols)) if e[:, j].std() > 0]
    rows = []
    for j in cols:
        ej, yj = e[:, j], y[:, j]
        sub = ej if len(ej) <= ks_n else rng.choice(ej, ks_n, replace=False)
        rows.append({"kurt": float(stats.kurtosis(ej)), "skew": float(stats.skew(ej)),
                     "ks": float(stats.kstest((sub - sub.mean()) / sub.std(), "norm").statistic),
                     "corr_ey": _safe_corr(ej, yj), "corr_e2y2": _safe_corr(ej**2, yj**2)})
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    return out


def latent_stats(ez, z, patch=8):
    """ez, z: lists of (C, H, W) maps."""
    E = np.concatenate([m.reshape(m.shape[0], -1) for m in ez], 1)
    c = np.corrcoef(E)
    off = np.abs(c[~np.eye(len(c), dtype=bool)])
    lag = []
    for m in ez:
        for ch in m:
            lag.append(_safe_corr(ch[:, 1:].ravel(), ch[:, :-1].ravel()))
            lag.append(_safe_corr(ch[1:, :].ravel(), ch[:-1, :].ravel()))
    pe, pv = [], []
    for m, zz in zip(ez, z):
        C, H, W = m.shape
        for i in range(0, H - patch + 1, patch):
            for j in range(0, W - patch + 1, patch):
                pe.append((m[:, i:i + patch, j:j + patch] ** 2).mean())
                pv.append(zz[:, i:i + patch, j:j + patch].var(axis=(1, 2)).mean())
    return {"offdiag_abs_corr_mean": float(off.mean()), "offdiag_abs_corr_max": float(off.max()),
            "spatial_lag1_corr_mean": float(np.mean(lag)),
            "content_spearman_patch": float(stats.spearmanr(pe, pv).statistic)}


def analyze(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    down = DEFAULT_DOWNSAMPLE
    if (out / "encode_config.json").exists():
        down = json.loads((out / "encode_config.json").read_text())["downsample"]
    if args.synthetic:
        latents = synthetic_latents()
    else:
        files = sorted((out / "latents").glob("*.npy"))
        latents = [np.load(f) for f in files]
        assert latents, f"no latents in {out / 'latents'}; run `encode` first"
    pixels = sum(z.shape[1] * z.shape[2] for z in latents) * down**2
    print(f"{len(latents)} latents, {pixels / 1e6:.2f} Mpixel, latent shape e.g. {latents[0].shape}")
    rng = np.random.default_rng(args.seed)
    results = {"n_latents": len(latents), "pixels": pixels, "targets_bpp": args.bpp, "coders": {}}
    fig_data = {}
    for name, f, basis in [("direct", 1, "std"), ("klt2", 2, "pca")]:
        tc = TransformCoder(latents, f, basis)
        ys = [tc.forward(z) for z in latents]
        results["coders"][name] = {"coef_variance_top8": tc.var[:8].tolist(), "rates": []}
        for target in args.bpp:
            delta = find_delta(ys, target, pixels)
            entry = {"target_bpp": target, "delta": delta, "bpp": bpp_for(ys, delta, pixels)}
            for mode in ("no_dither", "dither", "gauss_ref"):
                yh = [quantize(y, delta, mode, rng)[0] for y in ys]
                e = np.concatenate([a - b for a, b in zip(yh, ys)])
                y_all = np.concatenate(ys)
                ez = [tc.inverse(a, z.shape) - z for a, z in zip(yh, latents)]
                st = {"coef": column_stats(e, y_all, rng), "latent": latent_stats(ez, latents),
                      "latent_channel": column_stats(np.concatenate([m.reshape(m.shape[0], -1).T for m in ez]),
                                                     np.concatenate([z.reshape(z.shape[0], -1).T for z in latents]), rng),
                      "mse_latent": float(np.mean([((m) ** 2).mean() for m in ez]))}
                if mode == "no_dither":
                    st["zero_fraction"] = float(np.mean(np.concatenate([np.round(y / delta) for y in ys]) == 0))
                entry[mode] = st
                if name == "klt2" and target == args.bpp[len(args.bpp) // 2]:
                    fig_data[mode] = (e[:, min(4, e.shape[1] - 1)], ez[0])
            results["coders"][name]["rates"].append(entry)
            nd, di = entry["no_dither"], entry["dither"]
            print(f"[{name}] target {target:.3f} bpp -> Δ={delta:.3f} ({entry['bpp']:.4f} bpp), zero frac {nd['zero_fraction']:.2f} | "
                  f"coef kurt nd {nd['coef']['kurt']['median']:+.2f} / d {di['coef']['kurt']['median']:+.2f} | "
                  f"corr(e,y) nd {nd['coef']['corr_ey']['median']:+.2f} / d {di['coef']['corr_ey']['median']:+.2f} | "
                  f"latent kurt nd {nd['latent_channel']['kurt']['median']:+.2f} / d {di['latent_channel']['kurt']['median']:+.2f} | "
                  f"offdiag nd {nd['latent']['offdiag_abs_corr_mean']:.2f} / d {di['latent']['offdiag_abs_corr_mean']:.2f} | "
                  f"content ρ nd {nd['latent']['content_spearman_patch']:+.2f} / d {di['latent']['content_spearman_patch']:+.2f}", flush=True)
    results["verdict"] = verdict(results, args)
    (out / "quant_noise_stats.json").write_text(json.dumps(results, indent=2))
    (out / "summary.md").write_text(summary_md(results), encoding="utf-8")
    plot(fig_data, out)
    print(summary_md(results))


# Proposed go/no-go thresholds (see plan): no-dither error at <= 0.05 bpp is "clearly non-Gaussian" if
# ANY of these holds in the latent domain or the coefficient domain.
THRESHOLDS = {"abs_kurt": 0.5, "abs_corr_ey": 0.3, "offdiag": 0.1, "content_rho": 0.3}


def verdict(results, args):
    out = {}
    for name, r in results["coders"].items():
        rows = []
        for e in r["rates"]:
            nd = e["no_dither"]
            flags = {
                "coef_kurt": abs(nd["coef"]["kurt"]["median"]) > THRESHOLDS["abs_kurt"],
                "latent_kurt": abs(nd["latent_channel"]["kurt"]["median"]) > THRESHOLDS["abs_kurt"],
                "coef_corr_ey": abs(nd["coef"]["corr_ey"]["median"]) > THRESHOLDS["abs_corr_ey"],
                "latent_corr_ey": abs(nd["latent_channel"]["corr_ey"]["median"]) > THRESHOLDS["abs_corr_ey"],
                "latent_offdiag": nd["latent"]["offdiag_abs_corr_mean"] > THRESHOLDS["offdiag"] + e["gauss_ref"]["latent"]["offdiag_abs_corr_mean"],
                "latent_content": abs(nd["latent"]["content_spearman_patch"]) > THRESHOLDS["content_rho"],
            }
            rows.append({"target_bpp": e["target_bpp"], "flags": flags, "non_gaussian": any(flags.values())})
        out[name] = rows
    return {"thresholds": THRESHOLDS, "per_coder": out,
            "all_low_rate_non_gaussian": all(r["non_gaussian"] for rows in out.values() for r in rows)}


def summary_md(results):
    L = [f"# Quantization error vs. Gaussian — {results['n_latents']} latents", "",
         "Medians over coefficients / latent channels. nd = no dither, d = subtractive dither, g = Gaussian reference.", ""]
    for name, r in results["coders"].items():
        L += [f"## {name}", "", "| target bpp | Δ | zero frac (nd) | kurt coef nd / d / g | kurt latent nd / d / g | corr(e,y) coef nd / d | "
              "corr(e,y) latent nd / d | offdiag latent nd / d / g | lag-1 latent nd / d / g | content ρ nd / d / g | MSE latent nd / d |",
              "|" + "---|" * 11]
        for e in r["rates"]:
            nd, d, g = e["no_dither"], e["dither"], e["gauss_ref"]
            f3 = lambda k1, k2: " / ".join(f"{x[k1][k2]['median']:+.2f}" for x in (nd, d, g))
            fl = lambda k: " / ".join(f"{x['latent'][k]:+.2f}" for x in (nd, d, g))
            L.append(f"| {e['target_bpp']:.3f} | {e['delta']:.2f} | {nd['zero_fraction']:.2f} | {f3('coef', 'kurt')} | "
                     f"{f3('latent_channel', 'kurt')} | {nd['coef']['corr_ey']['median']:+.2f} / {d['coef']['corr_ey']['median']:+.2f} | "
                     f"{nd['latent_channel']['corr_ey']['median']:+.2f} / {d['latent_channel']['corr_ey']['median']:+.2f} | "
                     f"{fl('offdiag_abs_corr_mean')} | {fl('spatial_lag1_corr_mean')} | {fl('content_spearman_patch')} | "
                     f"{nd['mse_latent']:.3f} / {d['mse_latent']:.3f} |")
        L.append("")
    v = results["verdict"]
    L += ["## Go/no-go 1 (proposed thresholds)", "", f"Thresholds: {v['thresholds']}", ""]
    for name, rows in v["per_coder"].items():
        for r in rows:
            hit = [k for k, b in r["flags"].items() if b]
            L.append(f"- {name} @ {r['target_bpp']:.3f} bpp: {'NON-GAUSSIAN' if r['non_gaussian'] else 'gaussian-like'} ({', '.join(hit) or 'no flag'})")
    L += ["", f"**All tested low rates non-Gaussian (no dither): {v['all_low_rate_non_gaussian']}**"]
    return "\n".join(L)


def plot(fig_data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not fig_data:
        return
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    for k, mode in enumerate(("no_dither", "dither", "gauss_ref")):
        e, ez = fig_data[mode]
        ax = axs[0, k]
        ax.hist(e, bins=120, density=True, alpha=0.7)
        xs = np.linspace(e.min(), e.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), "k-", lw=1)
        ax.set_title(f"{mode}: coef #5 error (klt2)", fontsize=9)
        ax.set_yscale("log")
        axs[1, k].imshow(np.sqrt((ez**2).mean(0)), cmap="magma")
        axs[1, k].set_title(f"{mode}: |e_z| map, image 0", fontsize=9)
        axs[1, k].axis("off")
    fig.tight_layout()
    fig.savefig(out / "quant_noise.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--images", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--vae", default="Efficient-Large-Model/Sana_1600M_1024px_diffusers")
    e.add_argument("--vae_subfolder", default="vae")
    e.add_argument("--device", default="cuda")
    e.add_argument("--max_side", type=int, default=1024, help="downscale larger images (0 = keep)")
    a = sub.add_parser("analyze")
    a.add_argument("--out", required=True)
    a.add_argument("--bpp", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    a.add_argument("--synthetic", action="store_true")
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    encode(args) if args.cmd == "encode" else analyze(args)


if __name__ == "__main__":
    main()
