"""Configuration: one JSON file drives model shape, training stages, allocation and packaging.

Every key has a default here; a config file only lists what it changes (nested dicts are merged).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "model": {
        "N": 192,                  # analysis width
        "M": 320,                  # latent channels
        "Z": 192,                  # hyper-latent channels
        "slices": [16, 16, 32, 64, 192],  # uneven channel groups (ELIC); must sum to M
        "hyper_hidden": 192,
        "ctx_hidden": 160,         # integer entropy-parameter nets
        "gs_width": 192,           # synthesis width (bigger = better texture, slower decode)
        "gs_blocks": [2, 2, 2, 2], # residual blocks per synthesis stage (coarse -> fine)
        "ga_blocks": 2,
        "n_levels": 8,             # rate levels sharing one model (gain units)
        "qmap_block": 128,         # pixels per quality-map cell (multiple of 16)
        "qmap_range": 4,           # cell value k in [-R, R]: latent gain x 2**(k/4)
        "beta_levels": 16,         # realism beta transmitted as an integer in [0, beta_levels - 1]
        # ---- GC v2 options (plans/GC_v2_architecture_2026-11-23.md); defaults = GC v1
        "e1": False,               # E1: 5x5 integer context on the decoded anchors of each slice
        "e2": False,               # E2: deeper parameter nets (256 -> 192 -> 128)
        "split": None,             # None | "q": sigma (integer) on residual symbols, mu (float) on y_hat | "hyper"
        "cca": 0.0,                # CCA loss weight (auxiliary nets, training only); 0 = off
        "gs_version": 1,           # 2: SynthesisV2 (capacity at 1/16 and 1/8, window attention)
        "gs2_blocks": [6, 4, 2, 2],
        "gs2_attn": 2,
        "gs2_window": 8,
        "gs2_up": "shuffle",       # shuffle | haar
    },
    "data": {
        "train_dirs": [],          # folders of training images (recursive)
        "screen_dirs": [],         # screen / game content, sampled with screen_prob
        "screen_prob": 0.1,
        "valid_dir": "",           # CLIC validation images (30 PNG)
        "crop": 256,
        "min_side": 256,
        "downscale": [0.5, 1.0],   # random downscale of large photos before cropping (removes JPEG noise)
        "workers": 8,
    },
    "rate": {
        # target bpp per level for HiFiC-style rate control: lambda_rate = hi if bpp > target else lo
        "targets": [0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45],
        "lambda_hi": 8.0,
        "lambda_lo": 0.5,
        "lambdas": None,           # if set: fixed lambda_rate per level instead of the hi/lo target control
    },
    "loss": {
        "k_mse": 0.075 * 2 ** -5 * 255 ** 2,  # HiFiC k_M on [0,1] images
        "k_lpips": 1.0,
        "k_wd": 0.0,               # Wasserstein distortion (optional, VGG16 weights needed)
        "k_gan": 0.15,
        "k_reward": 0.0,           # Q-hat v2 distance (alignment stage)
        "beta_max": 1.0,           # realism beta in [0, beta_max]; GAN weight is multiplied by beta
    },
    "stages": {
        # name: steps, lr, batch, which losses; run in the order listed by pipeline --stages
        "rd": {"steps": 300000, "lr": 1e-4, "batch": 16, "gan": False, "lpips": True},
        "gan": {"steps": 600000, "lr": 1e-4, "lr_d": 1e-4, "batch": 16, "gan": True, "lpips": True},
        "rate": {"steps": 100000, "lr": 3e-5, "lr_d": 3e-5, "batch": 16, "gan": True, "lpips": True,
                  "levels": [2, 4, 6]},  # fine-tune with levels near the three CLIC rates only
        "align": {"steps": 50000, "lr": 1e-5, "lr_d": 1e-5, "batch": 8, "gan": True, "lpips": True,
                   "reward": True},
    },
    "train": {
        "amp": "bf16",             # bf16 | fp16 | none (integer nets always run in fp32 with STE)
        "log_every": 100,
        "ckpt_every": 5000,
        "ckpt_minutes": 15,        # also refresh last.pt this often (bounds the loss from a hard kill / preemption)
        "init_strict": True,       # --init / previous-stage weights must match the model exactly
        "eval_every": 20000,
        "eval_images": 6,          # validation images used for periodic eval (full size)
        "grad_clip": 1.0,
        "ema": 0.999,              # EMA decay of the codec weights (0 = off); model.pt holds the EMA weights
        "max_bad_steps": 50,       # abort after this many non-finite losses in a row
        "seed": 1,
        "cca_warmup": 5000,        # steps over which the CCA weight ramps from 0 (model.cca > 0 only)
        "compile": False,
    },
    "reward": {
        "sets_train": ["2021v", "2021t", "2022t"],
        "sets_test": ["2024t"],
        "n_per_set": 4000,
        "steps": 4000,
        "batch": 16,
        "lr": 1e-3,
        "crop": 256,
    },
    "alloc": {
        "budgets": [0.075, 0.15, 0.3],
        "levels": {"0.075": [0, 1, 2, 3], "0.15": [2, 3, 4, 5], "0.3": [4, 5, 6, 7]},
        "betas": [1.0],
        "qmap_iters": 6,           # greedy worst-crop quality-map iterations per (image, level)
        "crop": 512,
        "stride": 256,
        "objective": "mix",        # mean | max | mix  (mix = mean + kappa * max over crops)
        "kappa": 0.5,
        "zip_overhead": 120,       # bytes per file assumed for bs.zip framing
        "safety": 0.995,           # use this fraction of the byte budget
        "refine_steps": 0,         # encoder-side latent refinement steps (0 = off)
    },
    "runtime": {
        "entropy_device": "cuda",  # integer nets: exact on CPU and CUDA (float64 integers, cuDNN off)
        "synth_dtype": "fp16",
    },
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path: str | Path | None) -> dict:
    cfg = DEFAULTS if path is None else merge(DEFAULTS, json.loads(Path(path).read_text()))
    m = cfg["model"]
    assert sum(m["slices"]) == m["M"], "slices must sum to M"
    assert m["qmap_block"] % 64 == 0, "qmap_block must be a multiple of 64 (hyper grid)"
    assert len(cfg["rate"]["targets"]) == m["n_levels"]
    assert not cfg["rate"].get("lambdas") or len(cfg["rate"]["lambdas"]) == m["n_levels"]
    return copy.deepcopy(cfg)
