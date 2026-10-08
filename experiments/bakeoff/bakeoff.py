"""Base bake-off for CLIC 2027 (plan v2, weeks 2-3): every candidate on the 30 validation images, at the
three CLIC rates, under the corpus byte budget, scored with Q-hat.

Candidates (each gives several operating points per image; the allocator picks one per image):
  vtm420   C0  VTM 23.8 intra, 4:2:0 10-bit, QP grid                  (CLIC's VTM anchor)
  vtmscc   C0  VTM 23.8 intra, 4:4:4 + SCC tools (IBC, palette, ...), QP grid
  s1res    C1  DC-AE + S1 (real bitstream, run-1 checkpoint) + RGB residual coded by VTM 4:4:4
  msillm   C2  MS-ILLM (Muckley et al. 2023), public checkpoints 0.035 .. 0.45 bpp (CC-BY-NC weights)
  mbt      C3  mbt2018-mean from the CompressAI zoo, MSE, qualities 1..4 (fast learned-MSE anchor; Cheng2020
              is left out because its autoregressive context model decodes a 2K image in minutes)
  codlite  C5  CoD-Lite (2026), pixel-space one-step diffusion codec, public MIT checkpoints 0.0156 .. 0.5 bpp
  turbo    C4  probe: MS-ILLM q1..q3 reconstruction refined by SD-Turbo img2img (1 step, strength 0.15 / 0.3)
  coolchic C6  Cool-chic 5.x (Orange, BSD-3): overfitted per-image codec, ~2000 MAC/pixel decoder; --tune
              wasserstein = MSE + Wasserstein Distortion (Balle et al. CVPR 2025; Orange's CLIC 2025 entry).
              Encoding is a per-image optimisation (slow; CLIC does not time the encoder).
  mix          per-image choice over the union of all candidates (what mode switching could buy)

Steps (all resume: existing points/metrics are skipped):
  prepare   30 images (CLIC 2025 test = CLIC 2027 validation), budgets
  points    reconstructions + exact byte counts + decode timings     -> points/<cand>.csv, recon/<cand>/<key>/<img>.png
            + the metrics of each point as soon as it exists (recon/ is not packed, so they must not be split)
  metrics   PSNR, MS-SSIM, LPIPS, DISTS, Q-hat features per point     -> metrics.csv  (catch-up for older runs)
  allocate  per candidate and rate: one point per image, total bytes <= budget, maximise sum of Q-hat
            (Lagrangian sweep; falls back to -LPIPS without a Q-hat model)        -> allocation.csv
  summary   tables per rate (all / natural / screen) + crop sheets    -> summary.md, crops/*.jpg

    python bakeoff.py all --out work/bakeoff --vtm vtm --s1-ckpt s1_out/last.pt --qhat work/qhat/qhat_v0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "clic_b"))
sys.path.insert(0, str(ROOT / "experiments" / "qhat"))
from clic27.eval import metrics as M  # noqa: E402

DATA_URL = "https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip"
RATES = [0.075, 0.15, 0.3]
CANDS = ["vtm420", "vtmscc", "s1res", "msillm", "mbt", "codlite", "turbo", "coolchic"]
LABELS = {"ebfd571f": "screen", "bb7344a2": "screen", "86127fbd": "screen", "2a760bf1": "screen",
          "937476dd": "texture"}  # by eye in branch B; every other image is natural
VTM_QPS = [34, 37, 40, 43, 46, 50]  # 2026-10-06 run: vtm420 qp26/30 = 0.80/0.52 bpp corpus (useless), qp34 0.33, qp50 0.029
RES_QPS = [32, 37, 42, 47]
S1_LEVELS = [1, 2, 3, 4, 5]  # run-1 checkpoint on Kodak: 0.020, 0.034, 0.053, 0.071, 0.089 bpp
MSILLM = {1: "msillm_quality_1", 2: "msillm_quality_2", 3: "msillm_quality_3", 4: "msillm_quality_4",
          5: "msillm_quality_5"}
MBT_Q = [1, 2, 3, 4]
CODLITE = ["0_0156", "0_0312", "0_1250", "0_5000"]  # public checkpoints in the CLIC range (fixed-length VQ, bpp exact)
GENCODEC_COMMIT = "9b39a94d078fa2864e8246c2e4031cafaf756b84"
TURBO_BASE = [1, 2, 3]  # MS-ILLM qualities refined by SD-Turbo (0.043 / 0.080 / 0.150 bpp on these images)
TURBO_STRENGTH = [0.15, 0.3]
COOLCHIC_COMMIT = "a6fe38a414dd098b39c41636bd6e423626402f7e"  # Cool-chic 5.x, 2026-10
COOLCHIC_LAMBDAS = [0.001, 0.002, 0.004, 0.008, 0.016]  # first guess for 0.075-0.3 bpp on 2K images; calibrate
AE_REPO = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
HEADER_BYTES = 4  # per image: candidate/point id + flags; H and W are known to the decoder from the stream


def label(name):
    return LABELS.get(name[:8], "natural")


# ---------------------------------------------------------------- prepare
def prepare(a):
    if not M.images(a.data):
        a.data.mkdir(parents=True, exist_ok=True)
        z = a.data / "valid.zip"
        subprocess.run(["wget", "-q", "-O", str(z), DATA_URL], check=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(a.data)
        z.unlink()
    files = M.images(a.data)
    sizes = {p.stem: M.load(p).shape[:2] for p in files}
    total = sum(h * w for h, w in sizes.values())
    budgets = {str(r): math.floor(r * total / 8) for r in RATES}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "budgets.json").write_text(json.dumps(dict(n_images=len(files), total_pixels=total, budgets_bytes=budgets,
                                                        sizes=sizes), indent=1))
    print(len(files), "images,", total, "pixels, budgets", budgets)


def inputs(a):
    files = M.images(a.data)
    return {p.stem: p for p in files}


def selected(a):
    """Image names to produce points for: --only takes labels (screen, texture, natural) or name prefixes."""
    names = sorted(inputs(a))
    if not a.only:
        return names
    return [n for n in names if label(n) in a.only or any(n.startswith(p) for p in a.only)]


# ---------------------------------------------------------------- points: bookkeeping
class Points:
    """points/<cand>.csv: one row per (cand, key, name) with bytes and timings; recon PNGs under recon/<cand>/.
    One file per candidate, so runs that each produce different candidates (e.g. a CPU-only session for
    vtm420 and another for vtmscc) merge by copying their work dirs together. A legacy points.csv is read too.
    results.zip does not carry recon/, so a restored point without its recon and without metrics is forgotten
    (it is produced again); with a scorer, every new point gets its metrics right away."""

    def __init__(self, out: Path, scorer=None):
        self.out, self.scorer = out, scorer
        self.dir = out / "points"
        scored = metric_keys(out)
        self.rows, seen, lost = [], set(), {}
        for f in [out / "points.csv", *sorted(self.dir.glob("*.csv"))]:
            for r in (M.read_csv(f) if f.exists() else []):
                k = (r["cand"], r["key"], r["name"])
                if k in seen:
                    continue
                if k not in scored and not self.recon(*k).exists():
                    lost[r["cand"]] = lost.get(r["cand"], 0) + 1
                    continue
                seen.add(k)
                self.rows.append(r)
        self.have = seen
        for cand, n in lost.items():
            print(f"points: {n} {cand} points have neither recon nor metrics (lost with the previous session); "
                  "they will be produced again", flush=True)
            self._write(cand)

    def recon(self, cand, key, name):
        return self.out / "recon" / cand / key / f"{name}.png"

    def _write(self, cand):
        self.dir.mkdir(parents=True, exist_ok=True)
        M.write_csv([r for r in self.rows if r["cand"] == cand], self.dir / f"{cand}.csv")

    def add(self, row):
        self.rows.append(row)
        self.have.add((row["cand"], row["key"], row["name"]))
        self._write(row["cand"])
        if self.scorer:
            self.scorer(self, row)

    def todo(self, cand, keys, names):
        return [(k, n) for k in keys for n in names if (cand, k, n) not in self.have]


def metric_keys(out):
    path = out / "metrics.csv"
    return {(r["cand"], r["key"], r["name"]) for r in M.read_csv(path)} if path.exists() else set()


class Scorer:
    """Appends one metrics.csv row per point (PSNR, MS-SSIM, LPIPS, DISTS, Q-hat features). Called as each point is
    produced, so a session stopped by the time budget never leaves points whose recon is gone but metrics missing."""

    def __init__(self, a):
        self.a, self.path, self.feat = a, a.out / "metrics.csv", None
        self.rows = M.read_csv(self.path) if self.path.exists() else []
        self.have = {(r["cand"], r["key"], r["name"]) for r in self.rows}

    def __call__(self, P, r):
        k = (r["cand"], r["key"], r["name"])
        if k in self.have:
            return
        if self.feat is None:
            from qhat import Features
            self.feat = Features(M.device())
        x = M.load(inputs(self.a)[r["name"]])
        y = M.load(P.recon(*k))
        f = self.feat(x, y)
        f["mse"] = M.mse(x, y)
        self.rows.append(dict(cand=r["cand"], key=r["key"], name=r["name"], **f))
        self.have.add(k)
        M.write_csv(self.rows, self.path)


def to_u8(t):
    """1x3xHxW in [0, 1] -> HxWx3 uint8."""
    return t[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- VTM (C0) and the residual of C1
def _vtm_job(kind, name, qp, src, dst, vtm_dir, base_png=None):
    """Worker process. kind: vtm420 | vtmscc | res. Returns (bytes, scc flags used)."""
    import b6_vtm_residual as B6
    x = M.load(src)
    H, W = x.shape[:2]
    enc, cfgf = B6.vtm_bin(vtm_dir), str(Path(vtm_dir) / "cfg" / "encoder_intra_vtm.cfg")
    with tempfile.TemporaryDirectory() as td:
        yuv = f"{td}/in.yuv"
        if kind == "vtm420":
            fmt = "420"
            y, cb, cr = B6.rgb_to_yuv10(x)
            B6.write_yuv([y, B6.resize(cb, W // 2, H // 2), B6.resize(cr, W // 2, H // 2)], yuv)
        elif kind == "vtmscc":
            fmt = "444"
            B6.write_yuv(B6.rgb_to_yuv10(x), yuv)
        else:
            fmt = "444"
            xh = M.load(base_png).astype(np.float64)
            B6.write_yuv([x[..., i] - xh[..., i] + 512 for i in range(3)], yuv)
        err = ""
        for extra in (B6.SCC_FLAGS if kind == "vtmscc" else [[]]):
            try:
                size = B6.vtm_encode(enc, cfgf, yuv, W, H, fmt, qp, f"{td}/o.bin", f"{td}/o.yuv", extra)
                break
            except RuntimeError as e:
                err = str(e)
        else:
            raise RuntimeError(f"{kind} {name} qp{qp}: {err}")
        planes = B6.read_yuv(f"{td}/o.yuv", W, H, fmt)
        if kind == "vtm420":
            planes = [planes[0], B6.resize(planes[1], W, H), B6.resize(planes[2], W, H)]
        if kind == "res":
            rec = np.clip(np.round(xh + np.stack([p - 512 for p in planes], -1)), 0, 255).astype(np.uint8)
        else:
            rec = B6.yuv10_to_rgb(*planes)
    M.save(rec, dst)
    return size, " ".join(extra)


def _run_vtm_jobs(a, P, jobs):
    """jobs: list of (cand, key, name, kind, qp, base_png, base_bytes)."""
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_vtm_job, kind, name, qp, str(inputs(a)[name]), str(P.recon(cand, key, name)), a.vtm,
                          base_png): (cand, key, name, base_bytes) for cand, key, name, kind, qp, base_png, base_bytes in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            cand, key, name, base_bytes = futs[f]
            try:
                size, flags = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(jobs)}] FAILED {cand} {key} {name[:8]}: {e}", flush=True)
                continue
            P.add(dict(cand=cand, key=key, name=name, bytes=size + base_bytes + HEADER_BYTES, bytes_base=base_bytes,
                       bytes_res=size, t_dec=float("nan"), note=flags))
            print(f"[{i}/{len(jobs)}] {cand} {key} {name[:8]} {size + base_bytes} B", flush=True)


def points_vtm(a, P, names):
    jobs = []
    for cand in ("vtm420", "vtmscc"):
        if cand in a.cands:
            jobs += [(cand, f"qp{q}", n, cand, q, None, 0) for q, n in
                     ((int(k[2:]), n) for k, n in P.todo(cand, [f"qp{q}" for q in a.qps], names))]
    jobs.sort(key=lambda j: -j[4])  # high QPs first: they are fast, so a time-out leaves the most points
    if jobs:
        _run_vtm_jobs(a, P, jobs)


# ---------------------------------------------------------------- C1: DC-AE + S1 (+ VTM residual)
def points_s1res(a, P, names):
    import torch
    from clic27.codec.latent_codec import LatentCodec
    from clic27.nn.dcae import DCAE
    if not a.s1_ckpt or not Path(a.s1_ckpt).exists():
        print("s1res: no --s1-ckpt, skipping", flush=True)
        return
    dev = M.device()
    s = torch.load(a.s1_ckpt, map_location="cpu", weights_only=False)
    enc = LatentCodec(**s["config"])
    enc.load_state_dict(s["model"])
    enc = enc.to(dev).eval().prepare_coding()
    dec = LatentCodec(**s["config"])
    dec.load_state_dict(s["model"])
    dec = dec.to(dev).eval().prepare_coding()
    scaling = s.get("scaling", 0.41407)
    ae = DCAE.from_pretrained(AE_REPO, "vae").to(dev).enable_tiling()
    n_dec = sum(p.numel() for p in ae.decoder.parameters()) + sum(p.numel() for p in dec.parameters())
    base_keys = [f"s1L{lv}" for lv in S1_LEVELS]
    for key, name in P.todo("s1res", base_keys, names):
        lv = int(key[3:])
        x = M.load(inputs(a)[name])
        H, W = x.shape[:2]
        ph, pw = (-H) % 32, (-W) % 32
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            lat = ae.encode(t) * scaling
            blob = enc.compress(lat, lv, seed=lv)
            sync()
            t0 = time.time()
            lat_hat = dec.decompress(blob, device=dev)
            img = ae.decode(lat_hat / scaling)
            sync()
            t_dec = time.time() - t0
        rec = img[0, :, :H, :W].clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        M.save(rec, P.recon("s1res", key, name))
        P.add(dict(cand="s1res", key=key, name=name, bytes=len(blob) + HEADER_BYTES, bytes_base=len(blob), bytes_res=0,
                   t_dec=t_dec, note=f"decoder_params={n_dec}"))
        print(f"s1res {key} {name[:8]} {len(blob)} B  dec {t_dec:.2f}s", flush=True)
    del ae, enc, dec
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # residual on top of the two bases that leave room at 0.075 bpp
    base = {(r["key"], r["name"]): r for r in P.rows if r["cand"] == "s1res" and r["key"] in base_keys}
    jobs = []
    for lv in a.s1_res_levels:
        bk = f"s1L{lv}"
        for key, name in P.todo("s1res", [f"{bk}+qp{q}" for q in RES_QPS], names):
            if (bk, name) in base:
                jobs.append(("s1res", key, name, "res", int(key.split("qp")[1]), str(P.recon("s1res", bk, name)),
                             int(base[(bk, name)]["bytes_base"])))
    if jobs:
        _run_vtm_jobs(a, P, jobs)


# ---------------------------------------------------------------- C2 / C3: learned codecs with entropy coding
def _pip(*pkgs):
    """Install without touching the already-imported stack: an unpinned `pip install compressai` downgraded
    numpy 2 -> 1.26 on Kaggle and broke every later import (numpy.dtype size changed, torch_geometric)."""
    import importlib.metadata as md
    pins = []
    for dist in ("numpy", "torch", "torchvision", "scipy", "pillow"):
        try:
            pins.append(f"{dist}=={md.version(dist)}")
        except md.PackageNotFoundError:
            pass
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(chr(10).join(pins))
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", f.name, *pkgs])
    if r.returncode:  # the pins conflict with the package's own pins: install it alone, keep the stack
        print(f"pip: {pkgs} conflict with the installed stack; installing with --no-deps", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *pkgs], check=True)


def _learned(a, P, names, cand, models, load):
    """models: {key: loader arg}; load(arg) -> (model, compress(x01) -> (nbytes, state), decompress(state) -> x01)."""
    import torch
    dev = M.device()
    for key, arg in models.items():
        todo = [n for k, n in P.todo(cand, [key], names)]
        if not todo:
            continue
        model, comp, decomp, n_params = load(arg)
        for name in todo:
            x = M.load(inputs(a)[name])
            t = M.to_tensor(x, dev)
            with torch.no_grad():
                nbytes, state = comp(t)
                sync()
                t0 = time.time()
                y = decomp(state)
                sync()
                t_dec = time.time() - t0
            M.save(to_u8(y)[:x.shape[0], :x.shape[1]], P.recon(cand, key, name))
            P.add(dict(cand=cand, key=key, name=name, bytes=nbytes + HEADER_BYTES, bytes_base=nbytes, bytes_res=0,
                       t_dec=t_dec, note=f"decoder_params={n_params}"))
            print(f"{cand} {key} {name[:8]} {nbytes} B  dec {t_dec:.2f}s", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _msillm_load(hub_name):
    import torch
    try:
        import compressai  # noqa: F401
        import fvcore  # noqa: F401
    except ImportError:
        _pip("compressai", "fvcore")
    m = torch.hub.load("facebookresearch/NeuralCompression", hub_name, trust_repo=True)
    m = m.to(M.device()).eval()
    m.update()
    m.update_tensor_devices("compress")  # entropy coding on CPU, transforms on GPU

    def comp(t):
        c = m.compress(t, force_cpu=False)
        n = sum(len(s) for s in c.latent_strings) + sum(len(s) for s in c.hyper_latent_strings)
        return n, c

    n_params = sum(p.numel() for p in m.decoder.parameters()) + sum(p.numel() for p in m.hyper_synthesis_mean.parameters()) \
        + sum(p.numel() for p in m.hyper_synthesis_scale.parameters())
    return m, comp, lambda c: m.decompress(c, force_cpu=False), n_params


def points_msillm(a, P, names):
    _learned(a, P, names, "msillm", {f"q{q}": n for q, n in MSILLM.items()}, _msillm_load)


# ---------------------------------------------------------------- C4 / C5: one-step diffusion
def points_codlite(a, P, names):
    """CoD-Lite (Jia et al. 2026, microsoft/GenCodec, MIT code + weights): pixel-space one-step diffusion decoder on
    VQ indices sent at a fixed length (no entropy coding), so bytes = ceil(h/ds * w/ds * bits / 8)."""
    import torch
    from huggingface_hub import hf_hub_download
    src = a.out.parent / "GenCodec"
    if not (src / "CoD_Lite").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/microsoft/GenCodec.git", str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", GENCODEC_COMMIT], check=True)
    sys.path.insert(0, str(src / "CoD_Lite"))
    import yaml
    from cod.utils.test_utils import instantiate_class
    dev = M.device()
    # bf16 as trained on Ampere+; the T4 has no bf16, run fp32 there (fp16 risks overflow in a bf16-trained net)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        and torch.cuda.get_device_capability()[0] >= 8 else None
    for key in CODLITE:
        todo = [n for k, n in P.todo("codlite", [key], names)]
        if not todo:
            continue
        cfg = yaml.safe_load(Path(hf_hub_download("zhaoyangjia/CoD_Lite", f"CoD_Lite_bpp_{key}.yaml")).read_text())
        net = instantiate_class(cfg["model"]["net"])
        sd = torch.load(hf_hub_download("zhaoyangjia/CoD_Lite", f"CoD_Lite_bpp_{key}.pt"), map_location="cpu")
        sd = sd.get("state_dict", sd.get("module", sd))
        miss, extra = net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")}, strict=False)
        if miss or extra:
            print(f"codlite {key}: {len(miss)} missing / {len(extra)} unexpected keys, e.g. {(miss + extra)[:4]}", flush=True)
        net = net.to(dev).eval()  # eval() folds the t=0 modulation into the blocks
        enc = getattr(getattr(net, "y_embedder", None), "encoder", None)  # None in the released CoD-Lite nets
        n_params = sum(p.numel() for p in net.parameters()) - (sum(p.numel() for p in enc.parameters()) if enc is not None else 0)
        for name in todo:
            x = M.load(inputs(a)[name])
            H, W = x.shape[:2]
            ph, pw = (-H) % 64, (-W) % 64
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(255).to(dev)
            t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                bits = net.compress(t)
                sync()
                t0 = time.time()
                cond = net.decompress(bits, H + ph, W + pw, dev)
                y = net.inference(y=torch.zeros((1, 3, H + ph, W + pw), device=dev, dtype=cond.dtype), cond=cond)
                sync()
                t_dec = time.time() - t0
            # CoD's fp2uint8: the net outputs [-1, 1]
            rec = y[0, :, :H, :W].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            M.save(rec, P.recon("codlite", key, name))
            P.add(dict(cand="codlite", key=key, name=name, bytes=len(bits) + HEADER_BYTES, bytes_base=len(bits),
                       bytes_res=0, t_dec=t_dec, note=f"decoder_params={n_params}"))
            print(f"codlite {key} {name[:8]} {len(bits)} B  dec {t_dec:.2f}s", flush=True)
        del net
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def points_turbo(a, P, names):
    """C4 probe, no training: MS-ILLM reconstruction -> SD-Turbo img2img, one step at a low strength, empty prompt.
    Same bytes as the MS-ILLM point. Tells whether a one-step diffusion refiner on top of the best base is worth
    training; SD-Turbo works through the SD 2.1 VAE (24.7 dB ceiling on these images), so only the low rates matter."""
    import torch
    try:
        import diffusers  # noqa: F401
    except ImportError:
        _pip("diffusers", "accelerate")
    from diffusers import AutoPipelineForImage2Image
    from PIL import Image
    dev = M.device()
    keys = [f"q{q}s{int(s * 100)}" for q in TURBO_BASE for s in TURBO_STRENGTH]
    todo = P.todo("turbo", keys, names)
    if not todo:
        return
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                      variant="fp16").to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    n_turbo = sum(p.numel() for p in pipe.unet.parameters()) + sum(p.numel() for p in pipe.vae.parameters())
    for q in TURBO_BASE:
        mine = [(k, n) for k, n in todo if k.startswith(f"q{q}s")]
        if not mine:
            continue
        model, comp, decomp, n_ms = _msillm_load(MSILLM[q])
        for name in sorted({n for _, n in mine}):
            x = M.load(inputs(a)[name])
            H, W = x.shape[:2]
            with torch.no_grad():
                nbytes, state = comp(M.to_tensor(x, dev))
                sync()
                t0 = time.time()
                base = to_u8(decomp(state))[:H, :W]
                sync()
                t_base = time.time() - t0
            ph, pw = (-H) % 64, (-W) % 64
            img = Image.fromarray(np.pad(base, ((0, ph), (0, pw), (0, 0)), mode="reflect"))
            for s in TURBO_STRENGTH:
                key = f"q{q}s{int(s * 100)}"
                if (key, name) not in mine:
                    continue
                sync()
                t0 = time.time()
                out = pipe("", image=img, num_inference_steps=math.ceil(1 / s), strength=s, guidance_scale=0.0,
                           generator=torch.Generator(dev).manual_seed(0)).images[0]
                sync()
                t_dec = t_base + time.time() - t0
                M.save(np.asarray(out)[:H, :W], P.recon("turbo", key, name))
                P.add(dict(cand="turbo", key=key, name=name, bytes=nbytes + HEADER_BYTES, bytes_base=nbytes, bytes_res=0,
                           t_dec=t_dec, note=f"decoder_params={n_ms + n_turbo}"))
                print(f"turbo {key} {name[:8]} {nbytes} B  dec {t_dec:.2f}s", flush=True)
        del model
    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def points_coolchic(a, P, names):
    """Cool-chic (Orange-OpenSource/Cool-Chic, BSD-3): one overfitting run per (image, tune, lambda) with its own
    cc_encode.py / cc_decode.py, so bytes are the real .cool bitstream. t_dec is the wall time of cc_decode.py
    (includes Python start-up and imports, so it overstates the decoder itself); t_enc goes to the note."""
    import shutil
    src = a.out.parent / "Cool-Chic"
    if not (src / "cc_encode.py").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/Orange-OpenSource/Cool-Chic.git", str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", COOLCHIC_COMMIT], check=True)
    _pip("constriction==0.4.2", "einops", "fvcore", "ConfigArgParse")
    imgs = inputs(a)
    for tune in a.coolchic_tunes:
        for lm in a.coolchic_lambdas:
            key = f"{'wd' if tune == 'wasserstein' else 'mse'}{lm:g}"
            for _, name in P.todo("coolchic", [key], names):
                work = Path(tempfile.mkdtemp(prefix="coolchic_"))
                bs = work / "x.cool"
                t0 = time.time()
                r = subprocess.run([sys.executable, str(src / "cc_encode.py"), "--input", str(imgs[name]),
                                    "--output", str(bs), "--workdir", str(work), "--lmbda", str(lm), "--tune", tune,
                                    "--n_itr", str(a.coolchic_itr)], cwd=src, capture_output=True, text=True)
                t_enc = time.time() - t0
                if r.returncode or not bs.exists():
                    print(f"coolchic {key} {name[:8]} encode FAILED:", (r.stdout + r.stderr)[-2000:], flush=True)
                    shutil.rmtree(work, ignore_errors=True)
                    continue
                rec = P.recon("coolchic", key, name)
                rec.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                subprocess.run([sys.executable, str(src / "cc_decode.py"), "-i", str(bs), "-o", str(rec)], cwd=src,
                               check=True, capture_output=True)
                t_dec = time.time() - t0
                n = bs.stat().st_size
                P.add(dict(cand="coolchic", key=key, name=name, bytes=n + HEADER_BYTES, bytes_base=n, bytes_res=0,
                           t_dec=t_dec, note=f"t_enc={t_enc:.0f}s itr={a.coolchic_itr}"))
                print(f"coolchic {key} {name[:8]} {n} B  enc {t_enc:.0f}s  dec {t_dec:.1f}s", flush=True)
                shutil.rmtree(work, ignore_errors=True)


def points_mbt(a, P, names):
    import torch
    try:
        import compressai  # noqa: F401
    except ImportError:
        _pip("compressai")
    from compressai.zoo import mbt2018_mean

    def load(q):
        m = mbt2018_mean(quality=q, metric="mse", pretrained=True).to(M.device()).eval()
        m.update(force=True)

        def comp(t):
            x = t
            H, W = x.shape[-2:]
            ph, pw = (-H) % 64, (-W) % 64
            c = m.compress(torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate"))
            return sum(len(s[0]) for s in c["strings"]), c

        n_params = sum(p.numel() for p in m.g_s.parameters()) + sum(p.numel() for p in m.h_s.parameters())
        return m, comp, lambda c: m.decompress(c["strings"], c["shape"])["x_hat"], n_params

    _learned(a, P, names, "mbt", {f"q{q}": q for q in MBT_Q}, load)


def points(a):
    P = Points(a.out, Scorer(a))
    names = selected(a)
    for cand, fn in (("msillm", points_msillm), ("mbt", points_mbt), ("codlite", points_codlite), ("turbo", points_turbo),
                     ("coolchic", points_coolchic), ("s1res", points_s1res)):
        if cand in a.cands:
            try:
                fn(a, P, names)
            except Exception as e:  # noqa: BLE001  one broken candidate must not stop the others
                import traceback
                traceback.print_exc()
                print(f"== {cand} FAILED: {e}", flush=True)
    if {"vtm420", "vtmscc"} & set(a.cands):
        points_vtm(a, P, names)


# ---------------------------------------------------------------- metrics + Q-hat
def load_qhat(path):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def qhat_score(model, f):
    from qhat import transform
    return sum(w * transform(k, float(f[k])) for k, w in zip(model["features"], model["weights"]))


def metrics(a):
    """Points made before metrics were computed inline (or by an older version of this script)."""
    S = Scorer(a)
    P = Points(a.out)
    todo = [r for r in P.rows if (r["cand"], r["key"], r["name"]) not in S.have]
    for i, r in enumerate(todo):
        S(P, r)
        if i % 20 == 0:
            print(f"metrics {i + 1}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- allocation
def allocate_one(options, budget, units=20000):
    """options: {name: [(bytes, score, key), ...]}. One option per image, sum(bytes) <= budget, max sum(score).
    Multiple-choice knapsack by dynamic programming over byte costs rounded UP to budget/units, so the result
    always fits the budget and is optimal up to that rounding. Returns {name: option} or None if infeasible."""
    q = max(1, budget // units)
    U = budget // q
    names = list(options)
    NEG = -np.inf
    best = np.full(U + 1, NEG)
    best[0] = 0.0  # best[u]: max score with total rounded cost exactly u
    choice = []
    for n in names:
        new = np.full(U + 1, NEG)
        arg = np.full(U + 1, -1, dtype=np.int32)
        for j, (b, s, _) in enumerate(options[n]):
            c = -(-b // q)
            if c > U:
                continue
            cand = np.full(U + 1, NEG)
            cand[c:] = best[:U + 1 - c] + s
            better = cand > new
            new[better], arg[better] = cand[better], j
        best = new
        choice.append(arg)
    if not np.isfinite(best).any():
        return None
    u = int(np.argmax(best))
    sel = {}
    for n, arg in zip(reversed(names), reversed(choice)):
        o = options[n][arg[u]]
        sel[n] = o
        u -= -(-o[0] // q)
    return sel


def allocate(a):
    P = Points(a.out)
    met = {(r["cand"], r["key"], r["name"]): r for r in M.read_csv(a.out / "metrics.csv")}
    budgets = json.loads((a.out / "budgets.json").read_text())["budgets_bytes"]
    q = load_qhat(a.qhat)
    names = sorted(inputs(a))

    def score(m):
        return qhat_score(q, m) if q else -float(m["lpips_alex"])

    by_cand = {}
    for r in P.rows:
        k = (r["cand"], r["key"], r["name"])
        if k in met:
            by_cand.setdefault(r["cand"], {}).setdefault(r["name"], []).append((int(r["bytes"]), score(met[k]), r["key"]))
    by_cand["mix"] = {n: [(b, s, f"{c}/{k}") for c, d in by_cand.items() for b, s, k in d.get(n, [])] for n in names}
    out = []
    for cand, options in by_cand.items():
        if any(n not in options for n in names):
            print(f"allocate: {cand} misses images, skipped", flush=True)
            continue
        for rate in RATES:
            sel = allocate_one(options, budgets[str(rate)])
            if sel is None:
                print(f"allocate: {cand} cannot reach {rate} bpp (cheapest points exceed the budget)", flush=True)
                continue
            for n, (b, s, key) in sel.items():
                c, k = key.split("/", 1) if cand == "mix" else (cand, key)
                out.append(dict(cand=cand, rate=rate, name=n, point_cand=c, key=k, bytes=b, score=s))
    M.write_csv(out, a.out / "allocation.csv")
    (a.out / "allocation_meta.json").write_text(json.dumps(dict(objective="qhat_v0" if q else "-lpips_alex",
                                                                qhat=str(a.qhat) if q else None), indent=1))


# ---------------------------------------------------------------- summary
def crop_sheet(a, alloc, rate, names, cands, size=320):
    from PIL import Image, ImageDraw
    src = inputs(a)
    P = Points(a.out)
    rows = []
    for n in names:
        x = M.load(src[n])
        H, W = x.shape[:2]
        y0, x0 = (H - size) // 2, (W - size) // 2
        tiles = [("original", x[y0:y0 + size, x0:x0 + size])]
        for c in cands:
            r = alloc.get((c, rate, n))
            if r:
                im = M.load(P.recon(r["point_cand"], r["key"], n))
                tiles.append((f"{c} {int(r['bytes']) * 8 / (H * W):.3f}", im[y0:y0 + size, x0:x0 + size]))
        row = Image.new("RGB", (size * len(tiles), size + 18), "white")
        d = ImageDraw.Draw(row)
        for i, (t, im) in enumerate(tiles):
            row.paste(Image.fromarray(im), (i * size, 18))
            d.text((i * size + 4, 3), t, fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    yy = 0
    for r in rows:
        sheet.paste(r, (0, yy))
        yy += r.height
    (a.out / "crops").mkdir(exist_ok=True)
    sheet.save(a.out / "crops" / f"rate{rate}.jpg", quality=88)


def summary(a):
    P = Points(a.out)
    met = {(r["cand"], r["key"], r["name"]): r for r in M.read_csv(a.out / "metrics.csv")}
    pts = {(r["cand"], r["key"], r["name"]): r for r in P.rows}
    alloc_rows = M.read_csv(a.out / "allocation.csv")
    alloc = {(r["cand"], float(r["rate"]), r["name"]): r for r in alloc_rows}
    budgets = json.loads((a.out / "budgets.json").read_text())
    meta = json.loads((a.out / "allocation_meta.json").read_text())
    q = load_qhat(a.qhat)
    names = sorted(inputs(a))
    sizes = {n: tuple(v) for n, v in budgets["sizes"].items()}
    cands = [c for c in CANDS + ["mix"] if any(k[0] == c for k in alloc)]

    def stats(cand, rate, group):
        rs = [alloc[(cand, rate, n)] for n in names if (cand, rate, n) in alloc
              and (group == "all" or label(n) == group)]
        if not rs:
            return None
        ms = [met[(r["point_cand"], r["key"], r["name"])] for r in rs]
        px = [sizes[r["name"]][0] * sizes[r["name"]][1] for r in rs]
        t = [float(pts[(r["point_cand"], r["key"], r["name"])]["t_dec"]) for r in rs]
        t = [v for v in t if math.isfinite(v)]
        mean = lambda k: float(np.nanmean([float(m[k]) for m in ms]))  # noqa: E731
        return dict(n=len(rs), bytes=sum(int(r["bytes"]) for r in rs), bpp=sum(int(r["bytes"]) for r in rs) * 8 / sum(px),
                    psnr=M.pooled_psnr([float(m["mse"]) for m in ms], px), msssim=mean("msssim"),
                    lpips=mean("lpips_alex"), dists=mean("dists"),
                    qhat=float(np.mean([qhat_score(q, m) for m in ms])) if q else float("nan"),
                    t_dec=float(np.mean(t)) if t else float("nan"))

    L = ["# So sánh base trên 30 ảnh validation CLIC 2027", "",
         f"Ngân sách cả bộ ảnh (byte): {budgets['budgets_bytes']}. Mỗi ảnh chọn một điểm vận hành sao cho tổng byte "
         f"≤ ngân sách và tổng mục tiêu lớn nhất; mục tiêu: **{meta['objective']}**.", "",
         "Q̂ là tiện ích tuyến tính của Q̂ v0 (cao hơn = người chấm thích hơn; chênh 1.0 ≈ odds 2.7 lần). "
         "`t_dec`: giây/ảnh trên GPU của lần chạy này (không gồm VTM). `mix`: mỗi ảnh được chọn mode tốt nhất.", ""]
    res = {}
    for rate in RATES:
        for group in ("all", "natural", "screen"):
            L += [f"## {rate} bpp — {group}", "", "| cand | n | bpp | PSNR | MS-SSIM | LPIPS↓ | DISTS↓ | Q̂ | t_dec (s) |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for c in cands:
                s = stats(c, rate, group)
                if s:
                    res[f"{c}@{rate}/{group}"] = s
                    L.append(f"| {c} | {s['n']} | {s['bpp']:.4f} | {s['psnr']:.2f} | {s['msssim']:.4f} | {s['lpips']:.4f} "
                             f"| {s['dists']:.4f} | {s['qhat']:.3f} | {s['t_dec']:.2f} |")
            L.append("")
        try:
            crop_sheet(a, alloc, rate, names, [c for c in cands if c != "mix"])
        except Exception as e:  # noqa: BLE001
            print("crop sheet failed:", e)
    mix = [r for r in alloc_rows if r["cand"] == "mix"]
    if mix:
        L += ["## Ảnh nào được `mix` giao cho base nào", "", "| rate | " + " | ".join(CANDS) + " |",
              "|---|" + "---|" * len(CANDS)]
        for rate in RATES:
            cnt = {c: sum(1 for r in mix if float(r["rate"]) == rate and r["point_cand"] == c) for c in CANDS}
            L.append(f"| {rate} | " + " | ".join(str(cnt[c]) for c in CANDS) + " |")
    dec_params = {}
    for r in P.rows:
        if "decoder_params=" in r.get("note", ""):
            dec_params[r["cand"]] = int(r["note"].split("=")[1])
    if dec_params:
        L += ["", "Kích thước decoder (fp32): " + ", ".join(f"{c} {v * 4 / 1e6:.0f} MB" for c, v in dec_params.items())]
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    (a.out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(L) + "\n").encode("utf-8"))


# ---------------------------------------------------------------- driver
STEPS = ["prepare", "points", "metrics", "allocate", "summary"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--out", type=Path, default=Path("work/bakeoff"))
    ap.add_argument("--data", type=Path, default=None, help="default: <out>/valid")
    ap.add_argument("--cands", nargs="+", default=CANDS, choices=CANDS)
    ap.add_argument("--vtm", default="vtm", help="VTM 23.8 checkout with bin/ built")
    ap.add_argument("--s1-ckpt", default=None, help="S1 checkpoint (run 1, lambdas 0.03..4: covers 0.02..0.12 bpp)")
    ap.add_argument("--s1-res-levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--qhat", default=None, help="qhat_v0.json; without it the allocator maximises -LPIPS")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--qps", type=int, nargs="+", default=VTM_QPS, help="VTM QP grid for vtm420 / vtmscc")
    ap.add_argument("--only", nargs="+", default=None,
                    help="points for these images only: labels (screen, texture, natural) or name prefixes")
    ap.add_argument("--coolchic-lambdas", type=float, nargs="+", default=COOLCHIC_LAMBDAS)
    ap.add_argument("--coolchic-tunes", nargs="+", default=["wasserstein"], choices=["wasserstein", "mse"])
    ap.add_argument("--coolchic-itr", type=int, default=10000, help="Cool-chic training iterations per image")
    a = ap.parse_args(argv)
    a.data = a.data or a.out / "valid"
    steps = STEPS if a.step == "all" else [a.step]
    if steps[0] != "prepare" and not (a.data.exists() and inputs(a)):
        steps = ["prepare"] + steps  # a single later step (e.g. points in a CPU session) still needs the images
    for s in steps:
        print(f"===== {s}", flush=True)
        globals()[s](a)


if __name__ == "__main__":
    main()
