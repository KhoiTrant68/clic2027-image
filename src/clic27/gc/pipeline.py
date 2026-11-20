"""End-to-end driver: data -> reward -> train -> encode/allocate -> package -> verify (fresh-process decode) -> report.

    python -m clic27.gc.pipeline --config configs/gc_full.json --work WORK [--stages ...] [--smoke]

Stages (default: all except prefetch, in this order; each one resumes / skips finished work):
  prefetch download everything that needs the Internet (CLIC/DIV2K zips, CLIC rated crops, VGG16/AlexNet/LPIPS/DISTS
           weights into $TORCH_HOME). Run it on a machine with network; after it, every other stage runs offline.
  env      build the C rANS, print versions, check the GPU and the integer-net exactness CPU vs CUDA
  data     download the CLIC validation set (= CLIC 2025 test) and, if asked, DIV2K
  reward   fetch CLIC rated crops and fit Q-hat v2 (skipped if alloc/align do not use it and --no-reward)
  train    run the training stages listed in cfg["pipeline"]["train_stages"]
  encode   candidates (levels x quality-map trajectory) -> knapsack per budget -> real bitstreams + bs.zip
  package  decoder folder + decoder.zip
  verify   decode every bs.zip with the packaged decoder in a fresh process; metrics, timing, byte budget check
  report   WORK/report.md
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import config as C

STAGES = ["env", "data", "reward", "train", "encode", "package", "verify", "report", "diagnose"]
ALL_STAGES = ["prefetch"] + STAGES
CLIC_VALID_URL = "https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip"
DIV2K = {"div2k_train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
         "div2k_valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"}
SRC = Path(__file__).resolve().parents[2]


def sh(cmd, env=None, cwd=None):
    """Run a command; SIGTERM / SIGINT / SIGUSR1 sent to the pipeline are forwarded to it (training checkpoints)."""
    print("$ " + " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), env=env, cwd=cwd)
    names = [n for n in ("SIGTERM", "SIGINT", "SIGUSR1") if hasattr(signal, n)]
    old = {n: signal.signal(getattr(signal, n), lambda s, _f: p.send_signal(s)) for n in names}
    try:
        rc = p.wait()
    finally:
        for n, h in old.items():
            signal.signal(getattr(signal, n), h)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def py_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def download(url: str, dest: Path):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[data] {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def unzip(zp: Path, out: Path):
    marker = out / ".done"
    if marker.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(out)
    marker.write_text("ok")
    return out


# ---------------------------------------------------------------- stages
def st_prefetch(cfg, work, a):
    """Everything that needs the Internet, so compute nodes without network can run the rest."""
    root = Path(cfg["pipeline"]["data_root"])
    download(CLIC_VALID_URL, root / "downloads" / "clic2025_image_test.zip")
    for name in cfg["pipeline"].get("download", []):
        if name in DIV2K:
            download(DIV2K[name], root / "downloads" / f"{name}.zip")
    rw = Path(cfg["reward_dir"])
    if cfg["pipeline"].get("use_reward", True) and not (rw / "reward.pt").exists() and not (rw / "index.jsonl").exists():
        R = cfg["reward"]
        sh([sys.executable, "-m", "clic27.gc.reward", "fetch", "--out", rw, "--n", R["n_per_set"],
            "--sets", *(R["sets_train"] + R["sets_test"])], env=py_env())
    # Pretrained weights used by the losses, Q-hat v2 and the metrics land in $TORCH_HOME (run_e2e.sh sets it).
    from .losses import LPIPS, WassersteinLoss
    from .reward import PrefDistance
    LPIPS("alex"), WassersteinLoss(), PrefDistance(pretrained=True)
    import piq
    piq.DISTS()
    print(f"[prefetch] weights cached under {os.environ.get('TORCH_HOME', '~/.cache/torch')}", flush=True)
    print("[prefetch] done: copy GC_ROOT (data/, reward/, cache/) to the training machine if it differs", flush=True)


def st_env(cfg, work, a):
    import torch
    from . import rans_c
    from .model import GCodec
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    if rans_c.lib() is None:
        # numpy rANS is ~100x slower: encode would crawl and the CLIC decoder package would ship without _rans.so.
        if os.environ.get("GC_ALLOW_NUMPY_RANS") != "1":
            raise SystemExit("rANS C library missing: install gcc (or set CC) and re-run; "
                             "GC_ALLOW_NUMPY_RANS=1 only for tiny debug runs")
        print("rANS C library: MISSING, numpy fallback allowed by GC_ALLOW_NUMPY_RANS=1", flush=True)
    else:
        print("rANS C library: ok", flush=True)
    m = GCodec(cfg["model"])
    m.check_ranges()
    if torch.cuda.is_available():  # integer nets: CPU vs CUDA must agree exactly
        from .codec import _ent_ctx
        z = torch.randint(-20, 20, (1, cfg["model"]["Z"], 3, 4)).double()
        lv = torch.tensor([1])
        with torch.no_grad():
            for p in m.h_s.parameters():
                p.data.normal_(0, 0.3)
            a_cpu = m.h_s.forward_int(z, lv)
            with _ent_ctx("cuda"):
                a_gpu = m.h_s.to("cuda").forward_int(z.cuda(), lv).cpu()
        same = torch.equal(a_cpu, a_gpu)
        print(f"integer net CPU == CUDA: {same}", flush=True)
        if not same:
            raise SystemExit("integer nets differ between CPU and CUDA")


def st_data(cfg, work, a):
    root = Path(cfg["pipeline"]["data_root"])
    zp = download(CLIC_VALID_URL, root / "downloads" / "clic2025_image_test.zip")
    valid = unzip(zp, root / "clic_valid")
    for name in cfg["pipeline"].get("download", []):
        if name in DIV2K:
            unzip(download(DIV2K[name], root / "downloads" / f"{name}.zip"), root / name)
    print(f"[data] validation images: {len(list(valid.rglob('*.png')))}", flush=True)
    prep = cfg["pipeline"].get("prep_patches")  # {"out": dir, "per_image": 4, "size": 512, "workers": 32}
    if prep:
        from .data import prep_patches
        out = Path(prep["out"].replace("@data_root", str(root)))
        # On a re-run train_dirs already points at the patches; cut from (and verify against) the original sources.
        prep_patches(cfg["data"].get("patch_sources", cfg["data"]["train_dirs"]), out, prep.get("per_image", 4),
                     prep.get("size", 512), prep.get("workers", 32))
        cfg["data"].setdefault("patch_sources", cfg["data"]["train_dirs"])
        cfg["data"]["train_dirs"] = [str(out)]
        (work / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[data] training now reads patches from {out}", flush=True)


def st_reward(cfg, work, a):
    rw = Path(cfg["reward_dir"])
    if (rw / "reward.pt").exists():
        print("[reward] exists:", rw / "reward.pt")
        return
    R = cfg["reward"]
    sets = R["sets_train"] + R["sets_test"]
    sh([sys.executable, "-m", "clic27.gc.reward", "fetch", "--out", rw, "--n", R["n_per_set"], "--sets", *sets],
       env=py_env())
    sh([sys.executable, "-m", "clic27.gc.reward", "fit", "--out", rw, "--steps", R["steps"], "--batch", R["batch"],
        "--lr", R["lr"], "--crop", R["crop"], "--test-sets", *R["sets_test"]], env=py_env())


def st_train(cfg, work, a):
    cfg_path = work / "config.json"
    nproc = int(cfg["pipeline"].get("nproc", 1))
    inits = cfg["pipeline"].get("init") or {}  # {"rd": "/path/model.pt"}: start that stage from these weights
    for stage in cfg["pipeline"]["train_stages"]:
        if stage == "align" and not Path(cfg.get("reward_path", "")).exists():
            print("[train] skip align: no reward model", flush=True)
            continue
        cmd = ([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"] if nproc > 1
               else [sys.executable]) + ["-m", "clic27.gc.train", "--config", cfg_path, "--work", work / "train",
                                         "--stage", stage]
        if inits.get(stage):
            src = Path(inits[stage].replace("@data_root", cfg["pipeline"]["data_root"]))
            if not src.is_file():
                raise SystemExit(f"[train] pipeline.init.{stage} = {src} does not exist")
            cmd += ["--init", src]  # ignored once the stage has its own last.pt (resume wins)
        if cfg["pipeline"].get("max_minutes_per_stage"):
            cmd += ["--max-minutes", cfg["pipeline"]["max_minutes_per_stage"]]
        try:
            sh(cmd, env=py_env())
        except subprocess.CalledProcessError as e:
            partial = work / "train" / stage / "partial.json"
            if partial.exists():  # checkpointed early stop (signal / time budget): resumable, not a failure
                info = json.loads(partial.read_text())
                print(f"[train] {stage} stopped at step {info['step']}/{info['steps']} ({info['reason']}). "
                      f"Run the same pipeline command again to resume; later stages were not started.", flush=True)
                raise SystemExit(75) from None
            raise


def final_model(cfg, work) -> Path:
    for stage in reversed(cfg["pipeline"]["train_stages"]):
        p = work / "train" / stage / "model.pt"
        if p.exists():
            return p
    raise FileNotFoundError("no trained model.pt under " + str(work / "train"))


def zip_overhead(name: str) -> int:
    n = len(name) + 3  # "<name>.gc"
    return 30 + n + 46 + n  # local header + central directory entry (stored, no extras)


def st_encode(cfg, work, a):
    import torch
    from PIL import Image
    from . import alloc as AL
    from . import codec
    from .model import pad64
    from .package import write_bs_zip
    from .train import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(final_model(cfg, work), dev)
    A = cfg["alloc"]
    rt = cfg["runtime"]
    ent = rt["entropy_device"] if dev == "cuda" else "cpu"
    rw = None
    if Path(cfg.get("reward_path", "")).exists():
        from .reward import load_reward
        rw = load_reward(cfg["reward_path"], dev)
    scorer = AL.CropScorer(dev, rw, A["crop"], A["stride"])
    valid = sorted(Path(cfg["pipeline"]["valid_dir"]).rglob("*.png"))[: cfg["pipeline"].get("max_images") or None]
    imgs = {p.stem: np.array(Image.open(p).convert("RGB")) for p in valid}
    total_px = sum(v.shape[0] * v.shape[1] for v in imgs.values())
    levels = sorted({l for b in A["budgets"] for l in A["levels"][str(b)]})
    betas = [round(b * (model.beta_levels - 1)) for b in A["betas"]]
    cache = work / "encode" / "candidates.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cands = torch.load(cache, weights_only=False) if cache.exists() else {}
    t0 = time.time()
    from .package import sha256_file
    key = dict(model=sha256_file(final_model(cfg, work)), qmap_iters=A["qmap_iters"], objective=A["objective"], kappa=A["kappa"], crop=A["crop"],
               stride=A["stride"], scorer="reward" if rw is not None else "lpips")
    if cands.get("__key__") != key:  # trajectory settings changed: old candidates are not comparable
        cands = {"__key__": key}
    for name, arr in imgs.items():
        have = {(c.level, c.beta) for c in cands.get(name, [])}
        todo = [(l, b) for l in levels for b in betas if (l, b) not in have]  # new levels / betas only
        if not todo:
            continue
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].float().div(255).to(dev)
        cands[name] = cands.get(name, []) + [
            c for l, b in todo
            for c in AL.qmap_trajectory(model, x, l, b, scorer, A["qmap_iters"], A["objective"], A["kappa"], dev)]
        torch.save(cands, cache)
        print(f"[encode] candidates {name}: {len(cands[name])} ({time.time() - t0:.0f}s)", flush=True)
    names = sorted(imgs)
    summary = {}
    for bpp in A["budgets"]:
        out = work / "encode" / f"bpp-{bpp}"
        out.mkdir(parents=True, exist_ok=True)
        budget = bpp * total_px / 8 - 22  # minus the zip end-of-central-directory record
        allowed = set(A["levels"][str(bpp)])
        cl = [[c for c in cands[n] if c.level in allowed and c.beta in betas] for n in names]
        safety = A["safety"]
        for attempt in range(8):
            picks = AL.knapsack(cl, budget * safety, [zip_overhead(n) + 24 for n in names], units=6000)
            feasible = picks is not None
            if not feasible:
                picks = [int(np.argmin([c.est_bytes for c in cs])) for cs in cl]
            streams = {}
            for n, cs, j in zip(names, cl, picks):
                c = cs[j]
                x = torch.from_numpy(imgs[n]).permute(2, 0, 1)[None].float().div(255).to(dev)
                y_s = None
                if A["refine_steps"]:
                    xp, _ = pad64(x)
                    y_s = AL.refine_latent(model, xp, c.level, c.beta, c.cells, A["refine_steps"], 1e-2,
                                           lam=cfg["rate"]["lambda_lo"])
                streams[n] = codec.compress(model, x, c.level, c.beta, torch.from_numpy(c.cells)[None, None],
                                            ent_dev=ent, y_s=y_s)
            size = write_bs_zip(streams, out / "bs.zip")
            print(f"[encode] {bpp} bpp attempt {attempt}: bs.zip {size} B = {8 * size / total_px:.5f} bpp "
                  f"(budget {bpp}, feasible {feasible}, safety {safety:.4f})", flush=True)
            if 8 * size / total_px <= bpp or not feasible:
                break
            safety *= 0.995
        chosen = [dict(image=n, level=cs[j].level, beta=cs[j].beta, iter=cs[j].meta.get("iter"),
                       est_bytes=cs[j].est_bytes, bytes=len(streams[n]), obj=cs[j].obj,
                       cells_nonzero=int(np.count_nonzero(cs[j].cells)))
                  for n, cs, j in zip(names, cl, picks)]
        summary[str(bpp)] = dict(bs_zip_bytes=size, bpp=8 * size / total_px, budget_bpp=bpp, feasible=feasible,
                                 within_budget=8 * size / total_px <= bpp, images=chosen)
        (out / "allocation.json").write_text(json.dumps(summary[str(bpp)], indent=2))
    (work / "encode" / "summary.json").write_text(json.dumps(summary, indent=2))


def st_package(cfg, work, a):
    from .package import build, sha256_file
    out = work / "package"
    out.mkdir(parents=True, exist_ok=True)
    build(final_model(cfg, work), out / "decoder", cfg["runtime"], out / "decoder.zip")
    info = dict(decoder_zip=str(out / "decoder.zip"), bytes=(out / "decoder.zip").stat().st_size,
                sha256=sha256_file(out / "decoder.zip"))
    (out / "package.json").write_text(json.dumps(info, indent=2))
    print(f"[package] {info}", flush=True)


def st_verify(cfg, work, a):
    import shutil
    from PIL import Image
    from clic27.eval import metrics as M
    dec = work / "package" / "decoder"
    valid = {p.stem: p for p in Path(cfg["pipeline"]["valid_dir"]).rglob("*.png")}
    perc = M.Perceptual()
    res = {}
    for bdir in sorted((work / "encode").glob("bpp-*")):
        run = bdir / "verify"
        if run.exists():
            shutil.rmtree(run)
        run.mkdir()
        shutil.copy2(bdir / "bs.zip", run / "bs.zip")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # the decoder must work from its own folder only
        env.setdefault("PYTHON", sys.executable)  # same interpreter as this run (the CLIC image has python3 + torch)
        t0 = time.time()
        sh(["bash", dec / "decode", "--report", run / "decode_report.json"], env=env, cwd=run)
        wall = time.time() - t0
        rows, mses, pix = [], [], []
        for p in sorted((run / "images").glob("*.png")):
            o, x = M.load(valid[p.stem]), M.load(p)
            assert o.shape == x.shape, (p.name, o.shape, x.shape)
            r = dict(image=p.stem, **perc.all(o, x))
            rows.append(r)
            mses.append(r["mse"]); pix.append(o.shape[0] * o.shape[1])
        import hashlib
        hashes = {p.stem: hashlib.sha256(np.array(Image.open(p).convert("RGB")).tobytes()).hexdigest()
                  for p in sorted((run / "images").glob("*.png"))}
        (bdir / "expected.json").write_text(json.dumps(hashes, indent=1))  # for scripts/gc/l4_verify.sh
        n_exp = len(json.loads((bdir / "allocation.json").read_text())["images"])
        agg = dict(images=len(rows), expected=n_exp, decode_wall_s=wall,
                   pooled_psnr=M.pooled_psnr(mses, pix),
                   **{k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "msssim", "lpips", "dists")},
                   bpp=json.loads((bdir / "allocation.json").read_text())["bpp"])
        M.write_csv(rows, run / "metrics.csv")
        res[bdir.name] = agg
        print(f"[verify] {bdir.name}: {json.dumps(agg)}", flush=True)
        if len(rows) != n_exp:
            raise SystemExit(f"decoder produced {len(rows)} images, expected {n_exp}")
    (work / "verify.json").write_text(json.dumps(res, indent=2))


def st_report(cfg, work, a):
    lines = ["# GC run report", "", f"work: `{work}`", ""]
    v = json.loads((work / "verify.json").read_text()) if (work / "verify.json").exists() else {}
    if v:
        lines += ["| budget | bpp | images | PSNR (pooled) | MS-SSIM | LPIPS | DISTS | decode wall (s) |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, r in v.items():
            lines.append(f"| {k} | {r['bpp']:.5f} | {r['images']}/{r['expected']} | {r['pooled_psnr']:.2f} | "
                         f"{r['msssim']:.4f} | {r['lpips']:.4f} | {r['dists']:.4f} | {r['decode_wall_s']:.1f} |")
    pj = work / "package" / "package.json"
    if pj.exists():
        p = json.loads(pj.read_text())
        lines += ["", f"decoder.zip: {p['bytes'] / 1e6:.1f} MB, sha256 `{p['sha256']}`"]
    rj = Path(cfg.get("reward_dir", "")) / "report.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        lines += ["", f"Q-hat v2 held-out accuracy: {r['final']['acc']:.4f} (init {r['init']['acc']:.4f}, "
                      f"n={r['final']['n']}, tau {r['tau']:.3f})"]
    lines += ["", "Timings on this machine are not L4 timings; measure on an L4 before deciding speed."]
    (work / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def st_diagnose(cfg, work, a):
    from . import diagnose
    diagnose.main(["--work", str(work)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=ALL_STAGES)
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cfg = C.load(a.config)
    P = cfg.setdefault("pipeline", {})
    P.setdefault("data_root", str(work / "data"))
    P.setdefault("train_stages", ["rd", "gan", "rate"])
    P.setdefault("valid_dir", str(Path(P["data_root"]) / "clic_valid"))
    if not cfg["data"]["valid_dir"]:
        cfg["data"]["valid_dir"] = P["valid_dir"]
    for key in ("train_dirs", "screen_dirs"):  # "@data_root/x" -> <data_root>/x
        cfg["data"][key] = [d.replace("@data_root", P["data_root"]) for d in cfg["data"][key]]
    if P.get("prep_patches"):  # patches already cut from these exact sources by a `data` stage: train on them
        from .data import patches_ready
        pout = Path(P["prep_patches"]["out"].replace("@data_root", P["data_root"]))
        if patches_ready(pout, cfg["data"]["train_dirs"]):
            cfg["data"]["patch_sources"] = cfg["data"]["train_dirs"]
            cfg["data"]["train_dirs"] = [str(pout)]
    cfg.setdefault("reward_dir", str(work / "reward"))
    cfg.setdefault("reward_path", str(Path(cfg["reward_dir"]) / "reward.pt"))
    (work / "config.json").write_text(json.dumps(cfg, indent=2))
    for s in a.stages:
        if s == "reward" and not P.get("use_reward", True):
            print("[reward] disabled by config", flush=True)
            continue
        t = time.time()
        print(f"\n===== stage {s} =====", flush=True)
        globals()[f"st_{s}"](cfg, work, a)
        print(f"===== stage {s} done in {time.time() - t:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
