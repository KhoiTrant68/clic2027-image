"""One entry point for every GPU job: Kaggle or a rented machine, resumable, one results zip.

    python experiments/pipeline.py                     # default stages
    python experiments/pipeline.py gonogo1 s1          # only some stages (dependencies run if not done)
    python experiments/pipeline.py --hours 11 --dry-run

Stages (in order): check, data, cache, gonogo1, s1, s1_eval, [ceiling], [parity], [qhat], [bakeoff], pack
  check     GPU + Internet + packages (installs missing pip packages)
  data      DIV2K train HR (800 images) into <work>/data
  cache     DC-AE latents of the training images into <work>/latents (resumes; reuses latents found in inputs)
  gonogo1   go/no-go 1: exact-noise vs Gaussian-assumption denoiser (refuses < 208 training latents)
  s1        S1 selftest, then training (resumes from <work>/s1/last.pt); time-boxed by --hours
  s1_eval   Kodak, real bitstreams, vs the DC-AE ceiling
  ceiling   (optional) DC-AE / SD-VAE ceilings on Kodak, CLIC2020, DIV2K-val incl. FID/KID
  parity    (optional) ratflow.nn / ratflow.entropy vs diffusers and CPU vs GPU
  qhat      (optional, CLIC) Q-hat v0: metrics on sampled CLIC perceptual ratings + Bradley-Terry fit
  bakeoff   (optional, CLIC) base candidates on the 30 validation images at 0.075/0.15/0.3 bpp, corpus budget,
            Q-hat allocation (builds VTM 23.8; uses --s1-ckpt and work/qhat/qhat_v0.json when present)
  pack      <work>/results.zip with every summary (+ the S1 checkpoint)

State lives in <work>/state.json; finished stages are skipped on re-runs. On Kaggle, attach the previous
run's output as Input and the work directory is restored automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = Path("/kaggle/working").exists()
DEFAULT_STAGES = ["check", "data", "cache", "gonogo1", "s1", "s1_eval", "pack"]
ALL_STAGES = ["check", "data", "cache", "gonogo1", "s1", "s1_eval", "ceiling", "parity", "qhat", "bakeoff", "pack"]
NEEDS = {"qhat": ["check"], "bakeoff": ["check"], "cache": ["data"], "gonogo1": ["cache"], "s1": ["cache"], "s1_eval": ["s1"]}  # deps run if not done
PIP = {"safetensors": "safetensors", "huggingface_hub": "huggingface_hub", "scipy": "scipy", "lpips": "lpips",
       "piq": "piq", "torchmetrics": "torchmetrics", "torch_fidelity": "torch-fidelity", "PIL": "pillow"}
MIN_TRAIN_LATENTS = 208  # gonogo1: 200 train + 8 holdout


class StageTimeout(Exception):
    """The --hours budget ran out while a stage was running; its partial outputs are kept."""


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.state_path = self.work / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.log = open(self.work / "pipeline.log", "a", encoding="utf-8")
        self.commit = (ROOT / ".commit").read_text().strip() if (ROOT / ".commit").exists() else _git_rev()

    # ---------------------------------------------------------------- helpers
    def say(self, *msg):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - self.t0) / 60:5.1f}m] " + " ".join(map(str, msg))
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def run(self, *cmd, cwd=None):
        """Run a child process with live output. It is stopped when the wall-clock budget (--hours) runs out, so
        the pipeline can still pack and exit normally before Kaggle's hard 12 h limit kills everything."""
        cmd = [str(c) for c in cmd]
        self.say("$", " ".join(cmd))
        if self.a.dry_run:
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, cwd=cwd or ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        expired = threading.Event()

        def stop():
            expired.set()
            self.say("time budget reached: stopping", cmd[1] if len(cmd) > 1 else cmd[0])
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()

        timer = threading.Timer(max(1.0, self.hours_left() * 3600), stop)
        timer.daemon = True
        timer.start()
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                self.log.write(line)
            rc = p.wait()
        finally:
            timer.cancel()
        if expired.is_set():
            raise StageTimeout(" ".join(cmd))
        if rc:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")

    def py(self, script, *args):
        self.run(sys.executable, ROOT / script, *args)

    def hours_left(self):
        return self.a.hours - (time.time() - self.t0) / 3600

    def done(self, stage):
        return self.state.get(stage, {}).get("done", False)

    def mark(self, stage, **info):
        self.state[stage] = dict(done=True, seconds=round(time.time() - self._stage_t0), commit=self.commit, **info)
        if not self.a.dry_run:
            self.state_path.write_text(json.dumps(self.state, indent=1))

    def inputs(self):
        roots = [Path(p) for p in self.a.inputs] + ([Path("/kaggle/input")] if KAGGLE else [])
        return [r for r in roots if r.exists()]

    # ---------------------------------------------------------------- restore a previous run
    def restore(self):
        """Merge every previous work dir found in the inputs (oldest first), so a chain of runs
        (e.g. a GPU run, then a CPU-only VTM run) continues where each left off."""
        prevs = []
        for r in self.inputs():
            for st in r.rglob("state.json"):
                prev = st.parent
                if prev.resolve() != self.work.resolve() and (prev / "pipeline.log").exists():
                    prevs.append(prev)
        for prev in sorted(prevs, key=lambda d: (d / "state.json").stat().st_mtime):
            self.say("restoring previous work dir from", prev)
            if self.a.dry_run:
                continue
            old = json.loads((prev / "state.json").read_text())
            shutil.copytree(prev, self.work, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("data", "results.zip", "state.json", "pipeline.log"))
            for k, v in old.items():
                if v.get("done") or k not in self.state:
                    self.state[k] = v
        if prevs and not self.a.dry_run:
            self.state_path.write_text(json.dumps(self.state, indent=1))

    # ---------------------------------------------------------------- stages
    def check(self):
        missing = []
        for mod, pkg in PIP.items():
            try:
                __import__(mod)
            except ImportError:
                missing.append(pkg)
        net = all(_reachable(h) for h in ("pypi.org", "huggingface.co"))
        if not net:
            raise SystemExit("No Internet (pypi.org / huggingface.co unreachable). On Kaggle: Settings -> Internet -> On "
                             "(needs a phone-verified account).")
        if missing:
            self.run(sys.executable, "-m", "pip", "install", "-q", *missing)
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self.say("torch", torch.__version__, "| GPU:", gpu, "| work:", self.work)
        if not gpu and not self.a.allow_cpu:
            raise SystemExit("No GPU. On Kaggle: Settings -> Accelerator -> GPU (and check the weekly quota). "
                             "--allow-cpu runs anyway (hours per stage).")
        return dict(gpu=gpu, torch=torch.__version__)

    def _gather_latents(self):
        lat = self.work / "latents"
        lat.mkdir(parents=True, exist_ok=True)
        for r in self.inputs():  # reuse latents from attached inputs (e.g. an earlier s1_cache output)
            for f in r.rglob("latents/*.npy"):
                if not (lat / f.name).exists() and not self.a.dry_run:
                    shutil.copy(f, lat / f.name)
        return len(list(lat.glob("DIV2K_train_HR__*.npy")))

    def data(self):
        have = self._gather_latents()
        if have >= 800:
            return dict(div2k_latents=have, downloaded=False)
        self.py("experiments/s1/cache_latents.py", "--download-div2k", "--download-only", "--data", self.work / "data")
        return dict(div2k_latents=have, downloaded=True)

    def cache(self):
        lat = self.work / "latents"
        self._gather_latents()
        dirs = [d for d in [self.work / "data" / "DIV2K_train_HR", *map(Path, self.a.extra_images)] if d.exists()]
        if dirs:
            self.py("experiments/s1/cache_latents.py", "--images", *dirs, "--out", lat, "--dtype", "fp16",
                    *(["--allow-cpu"] if self.a.allow_cpu else []))
        n = len(list(lat.glob("*.npy")))
        self.say("latents:", n)
        if n < MIN_TRAIN_LATENTS and not self.a.dry_run:
            raise SystemExit(f"only {n} latents cached; need >= {MIN_TRAIN_LATENTS}")
        if not self.a.keep_images and not self.a.dry_run:
            shutil.rmtree(self.work / "data", ignore_errors=True)  # keep Kaggle outputs small
        return dict(latents=n)

    def gonogo1(self):
        g = self.work / "gonogo1"
        q = "experiments/quant_noise"
        self.py(f"{q}/prepare_gonogo_latents.py", "--from-cache", self.work / "latents", "--out", g / "div2k")
        self.py(f"{q}/prepare_gonogo_latents.py", "--kodak", "--data", g / "img", "--out", g / "kodak")
        self.py(f"{q}/prepare_gonogo_latents.py", "--clic-valid", "--data", g / "img", "--out", g / "clic2020_valid")
        dev = ["--device", "cuda"] if self.state.get("check", {}).get("gpu") else \
              ["--device", "cpu", "--steps", "4000", "--width", "64", "--blocks", "6", "--batch", "32", "--n_draws", "2"]
        self.py(f"{q}/denoiser_gonogo.py", "--train", g / "div2k", "--test", g / "kodak", g / "clic2020_valid",
                "--out", g / "result", *dev)
        shutil.rmtree(g / "img", ignore_errors=True)
        res = json.loads((g / "result" / "results.json").read_text()) if (g / "result" / "results.json").exists() else {}
        return dict(verdicts={k: v.get("verdict", v.get("pass")) for k, v in res.get("tests", {}).items()})

    def s1(self):
        out = self.work / "s1"
        self.py("experiments/s1/train_s1.py", "selftest", "--out", out)
        if not self.a.dry_run and not json.loads((out / "selftest.json").read_text())["ok"]:
            raise SystemExit("S1 selftest FAILED: the bitstream is not bit-exact; see pipeline.log")
        hours = max(0.25, self.hours_left() - self.a.reserve_hours)
        self.py("experiments/s1/train_s1.py", "train", "--latents", self.work / "latents", "--out", out,
                "--hours", f"{hours:.2f}", *self.a.s1_args)
        log = json.loads((out / "log.json").read_text()) if (out / "log.json").exists() else {}
        steps = log.get("log", [{}])[-1].get("step") if log.get("log") else None
        if steps is None or steps < self.a.s1_min_steps:
            self.say(f"S1 stopped at step {steps} (time budget): evaluating this checkpoint; re-run to resume")
            return dict(steps=steps, _incomplete=True)
        return dict(steps=steps)

    def s1_eval(self):
        self.py("experiments/s1/eval_s1.py", "--ckpt", self.work / "s1" / "last.pt", "--dataset", "kodak",
                "--data", self.work / "eval_img", "--out", self.work / "s1_eval")
        shutil.rmtree(self.work / "eval_img", ignore_errors=True)
        return {}

    def ceiling(self):
        self.py("experiments/ceiling/ceiling.py", "all", "--data", self.work / "ceiling_data",
                "--out", self.work / "ceiling")
        shutil.rmtree(self.work / "ceiling_data", ignore_errors=True)
        return {}

    def parity(self):
        self.run(sys.executable, "-m", "pip", "install", "-q", "diffusers", "accelerate")
        self.py("experiments/parity/parity.py", "all", "--out", self.work / "parity")
        return {}

    def qhat(self):
        self.py("experiments/qhat/qhat.py", "all", "--out", self.work / "qhat", *shlex.split(self.a.qhat_args),
                *(["--allow-cpu"] if self.a.allow_cpu else []))
        fit = json.loads((self.work / "qhat" / "fit.json").read_text()) if not self.a.dry_run else {}
        return dict(cross_2024=fit.get("cross", {}).get("acc"))

    def _find(self, *patterns):
        """First file matching one of the patterns under the work dir or the inputs (in pattern order)."""
        for pat in patterns:
            for r in [self.work] + self.inputs():
                hits = sorted(r.rglob(pat))
                if hits:
                    return hits[0]
        return None

    def _s1_ckpt_for_clic(self):
        """S1 checkpoint that reaches the CLIC rates: the one whose sibling log.json has the largest lambda
        (run 1, lambdas 0.03..4, covers 0.02..0.12 bpp; run 2 stops at 0.067 bpp). Any directory layout."""
        best = None
        for r in [self.work] + self.inputs():
            for ck in r.rglob("last.pt"):
                log = ck.parent / "log.json"
                try:
                    lam = max(json.loads(log.read_text())["lambdas"]) if log.exists() else 0.0
                except (ValueError, KeyError):
                    lam = 0.0
                if best is None or lam > best[0]:
                    best = (lam, ck)
        if best and best[0] < 2:
            self.say(f"WARNING: best S1 checkpoint {best[1]} has max lambda {best[0]}: it will not reach 0.075 bpp")
        return best[1] if best else None

    def build_vtm(self):
        vtm = self.work / "vtm"
        if not list(vtm.glob("bin/**/EncoderApp*")):
            self.run("bash", "-c", "set -e; cd '%s'; [ -d vtm ] || git clone -q --depth 1 --branch VTM-23.8 "
                     "https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm; cd vtm; mkdir -p build; cd build; "
                     "cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null; make -j$(nproc) EncoderApp 2>&1 | tail -2" % self.work)
        return vtm

    def bakeoff(self):
        args = shlex.split(self.a.bakeoff_args)
        cands = []
        for v in (args[args.index("--cands") + 1:] if "--cands" in args else []):
            if v.startswith("--"):
                break
            cands.append(v)
        if not cands or {"vtm420", "vtmscc", "s1res"} & set(cands):
            args += ["--vtm", self.build_vtm()]
        ckpt = self.a.s1_ckpt or self._s1_ckpt_for_clic()
        qhat = self._find("qhat/qhat_v0.json", "qhat_v0.json")
        self.say("bakeoff: S1 checkpoint", ckpt, "| Q-hat", qhat)
        self.py("experiments/bakeoff/bakeoff.py", self.a.bakeoff_step, "--out", self.work / "bakeoff",
                *(["--s1-ckpt", ckpt] if ckpt else []), *(["--qhat", qhat] if qhat else []), *args)
        info = dict(s1_ckpt=str(ckpt), qhat=str(qhat), step=self.a.bakeoff_step, cands=cands or "all")
        if self.a.bakeoff_step != "all" or cands:  # a partial run (one step / some candidates) is not "done"
            info["_incomplete"] = True
        return info

    def pack(self):
        z = self.work / "results.zip"
        keep = [p for p in self.work.rglob("*") if p.is_file() and (
            p.suffix in (".md", ".json", ".csv", ".log", ".jpg") or (p.name == "last.pt" and self.a.pack_ckpt))
            and not {"latents", "data", "vtm", "recon", "valid"} & set(p.parts) and p.name != "results.zip"]
        if not self.a.dry_run:
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
                for p in keep:
                    f.write(p, p.relative_to(self.work))
            if KAGGLE:
                shutil.copy(z, "/kaggle/working/results.zip")
        self.say("packed", len(keep), "files ->", z)
        return dict(files=len(keep))

    # ---------------------------------------------------------------- driver
    def main(self):
        self.say(f"ratflow pipeline @ {self.commit} | stages {self.a.stages} | budget {self.a.hours} h")
        self.restore()
        order = []
        for s in self.a.stages:
            for dep in NEEDS.get(s, []) + [s]:
                if dep not in order:
                    order.append(dep)
        order = [s for s in ALL_STAGES if s in order]
        for s in order:
            if s not in ("check", "pack") and self.done(s) and s not in self.a.redo:
                self.say(f"== {s}: already done, skipping")
                continue
            if s == "s1_eval" and not self.a.dry_run and not (self.work / "s1" / "last.pt").exists():
                self.say("== s1_eval: no S1 checkpoint yet, skipping")
                continue
            if self.hours_left() < 0.1 and s != "pack":
                self.say(f"== {s}: out of time budget, stopping (re-run to continue)")
                break
            self.say(f"== {s}")
            self._stage_t0 = time.time()
            try:
                info = getattr(self, s)() or {}
            except StageTimeout:
                self.state[s] = dict(done=False, timed_out=True, seconds=round(time.time() - self._stage_t0),
                                     commit=self.commit)
                self.state_path.write_text(json.dumps(self.state, indent=1))
                self.say(f"== {s}: stopped by the time budget; partial results kept. Re-run with this output "
                         "attached as Input to continue.")
                break
            except (Exception, SystemExit) as e:  # noqa: BLE001  record, pack what exists, re-raise
                self.state[s] = dict(done=False, error=str(e)[-2000:], commit=self.commit)
                self.state_path.write_text(json.dumps(self.state, indent=1))
                self.say(f"== {s}: FAILED: {e}")
                self._stage_t0 = time.time()
                self.pack()
                raise
            if info.pop("_incomplete", False):
                self.state[s] = dict(done=False, seconds=round(time.time() - self._stage_t0), commit=self.commit, **info)
                self.state_path.write_text(json.dumps(self.state, indent=1))
                self.say(f"== {s}: incomplete", info)
                continue
            self.mark(s, **info)
            self.say(f"== {s}: done", info)
        if "pack" not in order:
            self._stage_t0 = time.time()
            self.pack()


def _reachable(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def _git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"subset of {ALL_STAGES} (default: {DEFAULT_STAGES})")
    ap.add_argument("--work", default="/kaggle/working/work" if KAGGLE else "work")
    ap.add_argument("--hours", type=float, default=11.0 if KAGGLE else 24.0,
                    help="total wall-clock budget; running stages are stopped when it runs out (Kaggle kills at 12 h)")
    ap.add_argument("--reserve-hours", type=float, default=0.75, help="kept for s1_eval + pack after training")
    ap.add_argument("--inputs", nargs="*", default=[], help="dirs searched for a previous work dir / latents")
    ap.add_argument("--extra-images", nargs="*", default=[], help="more training image dirs (Flickr2K, LSDIR)")
    ap.add_argument("--s1-args", nargs=argparse.REMAINDER, default=[], help="passed to train_s1.py train (put last)")
    ap.add_argument("--qhat-args", default="", help='one string passed to qhat.py, e.g. --qhat-args="--n 2024t=8000"')
    ap.add_argument("--bakeoff-step", default="all", choices=["prepare", "points", "metrics", "allocate", "summary", "all"],
                    help="e.g. 'points' with --bakeoff-args='--cands vtm420' in a CPU-only session (no GPU quota)")
    ap.add_argument("--bakeoff-args", default="", help='one string passed to bakeoff.py, e.g. --bakeoff-args="--cands msillm mbt"')
    ap.add_argument("--s1-ckpt", default=None, help="S1 checkpoint for bakeoff (default: s1_out/last.pt in the inputs)")
    ap.add_argument("--s1-min-steps", type=int, default=150_000, help="S1 counts as done only after this many steps")
    ap.add_argument("--redo", nargs="*", default=[], help="stages to run again even if done")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--pack-ckpt", action="store_true", default=True)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)
    a.stages = a.stages or DEFAULT_STAGES
    bad = [x for x in a.stages + a.redo if x not in ALL_STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}; choose from {ALL_STAGES}")
    return a


if __name__ == "__main__":
    Pipeline(parse()).main()
