"""Training stages for GCodec.

    python -m clic27.gc.train --config CFG --work DIR --stage rd|gan|rate|align [--steps N] [--init CKPT]
    torchrun --nproc_per_node 8 -m clic27.gc.train ...      # multi-GPU (DDP)

Each stage writes DIR/<stage>/{last.pt, model.pt, log.jsonl, eval.jsonl}. last.pt holds model + discriminator +
optimisers + step and is resumed automatically. A stage without --init starts from the previous stage's
model.pt (rd -> gan -> rate -> align) when it exists.

model.pt is written only when the stage reached its step count. SIGTERM / SIGINT / SIGUSR1 (preemption) and the
--max-minutes budget both checkpoint last.pt, write DIR/<stage>/partial.json and exit with code 75 (EXIT_PARTIAL);
running the same command again resumes. last.pt is also refreshed every train.ckpt_minutes, so a hard kill loses
at most that much. Under DDP all stop decisions are agreed by all ranks (all_reduce) before anyone leaves the loop.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import config as C
from .data import CropDataset, list_images, random_cells
from .disc import MultiScaleD, d_loss, g_loss
from .losses import LPIPS, WassersteinLoss, psnr_t, rate_lambda
from .model import GCodec, count_params, pad64

ORDER = ["rd", "gan", "rate", "align"]
EXIT_PARTIAL = 75  # EX_TEMPFAIL: stage stopped before its step count (signal / time budget); rerun to resume


class _Stop:
    """Set by the signal handler; the training loop checkpoints and leaves at the next step boundary."""
    signum = 0


def _install_stop_handlers():
    def handler(signum, _frame):
        if not _Stop.signum:
            print(f"[train] signal {signum}: checkpoint at the next step, then exit {EXIT_PARTIAL}", flush=True)
        _Stop.signum = signum
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def _ignore_stop_signals():
    """DataLoader workers: a SIGTERM to the whole process group must not kill them before the trainer has saved
    last.pt (it would crash on the next batch instead); they exit with the trainer."""
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def unwrap(m):
    return m.module if hasattr(m, "module") else m


def build(cfg):
    model = GCodec(cfg["model"])
    model.check_ranges()
    return model


def save_model(model, path: Path, cfg, extra=None):
    m = unwrap(model)
    m.update_tables()
    torch.save(dict(cfg=cfg, state_dict=m.state_dict(), **(extra or {})), path)


def load_model(path, dev="cpu", cfg=None):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = GCodec((cfg or sd["cfg"])["model"])
    m.load_state_dict(sd["state_dict"])
    return m.to(dev).eval(), sd["cfg"]


@torch.no_grad()
def quick_eval(model, imgs, cfg, dev, lp, levels, beta_val=1.0):
    """Estimated-rate eval (no entropy coding) on full validation images, zero quality map, beta = 1."""
    m = unwrap(model).eval()
    rows = []
    for p in imgs:
        from PIL import Image
        a = np.array(Image.open(p).convert("RGB"))
        x = torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255).to(dev)
        xp, (H, W) = pad64(x)
        ch, cw = m.cell_shape(*xp.shape[-2:])
        for l in levels:
            lv = torch.tensor([l], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else nullcontext():
                out = m(xp, lv, torch.full((1,), beta_val, device=dev), torch.zeros(1, 1, ch, cw, dtype=torch.long, device=dev),
                        noise=False)
            xh = out["x_hat"][..., :H, :W].float().clamp(0, 1)
            rows.append(dict(image=Path(p).stem, level=l, bpp=float(out["bits"]) / (H * W),
                             psnr=float(psnr_t(xh, x)), lpips=float(lp(xh, x)) if lp else float("nan")))
    m.train()
    agg = {}
    for l in levels:
        rs = [r for r in rows if r["level"] == l]
        agg[l] = {k: float(np.mean([r[k] for r in rs])) for k in ("bpp", "psnr", "lpips")}
    return agg


def run_stage(cfg, work: Path, stage: str, steps=None, init=None, max_minutes=None):
    st = dict(cfg["stages"][stage])
    steps = steps or st["steps"]
    T = cfg["train"]
    ddp = "LOCAL_RANK" in os.environ
    _install_stop_handlers()
    if ddp and not dist.is_initialized():
        n_gpu = torch.cuda.device_count()
        world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
        # NCCL needs one GPU per rank; with fewer GPUs than ranks (a DDP test on one GPU) fall back to gloo.
        backend = os.environ.get("GC_DDP_BACKEND") or ("nccl" if n_gpu >= world else "gloo")
        dist.init_process_group(backend)
        if n_gpu:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % n_gpu)
        log(f"[{stage}] DDP: {dist.get_world_size()} ranks, backend {backend}, {n_gpu} visible GPU(s)")
    dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    seed = T["seed"] + (dist.get_rank() if ddp else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = work / stage
    out.mkdir(parents=True, exist_ok=True)
    model = build(cfg).to(dev)
    disc = MultiScaleD(cfg["model"]["M"]).to(dev) if st.get("gan") else None
    last = out / "last.pt"
    step = 0
    state = None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if disc is not None and state.get("disc"):
            disc.load_state_dict(state["disc"])
        step = state["step"]
        log(f"[{stage}] resume from step {step}")
    else:
        src = Path(init) if init else None
        if src is not None and not src.is_file():
            raise SystemExit(f"[{stage}] --init {src} does not exist")
        if src is None:
            i = ORDER.index(stage)
            for prev in reversed(ORDER[:i]):
                if (work / prev / "model.pt").exists():
                    src = work / prev / "model.pt"
                    break
                if (work / prev / "last.pt").exists():  # started but never finished: never skip over it silently
                    raise SystemExit(f"[{stage}] previous stage {prev} is unfinished (last.pt, no model.pt); "
                                     f"rerun {prev} to completion or pass --init explicitly")
        if src is not None:
            sd = torch.load(src, map_location="cpu", weights_only=False)
            if sd.get("complete") is False or (work / sd.get("stage", "?") / "partial.json").exists() and init is None:
                raise SystemExit(f"[{stage}] {src} comes from an unfinished stage; finish it (rerun) before {stage}")
            if "complete" not in sd:
                log(f"[{stage}] note: {src} predates completion tracking; make sure that stage reached its steps")
            strict = T.get("init_strict", True)
            missing, unexpected = model.load_state_dict(sd["state_dict"], strict=strict)
            log(f"[{stage}] init from {src} (stage {sd.get('stage')}, step {sd.get('step')}, strict={strict}, "
                f"missing {len(missing)}, unexpected {len(unexpected)})")
            if len(missing) > 0.5 * len(model.state_dict()):
                raise SystemExit(f"[{stage}] --init {src}: {len(missing)} missing tensors, not a GCodec of this shape")
            if disc is not None and (src.parent / "disc.pt").exists():
                disc.load_state_dict(torch.load(src.parent / "disc.pt", map_location="cpu"))
    if step >= steps:
        log(f"[{stage}] already done ({step} >= {steps})")
        if is_main() and not (out / "model.pt").exists():  # stopped between the last checkpoint and the export
            if state is not None and state.get("ema"):
                model.load_state_dict(state["ema"])
            save_model(model, out / "model.pt", cfg, dict(stage=stage, step=step, ema=bool(state and state.get("ema")),
                                                          complete=True))
        if ddp:
            dist.barrier()
        return out / "model.pt"
    log(f"[{stage}] params: model {count_params(model) / 1e6:.1f}M, g_s {count_params(model.g_s) / 1e6:.1f}M"
        + (f", disc {count_params(disc) / 1e6:.1f}M" if disc else ""))

    opt = torch.optim.Adam(model.parameters(), lr=st["lr"])
    opt_d = torch.optim.Adam(disc.parameters(), lr=st.get("lr_d", st["lr"]), betas=(0.5, 0.999)) if disc else None
    if state is not None:
        opt.load_state_dict(state["opt"])
        if opt_d is not None and state.get("opt_d"):
            opt_d.load_state_dict(state["opt_d"])
    ema = None
    if T.get("ema", 0) > 0:  # exponential moving average of the codec weights; exported as model.pt
        ema = copy.deepcopy(model).eval()
        ema.requires_grad_(False)
        if state is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev], find_unused_parameters=True)
        if disc is not None:
            disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[dev])

    D = cfg["data"]
    # the dataset samples a random image + crop per item, so no shuffling/sampler is needed (each worker and
    # each DDP rank has its own RNG stream); length only bounds one pass of the loader
    ds = CropDataset(D["train_dirs"], D["screen_dirs"], D["screen_prob"], D["crop"], D["min_side"], D["downscale"],
                     length=max(1000, min(steps, 200000) * st["batch"]))
    dl = torch.utils.data.DataLoader(ds, batch_size=st["batch"], shuffle=False, num_workers=D["workers"],
                                     pin_memory=True, drop_last=True, persistent_workers=D["workers"] > 0,
                                     worker_init_fn=lambda w: (_ignore_stop_signals(),
                                                               random.seed(seed * 1000 + w + step),
                                                               np.random.seed((seed * 1000 + w + step) % 2 ** 31)))
    Lw = cfg["loss"]
    lp = LPIPS().to(dev) if st.get("lpips") and Lw["k_lpips"] > 0 else None
    wd = WassersteinLoss().to(dev) if Lw["k_wd"] > 0 else None
    rw = None
    if st.get("reward") and Lw["k_reward"] > 0:
        from .reward import load_reward
        rw = load_reward(cfg["reward_path"], dev)
        for p in rw.parameters():
            p.requires_grad_(False)
    R = cfg["rate"]
    levels = st.get("levels") or list(range(cfg["model"]["n_levels"]))
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(T["amp"])
    actx = (lambda: torch.autocast("cuda", dtype=amp)) if (amp and dev.startswith("cuda")) else nullcontext
    valid = list_images([D["valid_dir"]])[: T["eval_images"]] if D["valid_dir"] else []
    m0 = unwrap(model)
    qb, Rq = m0.qmap_block, m0.qmap_range
    logf = open(out / "log.jsonl", "a") if is_main() else None
    def save_last():
        tmp = last.with_suffix(".tmp")
        torch.save(dict(model=unwrap(model).state_dict(), disc=unwrap(disc).state_dict() if disc else None,
                        ema=ema.state_dict() if ema is not None else None,
                        opt=opt.state_dict(), opt_d=opt_d.state_dict() if opt_d else None, step=step), tmp)
        tmp.replace(last)  # atomic: a crash while saving never leaves a broken last.pt

    t0, tl = time.time(), time.time()
    tck = time.time()  # last checkpoint (time-based checkpoints bound the work lost to a hard kill)
    stop_reason = None
    model.train()
    bad = 0
    it = iter(dl)
    while step < steps:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.to(dev, non_blocking=True)
        B, _, Hc, Wc = x.shape
        level = torch.tensor([random.choice(levels) for _ in range(B)], device=dev)
        beta = (torch.rand(B, device=dev) * Lw["beta_max"]) if disc is not None else torch.zeros(B, device=dev)
        cells = random_cells(B, -(-Hc // qb), -(-Wc // qb), Rq).to(dev)
        lr_scale = 0.1 if step > 0.9 * steps else 1.0
        for g in opt.param_groups:
            g["lr"] = st["lr"] * lr_scale

        if getattr(unwrap(model), "k_cca", 0) > 0:  # CCA: ramp the reversed gradient in over cca_warmup steps
            unwrap(model).cca_ramp = min(1.0, step / max(1, T.get("cca_warmup", 5000)))
        with actx():
            o = model(x, level, beta, cells)
        xh = o["x_hat"].float()
        bpp = o["bits"] / (Hc * Wc)
        if R.get("lambdas"):  # fixed rate weight per level (step D): no collapse of levels whose target is unreachable
            lam = torch.tensor(R["lambdas"], device=dev, dtype=bpp.dtype)[level]
        else:
            lam = rate_lambda(bpp.detach(), level, R["targets"], R["lambda_hi"], R["lambda_lo"])
        dist_t = Lw["k_mse"] * ((xh - x) ** 2).flatten(1).mean(1)
        parts = dict(mse=float(((xh - x) ** 2).mean()))
        if lp is not None:
            v = lp(xh, x)
            dist_t = dist_t + Lw["k_lpips"] * v
            parts["lpips"] = float(v.mean())
        if wd is not None:
            v = wd(xh, x)
            dist_t = dist_t + Lw["k_wd"] * v
            parts["wd"] = float(v.mean())
        if rw is not None:
            v = rw(x, xh)
            dist_t = dist_t + Lw["k_reward"] * v
            parts["reward"] = float(v.mean())
        loss = (lam * bpp).mean() + dist_t.mean()
        if "bits_cca" in o:  # CCA aux rate: minimised by the aux nets, reversed (x k_cca) into the latents
            v = o["bits_cca"] / (Hc * Wc)
            loss = loss + (lam * v).mean()
            parts["cca"] = float(v.mean())
        if disc is not None:
            d0 = unwrap(disc)  # no DDP hooks for the generator pass; D params frozen here
            d0.requires_grad_(False)
            with actx():
                fake = d0(xh, o["y_hat"])
            d0.requires_grad_(True)
            gl = g_loss([f.float() for f in fake], weight=beta)
            loss = loss + Lw["k_gan"] * gl
            parts["g"] = float(gl)
        time_up = bool(max_minutes) and (time.time() - t0) / 60 > max_minutes
        flags = torch.tensor([float(not torch.isfinite(loss)), float(_Stop.signum > 0), float(time_up)], device=dev)
        if ddp:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)  # every rank skips / stops together (no rank left waiting)
        if flags[1].item() > 0 or flags[2].item() > 0:
            stop_reason = "signal" if flags[1].item() > 0 else "max_minutes"
            break
        if flags[0].item() > 0:
            bad += 1
            log(f"[{stage}] step {step}: non-finite loss ({parts}), skipped ({bad} in a row)")
            opt.zero_grad(set_to_none=True)
            if bad > T.get("max_bad_steps", 50):
                raise SystemExit(f"[{stage}] {bad} non-finite steps in a row: lower lr / check data (see docs)")
            continue
        bad = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
        opt.step()
        if ema is not None:
            d = min(T["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), unwrap(model).parameters()):
                    pe.lerp_(p.detach(), 1 - d)
                for be, b in zip(ema.buffers(), unwrap(model).buffers()):
                    be.copy_(b)
        parts["gnorm"] = float(gnorm)
        parts["over"] = float((bpp.detach() > torch.tensor(R["targets"], device=dev)[level]).float().mean())

        if disc is not None:
            yh = o["y_hat"].detach()
            with actx():
                both = disc(torch.cat([x, xh.detach()]), torch.cat([yh, yh]))  # one forward (DDP-safe)
            rl = [b[:B].float() for b in both]
            fl = [b[B:].float() for b in both]
            dl_ = d_loss(rl, fl)
            opt_d.zero_grad(set_to_none=True)
            if torch.isfinite(dl_):
                dl_.backward()
                opt_d.step()
            parts["d"] = float(dl_)
            parts["d_real"] = float(sum(torch.sigmoid(r).mean() for r in rl) / len(rl))  # D's P(real) on real
            parts["d_fake"] = float(sum(torch.sigmoid(f).mean() for f in fl) / len(fl))  # ... on reconstructions

        step += 1
        if is_main() and (step % T["log_every"] == 0 or step == 1):
            rec = dict(step=step, loss=float(loss), bpp=float(bpp.mean()), psnr=float(psnr_t(xh, x).mean()),
                       it_s=T["log_every"] / max(time.time() - tl, 1e-9), **parts)
            tl = time.time()
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"[{stage}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        done = step >= steps
        if is_main() and (step % T["ckpt_every"] == 0 or done
                          or (time.time() - tck) / 60 > T.get("ckpt_minutes", 15)):
            save_last()
            tck = time.time()
        if is_main() and valid and (step % T["eval_every"] == 0 or done):
            agg = quick_eval(ema if ema is not None else model, valid, cfg, dev, lp or LPIPS().to(dev),
                             sorted(set(levels)),
                             beta_val=Lw["beta_max"] if disc is not None else 0.0)
            with open(out / "eval.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, levels=agg)) + "\n")
            log(f"[{stage}] eval step {step}: " + json.dumps(agg))
        if done:
            break
    if stop_reason is not None:  # stopped early: checkpoint, record, and do NOT export model.pt for the next stage
        if is_main():
            save_last()
            (out / "partial.json").write_text(json.dumps(dict(stage=stage, step=step, steps=steps,
                                                              reason=stop_reason, signal=_Stop.signum)))
            log(f"[{stage}] stopped at step {step}/{steps} ({stop_reason}); last.pt saved. "
                f"Rerun the same command to resume. Exit {EXIT_PARTIAL}.")
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        raise SystemExit(EXIT_PARTIAL)
    if is_main():
        (out / "partial.json").unlink(missing_ok=True)
        save_model(ema if ema is not None else model, out / "model.pt", cfg,
                   dict(stage=stage, step=step, ema=ema is not None, complete=True))
        save_model(model, out / "model_raw.pt", cfg, dict(stage=stage, step=step, complete=True))
        if disc is not None:
            torch.save(unwrap(disc).state_dict(), out / "disc.pt")
        log(f"[{stage}] wrote {out / 'model.pt'} at step {step}")
    if ddp:
        dist.barrier()
    return out / "model.pt"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", required=True, choices=ORDER)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--init")
    ap.add_argument("--max-minutes", type=float)
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    run_stage(cfg, Path(a.work), a.stage, a.steps, a.init, a.max_minutes)


if __name__ == "__main__":
    main()
