"""DiT-IC checks for a rented L4 (2026-10-13): is entropy decoding bit-exact across machines, and how fast is the
decoder there. Same code runs on our L40 server, so bitstreams can be carried between machines.

    python l4_check.py encode --q q3 --out streams_<host>.pt     # 30 validation images; records what a decoder must see
    python l4_check.py decode --streams streams_<other host>.pt  # decode here, count index / symbol mismatches
    python l4_check.py speed --decoder teacher|student [--students DIR]   # CLIC-style: fresh process, each image once
    python l4_check.py compare streams_A.pt streams_B.pt         # no GPU: same bytes, same indexes / symbols?

Settings are fixed on both sides: codec fp32 with cuDNN deterministic, no TF32; DiT and DC-AE decoder bf16 with the
NCHW RMSNorm, full frame, latent cut to the image and padded to a multiple of 16 latents (the bake-off defaults).
Needs DITIC_DIR (DiT-IC checkout) and experiments/ditic_distill/common.py.
"""
import argparse
import json
import math
import os
import platform
import resource
import socket
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ditic_distill"))
from common import (build_student, load_ditic, load_student_weights, patch_resblocks, res_forward_fast,  # noqa: E402
                    rms_scale)

BF16 = torch.bfloat16


def fix_settings():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def env_info():
    return dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=platform.python_version(),
                driver=_driver())


def _driver():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()[0]  # one line per GPU
    except Exception:  # noqa: BLE001
        return "?"


def dec_forward(self, z):
    h = self.conv_in(z) + z.repeat_interleave(self.in_shortcut_repeats, dim=1,
                                              output_size=z.shape[1] * self.in_shortcut_repeats) \
        if self.in_shortcut else self.conv_in(z)
    for up in reversed(self.up_blocks):
        h = up(h)
    n = self.norm_out
    h = h * rms_scale(h, n)
    h = torch.addcmul(n.bias.view(1, -1, 1, 1), h, n.weight.view(1, -1, 1, 1))
    return self.conv_out(self.conv_act(h))


def load(q, student_dir=None, encoder=False):
    """DiT-IC checkpoint q, decoder as it would ship; with student_dir, its distilled DC-AE decoder."""
    net = load_ditic(q)
    if not encoder:
        del net.aux_codec, net.vae.encoder
    if student_dir:
        ck = torch.load(Path(student_dir) / f"{q}.pt", map_location="cpu", weights_only=False)
        st = build_student(net.vae.decoder, ck["channels"])
        load_student_weights(st, ck)
        net.vae.decoder = st.to("cuda").eval()
    for m in (net.DiT, net.prompter, net.vae.decoder):
        m.to(BF16)
    patch_resblocks(net.vae.decoder, res_forward_fast)
    net.vae.decoder.forward = types.MethodType(dec_forward, net.vae.decoder)
    return net


def recorder(codec):
    """Keep each checkerboard group's scale-table indexes and decoded symbols (what must match across machines)."""
    log = []
    orig = type(codec).decompress_group_with_mask

    def wrapped(self, gc, scales, means, mask, decoder, cdf, cdf_lengths, offsets):
        out = orig(self, gc, scales, means, mask, decoder, cdf, cdf_lengths, offsets)
        idx = gc.build_indexes(self.sequeeze_with_mask(scales, mask))
        sym = self.sequeeze_with_mask(out, mask) - self.sequeeze_with_mask(means, mask)
        log.append((idx.to(torch.uint8).cpu().numpy().ravel(), sym.round().to(torch.int16).cpu().numpy().ravel()))
        return out

    codec.decompress_group_with_mask = types.MethodType(wrapped, codec)
    return log


@torch.no_grad()
def decompress(net, strings, shape, H, W, bucket=16):
    mean, logs, res, prompt = net.codec.decompress(strings, shape)
    scale = torch.exp(0.5 * torch.clamp(logs, -30.0, 20.0))
    enc = net.prompter(prompt.to(BF16))
    ts = net.sched.base_scheduler.timesteps.expand(1) * net.DiT.config.timestep_scale
    pred = net.DiT(mean.to(BF16), encoder_hidden_states=enc, timestep=ts.to(mean.device), return_dict=False)[0]
    z = (net.sched.step(pred.float(), scale, net.timesteps, mean, return_dict=True) + res) / net.vae.config.scaling_factor
    z = z[:, :, :min(z.shape[2], math.ceil(H / 32) + 2), :min(z.shape[3], math.ceil(W / 32) + 2)]
    z = F.pad(z, (0, (-z.shape[3]) % bucket, 0, (-z.shape[2]) % bucket), mode="replicate")
    net.vae.disable_tiling()
    y = net.vae.decode(z.to(BF16), return_dict=False)[0]
    return y[0, :, :H, :W].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def psnr(x, y):
    m = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * math.log10(255 ** 2 / m)


def images(valid):
    return sorted(p for p in Path(valid).iterdir() if p.suffix.lower() == ".png")


def cmd_encode(a):
    net = load(a.q, encoder=True)
    rec = recorder(net.codec)
    out = dict(env=env_info(), q=a.q, images={})
    for p in images(a.valid):
        x = np.asarray(Image.open(p).convert("RGB"))
        H, W = x.shape[:2]
        t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
        t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
        net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,  # the fp32 encoder: tiles
                              tile_sample_stride_height=896, tile_sample_stride_width=896)
        with torch.no_grad():
            o = net.compress(t)
        rec.clear()
        y = decompress(net, o["strings"], o["shape"], H, W)  # what this machine's decoder sees
        out["images"][p.stem[:8]] = dict(strings=o["strings"], shape=tuple(o["shape"]), H=H, W=W,
                                         idx=[g[0] for g in rec], sym=[g[1] for g in rec], psnr=psnr(x, y))
        print(f"{p.stem[:8]} {sum(len(s[0]) for s in o['strings'])} B, local decode PSNR {psnr(x, y):.2f}", flush=True)
    torch.save(out, a.out)
    print(f"saved {a.out} from {json.dumps(out['env'])}", flush=True)


def cmd_decode(a):
    src = torch.load(a.streams, weights_only=False)
    net = load(src["q"])
    rec = recorder(net.codec)
    here = env_info()
    print(f"streams encoded on {json.dumps(src['env'])}\ndecoding on       {json.dumps(here)}", flush=True)
    bad_imgs, rows = 0, []
    for name, r in src["images"].items():
        rec.clear()
        y = decompress(net, r["strings"], r["shape"], r["H"], r["W"])
        idx_bad = sum(int((g[0] != e).sum()) for g, e in zip(rec, r["idx"]))
        sym_bad = sum(int((g[1] != e).sum()) for g, e in zip(rec, r["sym"]))
        x = np.asarray(Image.open(next(Path(a.valid).glob(f"{name}*.png"))).convert("RGB"))
        p = psnr(x, y)
        bad_imgs += bool(idx_bad or sym_bad)
        rows.append(dict(name=name, idx_bad=idx_bad, sym_bad=sym_bad, psnr_here=p, psnr_encoder_side=r["psnr"]))
        print(f"{name}: index mismatches {idx_bad}, symbol mismatches {sym_bad}, PSNR here {p:.2f} "
              f"(encoder side {r['psnr']:.2f})", flush=True)
    verdict = "BIT-EXACT" if not bad_imgs else f"MISMATCH on {bad_imgs}/{len(rows)} images"
    print(f"== {verdict}: {src['env']['gpu']} ({src['env']['host']}) -> {here['gpu']} ({here['host']})", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(encoder=src["env"], decoder=here, verdict=verdict, images=rows), indent=1))


def cmd_speed(a):
    """One fresh pass over the 30 images, each decoded once, as CLIC times it (model load reported apart: a submission
    would ship compact bf16 weights instead of building the fp32 net from the 4.3 GB checkpoint)."""
    src = torch.load(a.streams, weights_only=False)
    torch.cuda.synchronize(); t0 = time.time()
    net = load(src["q"], a.students if a.decoder == "student" else None)
    torch.cuda.synchronize(); t_load = time.time() - t0
    ts = []
    for name, r in src["images"].items():
        torch.cuda.synchronize(); t0 = time.time()
        decompress(net, r["strings"], r["shape"], r["H"], r["W"])
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    res = dict(env=env_info(), decoder=a.decoder, q=src["q"], tag=a.tag, load_s=t_load, first_s=ts[0],
               mean_rest_s=float(np.mean(ts[1:])), total_s=float(sum(ts)), n=len(ts),
               peak_gb=torch.cuda.max_memory_allocated() / 2**30, cpus=len(os.sched_getaffinity(0)),
               torch_threads=torch.get_num_threads(), peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    print(f"== {a.decoder} on {res['env']['gpu']}{' ' + a.tag if a.tag else ''}: load {t_load:.1f} s | first image "
          f"{ts[0]:.2f} s | images 2..{len(ts)} mean {res['mean_rest_s']:.3f} s | decode total {res['total_s']:.1f} s | "
          f"peak {res['peak_gb']:.1f} GB | {res['cpus']} CPUs, RAM peak {res['peak_rss_gb']:.1f} GB", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def cmd_compare(a):
    """Two streams files, e.g. from two L4s, offline. Same bitstreams and same recorded indexes / symbols means a decoder
    on either machine reads the other's streams exactly (each machine's self check already shows its decode repeats);
    different bitstreams need a real cross decode (`decode --streams` on the other machine)."""
    A, B = (torch.load(p, weights_only=False, map_location="cpu") for p in (a.a, a.b))
    print(f"A: {json.dumps(A['env'])}\nB: {json.dumps(B['env'])}")
    names = [n for n in A["images"] if n in B["images"]]
    n_bytes = n_rec = 0
    for n in names:
        ra, rb = A["images"][n], B["images"][n]
        same_bytes = ra["strings"] == rb["strings"]
        same_rec = len(ra["idx"]) == len(rb["idx"]) and all(
            x.shape == y.shape and (x == y).all() for x, y in zip(ra["idx"] + ra["sym"], rb["idx"] + rb["sym"]))
        n_bytes += same_bytes
        n_rec += same_rec
        print(f"{n}: bitstream {'same' if same_bytes else 'DIFFERENT'}, decoded indexes/symbols "
              f"{'same' if same_rec else 'DIFFERENT'}, PSNR {ra['psnr']:.2f} / {rb['psnr']:.2f}")
    print(f"== {A['env']['gpu']} ({A['env']['host']}) vs {B['env']['gpu']} ({B['env']['host']}): "
          f"bitstreams same on {n_bytes}/{len(names)}, decoded indexes/symbols same on {n_rec}/{len(names)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--q", default="q3")
    e.add_argument("--out", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--streams", required=True)
    d.add_argument("--json", default=None)
    s = sub.add_parser("speed")
    s.add_argument("--streams", required=True)
    s.add_argument("--decoder", choices=["teacher", "student"], default="teacher")
    s.add_argument("--students", default=None)
    s.add_argument("--json", default=None)
    s.add_argument("--tag", default="", help="label for the result line, e.g. the emulation setting")
    for p in (e, d, s):
        p.add_argument("--valid", required=True, help="dir with the 30 CLIC validation PNGs")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    fix_settings()
    dict(encode=cmd_encode, decode=cmd_decode, speed=cmd_speed, compare=cmd_compare)[a.cmd](a)


if __name__ == "__main__":
    main()
