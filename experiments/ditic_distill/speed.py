"""Decode speed of the teacher vs the student DC-AE decoder on the CLIC validation latents (from prep.py), bf16,
NCHW RMSNorm, full frame, latent padded to a multiple of 16 (the bake-off default). Each decoder in a fresh pass that
decodes every image once (as CLIC counts), then the same pass again (warm). Optionally loads a trained student.

    python speed.py --q q3 --lat ~/clic/distill/lat_q3/clic_val [--ckpt ~/clic/distill/run_q3/best.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STUDENT_CHANNELS, build_student, load_ditic, load_student_weights, patch_resblocks, res_forward_fast  # noqa: E402,E501

BF16 = torch.bfloat16


def dec_forward(self, z):
    from common import rms_scale
    h = self.conv_in(z) + z.repeat_interleave(self.in_shortcut_repeats, dim=1,
                                              output_size=z.shape[1] * self.in_shortcut_repeats) \
        if self.in_shortcut else self.conv_in(z)
    for up in reversed(self.up_blocks):
        h = up(h)
    n = self.norm_out
    h = h * rms_scale(h, n)
    h = torch.addcmul(n.bias.view(1, -1, 1, 1), h, n.weight.view(1, -1, 1, 1))
    return self.conv_out(self.conv_act(h))


def run(dec, lats, label):
    for rnd in ("cold", "warm"):
        ts = []
        for z in lats:
            z = F.pad(z, (0, (-z.shape[3]) % 16, 0, (-z.shape[2]) % 16), mode="replicate")
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                dec(z)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        print(f"{label:8s} {rnd}: first {ts[0]:.2f} s, images 2..{len(ts)} mean {np.mean(ts[1:]):.3f} s, "
              f"total {sum(ts):.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q3")
    ap.add_argument("--lat", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--channels", default=",".join(map(str, STUDENT_CHANNELS)))
    a = ap.parse_args()
    net = load_ditic(a.q)
    teacher = net.vae.decoder
    del net.DiT, net.aux_codec, net.vae.encoder
    teacher.to(BF16).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False) if a.ckpt else {}
    student = build_student(teacher, ck.get("channels", tuple(int(c) for c in a.channels.split(","))))
    if ck:
        load_student_weights(student, ck)
    student = student.to("cuda").to(BF16).eval()
    import types
    for d in (teacher, student):
        patch_resblocks(d, res_forward_fast)
        d.forward = types.MethodType(dec_forward, d)
    lats = [torch.load(p)["z"].cuda().to(BF16) for p in sorted(a.lat.glob("*.pt"))]
    print(f"{len(lats)} latents; teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.0f} M params, "
          f"student {sum(p.numel() for p in student.parameters()) / 1e6:.0f} M", flush=True)
    for d, label in ((teacher, "teacher"), (student, "student"), (teacher, "teacher")):
        torch.cuda.reset_peak_memory_stats()
        run(d, lats, label)


if __name__ == "__main__":
    main()
