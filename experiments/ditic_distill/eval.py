"""Teacher vs distilled student on the 30 CLIC validation images: the same DiT-IC latents (same bitstream, prep.py),
decoded full frame with the latent padded to a multiple of 16 (the bake-off default); metrics against the originals
(the bake-off's PSNR / MS-SSIM / LPIPS / DISTS / WD) and Q-hat v0. Writes summary.md, metrics.csv and crop sheets.

    python eval.py --q q3 --lat ~/clic/distill/lat_q3/clic_val --ckpt ~/clic/distill/run_q3/best.pt \
        --valid ~/clic/work_full_gpu/bakeoff/valid --qhat ~/clic/inputs/qhat_v0/qhat/qhat_v0.json --out ~/clic/distill/eval_q3
"""
import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "qhat"))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "bakeoff"))
from common import STUDENT_CHANNELS, build_student, load_ditic, load_student_weights, patch_resblocks, res_forward_fast  # noqa: E402,E501
from speed import dec_forward  # noqa: E402

BF16 = torch.bfloat16


def decode(dec, z, H, W):
    z = F.pad(z, (0, (-z.shape[3]) % 16, 0, (-z.shape[2]) % 16), mode="replicate")
    with torch.no_grad():
        y = dec(z)
    return y[0, :, :H, :W].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q3")
    ap.add_argument("--lat", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--qhat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from bakeoff import qhat_score  # noqa: E402
    from clic27.eval import metrics as M  # noqa: E402
    from qhat import Features  # noqa: E402
    qm = json.loads(a.qhat.read_text())

    net = load_ditic(a.q)
    teacher = net.vae.decoder
    del net.DiT, net.aux_codec, net.vae.encoder
    teacher.to(BF16).eval()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    student = build_student(teacher, ck.get("channels", STUDENT_CHANNELS))
    load_student_weights(student, ck)
    student = student.to("cuda").to(BF16).eval()
    for d in (teacher, student):
        patch_resblocks(d, res_forward_fast)
        d.forward = types.MethodType(dec_forward, d)
    feat = Features(M.device())

    rows, crops = [], []
    for p in sorted(a.lat.glob("*.pt")):
        r = torch.load(p)
        name, H, W = p.stem, r["H"], r["W"]
        z = r["z"].cuda().to(BF16)
        x = M.load(next(a.valid.glob(f"{name}*.png")))
        recs = {"teacher": decode(teacher, z, H, W), "student": decode(student, z, H, W)}
        for who, y in recs.items():
            f = feat(x, y)
            f["mse"] = M.mse(x, y)
            rows.append(dict(name=name, who=who, bpp=r["bytes"] * 8 / (H * W), qhat_v0=qhat_score(qm, f), **f))
        s = 384
        y0, x0 = (H - s) // 2, (W - s) // 2
        crops.append([x[y0:y0 + s, x0:x0 + s], recs["teacher"][y0:y0 + s, x0:x0 + s], recs["student"][y0:y0 + s, x0:x0 + s]])
        print(name, " ".join(f"{rw['who']} Q {rw['qhat_v0']:+.3f} LPIPS {rw['lpips_alex']:.4f}" for rw in rows[-2:]), flush=True)

    with open(a.out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["psnr", "msssim", "lpips_alex", "dists", "wd3", "qhat_v0"]
    mean = {who: {k: float(np.mean([float(r[k]) for r in rows if r["who"] == who])) for k in keys} for who in ("teacher", "student")}
    d = [float(r2["qhat_v0"]) - float(r1["qhat_v0"]) for r1, r2 in zip(rows[0::2], rows[1::2])]
    L = [f"# Student vs teacher, DiT-IC {a.q}, {len(d)} CLIC validation images", "",
         f"Checkpoint: {a.ckpt} (step {ck.get('step')}). Same bitstreams (mean {np.mean([r['bpp'] for r in rows]):.4f} bpp).", "",
         "| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |", "|---|---|---|---|---|---|---|"]
    for who in ("teacher", "student"):
        m = mean[who]
        L.append(f"| {who} | {m['psnr']:.3f} | {m['msssim']:.4f} | {m['lpips_alex']:.4f} | {m['dists']:.4f} | {m['wd3']:.4f} | {m['qhat_v0']:+.3f} |")
    L += ["", f"Q-hat v0 student - teacher: mean {np.mean(d):+.3f}, worst {min(d):+.3f}, images worse by > 0.1: "
              f"{sum(v < -0.1 for v in d)}/{len(d)}"]
    (a.out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    for k in range(0, len(crops), 6):
        sheet = Image.new("RGB", (384 * 3, (384 + 16) * len(crops[k:k + 6])), "white")
        dr = ImageDraw.Draw(sheet)
        for i, trip in enumerate(crops[k:k + 6]):
            for j, (lab, im) in enumerate(zip(("original", "teacher", "student"), trip)):
                sheet.paste(Image.fromarray(im), (384 * j, i * 400 + 16))
                dr.text((384 * j + 3, i * 400 + 2), lab, fill="black")
        sheet.save(a.out / f"crops_{k // 6}.jpg", quality=90)


if __name__ == "__main__":
    main()
