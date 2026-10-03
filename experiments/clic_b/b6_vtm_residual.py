"""B6: real VTM anchors and a DC-AE + residual lower bound, on a 7-image subset.

Configs written to --out/recon/<cfg>/<name>.png:
  dcae              DC-AE f32c32 autoencode (ceiling, no quantisation)
  vtm420@<r>        VTM intra, 4:2:0 10-bit (close to the CLIC VTM baseline)
  vtmscc@<r>        VTM intra, 4:4:4 10-bit + SCC tools (IBC, palette, BDPCM, TS)
  dcae+res@<r>      DC-AE ceiling + residual (x - x̂) coded by VTM 4:4:4 with (r - BASE_BPP) bpp.
                    Pessimistic stand-in for a learned conditional residual layer; assumes the
                    base costs BASE_BPP (the CVPR codec's operating range).

Steps: prepare, dcae, vtm, inventory, metrics, crops, summary   (or: all)
Every step skips outputs that already exist, so the notebook can be re-run after a timeout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

import b_analysis as B

SEL = {  # prefix -> label (labels confirmed by eye in B5)
    "ebfd571f": "screen", "bb7344a2": "screen", "86127fbd": "screen", "2a760bf1": "screen",
    "da52c4f6": "natural", "2684452d": "natural", "608cb09e": "natural",
}
RATES = [0.075, 0.15, 0.3]
BASE_BPP = 0.03
QP0 = {0.075: 42, 0.15: 37, 0.3: 32}
SCC_FLAGS = [
    ["--IBC=1", "--PLT=1", "--BDPCM=1", "--TransformSkip=1", "--TransformSkipFast=0"],
    ["--IBC=1", "--PLT=1"],
    [],
]
KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB


# ---------------------------------------------------------------- colour / yuv io
def rgb_to_yuv10(x):
    """uint8 RGB -> full-range BT.709 YCbCr planes in 10-bit code values (float)."""
    r, g, b = (x[..., i].astype(np.float64) for i in range(3))
    y = KR * r + KG * g + KB * b
    cb = (b - y) / (2 * (1 - KB)) + 128
    cr = (r - y) / (2 * (1 - KR)) + 128
    return [4 * y, 4 * cb, 4 * cr]


def yuv10_to_rgb(y, cb, cr):
    y, cb, cr = y / 4, cb / 4 - 128, cr / 4 - 128
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return np.clip(np.round(np.stack([r, g, b], -1)), 0, 255).astype(np.uint8)


def resize(p, w, h):
    return np.asarray(Image.fromarray(p.astype(np.float32), "F").resize((w, h), Image.LANCZOS), np.float64)


def write_yuv(planes, path):
    np.concatenate([np.clip(np.round(p), 0, 1023).astype("<u2").ravel() for p in planes]).tofile(path)


def read_yuv(path, W, H, fmt):
    d = np.fromfile(path, dtype="<u2")
    for m in (1, 8, 16, 32, 64, 128):  # the recon may be padded to the CU grid
        Wp, Hp = -(-W // m) * m, -(-H // m) * m
        cw, ch = (Wp, Hp) if fmt == "444" else (Wp // 2, Hp // 2)
        if d.size == Wp * Hp + 2 * cw * ch:
            break
    else:
        raise RuntimeError(f"{path}: size {d.size} does not match {W}x{H} {fmt}")
    y = d[:Wp * Hp].reshape(Hp, Wp)
    cb = d[Wp * Hp:Wp * Hp + cw * ch].reshape(ch, cw)
    cr = d[Wp * Hp + cw * ch:].reshape(ch, cw)
    c = (W, H) if fmt == "444" else (W // 2, H // 2)
    return [y[:H, :W].astype(np.float64), cb[:c[1], :c[0]].astype(np.float64), cr[:c[1], :c[0]].astype(np.float64)]


# ---------------------------------------------------------------- VTM
def vtm_bin(vtm_dir):
    for n in ("EncoderAppStatic", "EncoderApp"):
        hits = list(Path(vtm_dir, "bin").rglob(n))
        if hits:
            return str(hits[0])
    raise FileNotFoundError("VTM EncoderApp not found; run the build cell first")


def vtm_encode(enc, cfg, yuv, W, H, fmt, qp, bin_, rec, extra):
    cmd = [enc, "-c", cfg, "-i", yuv, "-wdt", str(W), "-hgt", str(H), "-fr", "1", "-f", "1",
           "--InputBitDepth=10", "--InternalBitDepth=10", "--OutputBitDepth=10",
           f"--InputChromaFormat={fmt}", f"--ChromaFormatIDC={fmt}", "--ConformanceWindowMode=1",
           "-q", str(qp), "-b", bin_, "-o", rec] + extra
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-500:])
    return os.path.getsize(bin_)


def qp_search(encode, target, qp0, max_iter=7):
    """Smallest QP whose bitstream fits target bytes. encode(qp) -> bytes."""
    tried = {}

    def f(q):
        if q not in tried:
            tried[q] = encode(q)
        return tried[q]

    q = qp0
    for _ in range(max_iter):
        s = f(q)
        fit = [k for k, v in tried.items() if v <= target]
        over = [k for k, v in tried.items() if v > target]
        if fit and over:
            lo, hi = max(over), min(fit)  # lo too big, hi fits
            if hi - lo <= 1:
                break
            t = math.log(tried[lo] / target) / math.log(tried[lo] / tried[hi])
            q = min(max(int(round(lo + t * (hi - lo))), lo + 1), hi - 1)
        elif fit:  # everything fits -> lower QP (better quality)
            q = max(0, min(fit) - max(1, round(6 * math.log2(target / s))))
            if q in tried:
                break
        else:  # nothing fits -> raise QP
            q = min(63, max(over) + max(1, round(6 * math.log2(s / target))))
            if q in tried:
                break
    fit = [k for k, v in tried.items() if v <= target]
    return (min(fit), tried[min(fit)], len(tried)) if fit else (max(tried), tried[max(tried)], len(tried))


def job(kind, name, rate, data, out, vtm_dir):
    """Runs in a worker process. kind in {vtm420, vtmscc, res}."""
    cfg_name = {"vtm420": "vtm420", "vtmscc": "vtmscc", "res": "dcae+res"}[kind]
    dst = Path(out) / "recon" / f"{cfg_name}@{rate}" / f"{name}.png"
    if dst.exists():
        return None
    x = B.load(Path(data) / f"{name}.png")
    H, W = x.shape[:2]
    enc, cfgf = vtm_bin(vtm_dir), str(Path(vtm_dir) / "cfg" / "encoder_intra_vtm.cfg")
    with tempfile.TemporaryDirectory() as td:
        yuv = f"{td}/in.yuv"
        if kind == "vtm420":
            fmt = "420"
            y, cb, cr = rgb_to_yuv10(x)
            write_yuv([y, resize(cb, W // 2, H // 2), resize(cr, W // 2, H // 2)], yuv)
            target = math.floor(rate * H * W / 8)
        elif kind == "vtmscc":
            fmt = "444"
            write_yuv(rgb_to_yuv10(x), yuv)
            target = math.floor(rate * H * W / 8)
        else:
            fmt = "444"
            xh = B.load(Path(out) / "recon" / "dcae" / f"{name}.png").astype(np.float64)
            write_yuv([x[..., i] - xh[..., i] + 512 for i in range(3)], yuv)  # RGB residual, offset
            target = math.floor((rate - BASE_BPP) * H * W / 8)
        extras = SCC_FLAGS if kind == "vtmscc" else [[]]
        for extra in extras:  # fall back if this VTM build rejects an SCC option
            try:
                enc_fn = lambda q: vtm_encode(enc, cfgf, yuv, W, H, fmt, q,  # noqa: E731
                                              f"{td}/q{q}.bin", f"{td}/q{q}.yuv", extra)
                qp, size, n_enc = qp_search(enc_fn, target, QP0[rate])
                break
            except RuntimeError as e:
                err = str(e)
        else:
            raise RuntimeError(f"{kind} {name} {rate}: {err}")
        planes = read_yuv(f"{td}/q{qp}.yuv", W, H, fmt)
        if kind == "vtm420":
            planes = [planes[0], resize(planes[1], W, H), resize(planes[2], W, H)]
        if kind == "res":
            rec = np.clip(np.round(xh + np.stack([p - 512 for p in planes], -1)), 0, 255).astype(np.uint8)
            size_total = size + math.floor(BASE_BPP * H * W / 8)  # nominal base cost
        else:
            rec = yuv10_to_rgb(*planes)
            size_total = size
    B.save(rec, dst)
    return dict(kind=kind, name=name, rate=rate, qp=qp, bytes_coded=size, bytes_total=size_total,
                bpp=round(size_total * 8 / (H * W), 4), n_encodes=n_enc, scc_flags=" ".join(extra))


# ---------------------------------------------------------------- steps
def prepare(args):
    full = Path("data/valid")
    B.download(full)
    args.data.mkdir(parents=True, exist_ok=True)
    for p in B.images(full):
        if p.stem[:8] in SEL and not (args.data / p.name).exists():
            shutil.copy(p, args.data / p.name)
    print(sorted(p.stem[:8] for p in B.images(args.data)))


def dcae(args):
    B.ae_roundtrip(args, "dcae")


def vtm(args):
    names = [p.stem for p in B.images(args.data)]
    jobs = [(k, n, r) for r in RATES for n in names for k in ("vtm420", "vtmscc", "res")]
    log_path = args.out / "vtm_log.csv"
    log = B.read_csv(log_path) if log_path.exists() else []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, k, n, r, str(args.data), str(args.out), args.vtm): (k, n, r) for k, n, r in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                row = f.result()
            except Exception as e:  # keep going; report at the end
                print(f"[{i}/{len(jobs)}] FAILED {futs[f]}: {e}")
                continue
            if row:
                log.append(row)
                B.write_csv(log, log_path)
            print(f"[{i}/{len(jobs)}]", row or f"skip (exists) {futs[f]}")


def inventory(args):
    B.inventory(args)
    rows = B.read_csv(args.out / "inventory.csv")
    for r in rows:
        r["label"] = SEL[r["name"][:8]]
    B.write_csv(rows, args.out / "inventory.csv")


def metrics(args):
    B.metrics(args)


def crops(args):
    B.crops(args)


def summary(args):
    inv = {r["name"]: r for r in B.read_csv(args.out / "inventory.csv")}
    rows = B.read_csv(args.out / "metrics.csv")
    log = {(r["kind"].replace("res", "dcae+res"), r["name"], r["rate"]): r
           for r in B.read_csv(args.out / "vtm_log.csv")}

    def mean(rs, k):
        v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
        return sum(v) / len(v) if v else float("nan")

    L = ["# B6: VTM thật, VTM-SCC, và DC-AE + residual", "",
         f"Base của `dcae+res` giả định tốn {BASE_BPP} bpp (danh nghĩa), residual được mã hóa bằng VTM 4:4:4 "
         "với phần bit còn lại. Đây là **cận dưới** cho một lớp residual học được.", ""]
    for g in ("screen", "natural"):
        L += [f"## {g}", "", "| rate | cfg | bpp thật | PSNR | LPIPS↓ | DISTS↓ | text PSNR | OCR CER↓ |",
              "|---|---|---|---|---|---|---|---|"]
        rs_g = [r for r in rows if inv[r["name"]]["label"] == g]
        dc = [r for r in rs_g if r["cfg"] == "dcae"]
        L.append(f"| ∞ | dcae (trần) | – | {mean(dc, 'psnr'):.2f} | {mean(dc, 'lpips'):.4f} | {mean(dc, 'dists'):.4f} "
                 f"| {mean(dc, 'text_psnr'):.2f} | {mean(dc, 'ocr_cer'):.3f} |")
        for rate in RATES:
            for c in ("vtm420", "vtmscc", "dcae+res"):
                rs = [r for r in rs_g if r["cfg"] == f"{c}@{rate}"]
                if not rs:
                    continue
                bpp = [float(log[(c, r["name"], str(rate))]["bpp"]) for r in rs if (c, r["name"], str(rate)) in log]
                L.append(f"| {rate} | {c} | {sum(bpp) / max(len(bpp), 1):.3f} | {mean(rs, 'psnr'):.2f} "
                         f"| {mean(rs, 'lpips'):.4f} | {mean(rs, 'dists'):.4f} | {mean(rs, 'text_psnr'):.2f} "
                         f"| {mean(rs, 'ocr_cer'):.3f} |")
        L.append("")
    L += ["## Từng ảnh (PSNR / OCR CER)", "", "| ảnh | nhãn | cfg | PSNR | CER |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (inv[r["name"]]["label"], r["name"], r["cfg"])):
        L.append(f"| {r['name'][:8]} | {inv[r['name']]['label']} | {r['cfg']} | {float(r['psnr']):.2f} "
                 f"| {float(r['ocr_cer']):.3f} |" if r.get("ocr_cer") else
                 f"| {r['name'][:8]} | {inv[r['name']]['label']} | {r['cfg']} | {float(r['psnr']):.2f} | – |")
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "dcae", "vtm", "inventory", "metrics", "crops", "summary", "all"])
    ap.add_argument("--data", type=Path, default=Path("data/b6"))
    ap.add_argument("--out", type=Path, default=Path("b6_out"))
    ap.add_argument("--vtm", default="vtm")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    steps = ["prepare", "dcae", "vtm", "inventory", "metrics", "crops", "summary"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"===== {s}")
        globals()[s](args)


if __name__ == "__main__":
    main()
