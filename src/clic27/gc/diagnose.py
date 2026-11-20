"""Health check of a GC run directory: reads logs and artefacts, prints findings with suggested actions.

    python -m clic27.gc.diagnose --work RUN [--out RUN/diagnose.md]

Every finding has an id (T*, E*, A*, D*, P*, R*) that matches a section of docs/GC_HUONG_DAN.md, where the
situation and the remedies are described in detail. Nothing here changes the run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

STAGES = ["rd", "gan", "rate", "align"]
L4_MAX_DECODER = 4 * 1000 ** 3


def jl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class Report:
    def __init__(self):
        self.rows = []

    def add(self, code, level, msg, action=""):
        self.rows.append((code, level, msg, action))

    def md(self):
        icon = {"ok": "OK", "warn": "CHÚ Ý", "bad": "LỖI", "info": "info"}
        out = ["| mã | mức | quan sát | hướng xử lý |", "|---|---|---|---|"]
        for c, l, m, a in self.rows:
            out.append(f"| {c} | {icon[l]} | {m} | {a} |")
        return "\n".join(out)


def tail_mean(rows, key, frac=0.2):
    v = [r[key] for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])]
    if not v:
        return None
    k = max(1, int(len(v) * frac))
    return float(np.mean(v[-k:]))


def check_training(work: Path, cfg: dict, rep: Report):
    targets = cfg["rate"]["targets"]
    for st in STAGES:
        d = work / "train" / st
        rows = jl(d / "log.jsonl")
        if not rows:
            continue
        last = rows[-1]["step"]
        steps = cfg["stages"][st]["steps"]
        done = (d / "model.pt").exists()
        its0 = tail_mean(rows, "it_s") or 0.0
        rep.add(f"T0-{st}", "info", f"{st}: bước {last}/{steps}, model.pt {'có' if done else 'chưa có'}, "
                f"{its0:.1f} it/s", "")
        over = tail_mean(rows, "over")
        if over is not None:
            if over > 0.6:
                rep.add("T1", "bad" if st != "rd" else "warn",
                        f"{st}: {over:.0%} mẫu train vượt bpp mục tiêu của level",
                        "Tăng rate.lambda_hi hoặc train thêm; xem T1")
            elif over < 0.15:
                rep.add("T2", "warn", f"{st}: chỉ {over:.0%} mẫu vượt mục tiêu: model có thể đang dưới mục tiêu nhiều",
                        "Giảm rate.lambda_lo hoặc kiểm tra eval bpp từng level; xem T2")
        its = tail_mean(rows, "it_s")
        if its is not None and its < 3:
            rep.add("T8", "warn", f"{st}: chỉ {its:.1f} it/s", "Kiểm tra loader (prep_patches, workers); xem T8")
        if st in ("gan", "rate", "align"):
            dr, df = tail_mean(rows, "d_real"), tail_mean(rows, "d_fake")
            if dr is not None and df is not None:
                if dr > 0.95 and df < 0.05:
                    rep.add("T4", "bad", f"{st}: D thắng tuyệt đối (P(real|real)={dr:.2f}, P(real|fake)={df:.2f})",
                            "Giảm lr_d / tăng k_gan dần; xem T4")
                elif abs(dr - df) < 0.02:
                    rep.add("T5", "warn", f"{st}: D không phân biệt được (real {dr:.2f} / fake {df:.2f})",
                            "Kiểm tra D có học (lr_d), hoặc G đã sinh tốt; xem T5")
        g = tail_mean(rows, "gnorm")
        if g is not None and g > 50 * cfg["train"]["grad_clip"]:
            rep.add("T6", "warn", f"{st}: gradient norm trung bình {g:.1f} (bị clip liên tục)", "Giảm lr; xem T6")
        ps = [r.get("psnr") for r in rows if "psnr" in r]
        if len(ps) > 10 and st != "rd":
            a, b = np.mean(ps[: len(ps) // 5]), np.mean(ps[-len(ps) // 5:])
            if b < a - 2.0:
                rep.add("T7", "warn", f"{st}: PSNR train giảm {a - b:.1f} dB từ đầu stage",
                        "GAN đang hy sinh fidelity quá nhiều; tăng k_mse/k_lpips, giảm beta_max; xem T7")
        ev = jl(d / "eval.jsonl")
        if ev:
            lv = ev[-1]["levels"]
            ks = sorted(lv, key=int)
            bpps = [lv[k]["bpp"] for k in ks]
            psn = [lv[k]["psnr"] for k in ks]
            if any(b2 < b1 * 0.98 for b1, b2 in zip(bpps, bpps[1:])):
                rep.add("T3", "bad", f"{st}: bpp eval không tăng theo level: {[round(b, 3) for b in bpps]}",
                        "Level chưa tách: train thêm, kiểm tra log_gain; xem T3")
            if any(p2 < p1 - 0.3 for p1, p2 in zip(psn, psn[1:])):
                rep.add("T3", "warn", f"{st}: PSNR eval không tăng theo level: {[round(p, 2) for p in psn]}", "Xem T3")
            lo, hi = min(bpps), max(bpps)
            msg = f"{st}: eval bpp phủ [{lo:.3f}, {hi:.3f}] (levels {ks[0]}..{ks[-1]})"
            ok = lo <= 0.075 * 0.85 and hi >= 0.3
            rep.add("E1", "ok" if ok else "warn", msg, "" if ok else "Không phủ đủ 0,075–0,3 bpp; xem E1")
            for k in ks:
                t = targets[int(k)]
                if abs(lv[k]["bpp"] - t) > 0.5 * t:
                    rep.add("E2", "warn", f"{st}: level {k} ảnh full {lv[k]['bpp']:.3f} bpp, mục tiêu {t}",
                            "Crop train khác ảnh full; dựa vào phân bổ, không cần khớp; xem E2")
                    break


def check_reward(work: Path, cfg: dict, rep: Report):
    rj = Path(cfg.get("reward_dir", work / "reward")) / "report.json"
    if not rj.exists():
        rep.add("R0", "info", "Không có reward model (alloc dùng LPIPS)", "")
        return
    r = json.loads(rj.read_text())
    acc, init, n = r["final"]["acc"], r["init"]["acc"], r["final"]["n"]
    se = math.sqrt(max(acc * (1 - acc), 1e-9) / max(n, 1))
    lvl = "ok" if acc > init + 2 * se else "warn"
    rep.add("R1", lvl, f"Q̂ v2 held-out acc {acc:.3f} (khởi tạo {init:.3f}, n={n}, ±{2 * se:.3f}), tau {r['tau']:.2f}",
            "" if lvl == "ok" else "Reward chưa hơn LPIPS-style init: tăng n_per_set/steps; xem R1")


def check_encode(work: Path, cfg: dict, rep: Report):
    for d in sorted((work / "encode").glob("bpp-*")):
        aj = d / "allocation.json"
        if not aj.exists():
            continue
        a = json.loads(aj.read_text())
        b = a["budget_bpp"]
        if not a["feasible"]:
            rep.add("A1", "bad", f"{d.name}: không tìm được phân bổ dưới ngân sách (đang {a['bpp']:.4f} bpp)",
                    "Model không xuống đủ thấp: thêm level thấp / train lại; xem A1")
        elif not a["within_budget"]:
            rep.add("A2", "bad", f"{d.name}: bs.zip {a['bpp']:.5f} bpp > {b}", "Giảm alloc.safety; xem A2")
        else:
            use = a["bpp"] / b
            rep.add("A3", "ok" if use > 0.97 else "warn", f"{d.name}: {a['bpp']:.5f} bpp = {use:.1%} ngân sách",
                    "" if use > 0.97 else "Phí byte: tăng safety, thêm candidate; xem A3")
        imgs = a["images"]
        levels = cfg["alloc"]["levels"][str(b)]
        at_min = sum(i["level"] == min(levels) for i in imgs)
        at_max = sum(i["level"] == max(levels) for i in imgs)
        if at_max > 0.3 * len(imgs):
            rep.add("A4", "warn", f"{d.name}: {at_max}/{len(imgs)} ảnh ở level cao nhất được phép",
                    "Mở rộng alloc.levels lên trên; xem A4")
        if at_min > 0.3 * len(imgs):
            rep.add("A4", "warn", f"{d.name}: {at_min}/{len(imgs)} ảnh ở level thấp nhất được phép",
                    "Mở rộng alloc.levels xuống dưới; xem A4")
        errs = [abs(i["bytes"] - i["est_bytes"]) / max(i["est_bytes"], 1) for i in imgs]
        if errs and max(errs) > 0.05:
            rep.add("A5", "warn", f"{d.name}: byte thật lệch ước lượng tới {max(errs):.1%}",
                    "Ước lượng rate kém (scale table/escape); xem A5")


def check_verify(work: Path, rep: Report):
    v = work / "verify.json"
    if not v.exists():
        return
    res = json.loads(v.read_text())
    for k, r in res.items():
        if r["images"] != r["expected"]:
            rep.add("D1", "bad", f"{k}: decoder ra {r['images']}/{r['expected']} ảnh", "Xem D1")
        rp = work / "encode" / k / "verify" / "decode_report.json"
        if rp.exists():
            d = json.loads(rp.read_text())
            ent = [x["entropy"] for x in d["rows"]]
            syn = [x["synthesis"] for x in d["rows"]]
            per = d["wall"] / max(len(d["rows"]), 1)
            rep.add("D2", "info", f"{k}: {per:.2f} s/ảnh trên máy này (entropy TB {np.mean(ent):.2f}s, synthesis "
                    f"{np.mean(syn):.2f}s, entropy_device {d['entropy_device']})",
                    "Đây không phải số L4; đo bằng scripts/gc/l4_verify.sh; xem D2")


def check_package(work: Path, rep: Report):
    p = work / "package" / "package.json"
    if not p.exists():
        return
    info = json.loads(p.read_text())
    lvl = "ok" if info["bytes"] < 0.9 * L4_MAX_DECODER else "bad"
    rep.add("P1", lvl, f"decoder.zip {info['bytes'] / 1e6:.1f} MB", "" if lvl == "ok" else "Vượt/gần 4 GB; xem P1")
    if not (work / "package" / "decoder" / "lib" / "clic27" / "gc" / "_rans.so").exists():
        rep.add("P2", "bad", "Gói decoder không có _rans.so build sẵn", "Xem P2")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    work = Path(a.work)
    cfg = json.loads((work / "config.json").read_text())
    rep = Report()
    check_training(work, cfg, rep)
    check_reward(work, cfg, rep)
    check_encode(work, cfg, rep)
    check_verify(work, rep)
    check_package(work, rep)
    text = f"# Chẩn đoán run `{work}`\n\nMã lỗi tra trong docs/GC_HUONG_DAN.md (mục 9).\n\n{rep.md()}\n"
    out = Path(a.out) if a.out else work / "diagnose.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
