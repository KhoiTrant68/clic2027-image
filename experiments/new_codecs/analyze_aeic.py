"""Finish the 30-image AEIC experiment against the integrated PULSE baseline.

Waits for both independent decoders, scores with the existing scorer, and creates
a local blind A/B review pack. Human votes are never inferred from metric scores.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import time


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    home = a.root.parent
    baseline = home / "research_20261015/integrated_bakeoff"
    repo = home / "research_20261015/integration_repo"
    data = home / "work_full_gpu/bakeoff/valid"
    runs = [a.run / f"aeic_{v}_ft2" for v in ("se", "me")]
    names = sorted(p.stem for p in data.glob("*.png"))
    assert len(names) == 30, "Full experiment requires exactly 30 images"
    status = a.run / "analysis_status.json"

    def report(state, **kwargs):
        status.write_text(json.dumps(dict(state=state, updated_unix=time.time(), **kwargs), indent=2))

    try:
        report("waiting_for_both_decoders")
        deadline = time.monotonic() + 3 * 3600
        while time.monotonic() < deadline:
            states = [json.loads((r / "pair_status.json").read_text())["state"] for r in runs]
            if "failed" in states:
                raise RuntimeError(f"Codec phase failed: {states}")
            if states == ["complete", "complete"]:
                break
            time.sleep(30)
        else:
            raise TimeoutError("AEIC stages did not finish within 3 hours")
        for run in runs:
            parity = json.loads((run / "parity.json").read_text())
            assert len(parity) == 30 and all(p["same_pixels"] for p in parity)
            assert {p["name"] for p in parity} == set(names)
            for name in names:
                path = run / "q2" / (name + ".json")
                r = json.loads(path.read_text())
                assert r["status"] == "ok" and r["bytes"] == path.with_suffix(".bin").stat().st_size
        # Both codec processes have exited; select the GPU with most free memory.
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        gpu, _ = max(([int(x.strip()) for x in row.split(",")] for row in rows), key=lambda x: x[1])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AEIC_SHARED_SMALL="1")
        report("scoring", gpu=gpu)
        with (a.run / "score.log").open("w") as log:
            subprocess.run([str(home / "venv/bin/python"), str(a.root / "score.py"),
                "--repo", str(repo), "--baseline", str(baseline),
                "--qhat", str(home / "inputs/qhat_v0/qhat/qhat_v0.json"),
                "--data", str(data), "--out", str(a.run / "scores"),
                "--runs", *map(str, runs)], env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=3600)
        report("making_comparisons")
        scores = a.run / "scores"
        summary = read_csv(scores / "summary.csv")
        reference = json.loads((baseline / "summary.json").read_text())
        comparisons = []
        allocations = read_csv(scores / "allocation.csv")
        changes = []
        for rate in (0.075, 0.15, 0.3):
            old = next(r for r in summary if r["cand"] == "baseline_mix" and float(r["rate"]) == rate)
            new = next(r for r in summary if r["cand"] == "expanded_mix" and float(r["rate"]) == rate)
            assert abs(float(old["qhat"]) - reference[f"mix@{rate}/all"]["qhat"]) < 1e-6, "Baseline reproduction failed"
            before = {r["name"]: r for r in allocations if r["cand"] == "baseline_mix" and float(r["rate"]) == rate}
            after = {r["name"]: r for r in allocations if r["cand"] == "expanded_mix" and float(r["rate"]) == rate}
            assert len(before) == len(after) == 30
            counts = {}
            for name in names:
                b, n = before[name], after[name]
                cand = n["key"].split("/")[0]
                counts[cand] = counts.get(cand, 0) + 1
                if b["key"] != n["key"]:
                    changes.append(dict(rate=rate, name=name, before=b["key"], after=n["key"],
                        before_bytes=int(b["bytes"]), after_bytes=int(n["bytes"]),
                        before_qhat=float(b["score"]), after_qhat=float(n["score"]),
                        delta_qhat=float(n["score"]) - float(b["score"])))
            comparisons.append(dict(rate=rate, baseline_qhat=float(old["qhat"]),
                expanded_qhat=float(new["qhat"]), delta_qhat=float(new["qhat"]) - float(old["qhat"]),
                baseline_bpp=float(old["bpp"]), expanded_bpp=float(new["bpp"]),
                aeic_selected=sum(v for k, v in counts.items() if k.startswith("aeic_")),
                counts=counts))
        (a.run / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        write_csv(a.run / "changed_images.csv", changes)

        from PIL import Image
        import numpy as np
        recon = {"pulse": baseline / "recon/pulse",
            "msillm": home / "work_full_gpu/bakeoff/recon/msillm",
            "codlite": home / "work_full_gpu/bakeoff/recon/codlite",
            "ditic": home / "work_new/bakeoff/recon/ditic",
            "ditics": home / "work_ditics/bakeoff/recon/ditics",
            "vtmscc": home / "work_full_vtm/bakeoff/recon/vtmscc",
            "aeic_se": runs[0], "aeic_me": runs[1]}

        def image_path(key, name):
            cand, quality = key.split("/")
            return recon[cand] / quality / (name + ".png")

        review = a.run / "blind_review"
        (review / "img").mkdir(parents=True, exist_ok=True)
        public, private = [], []
        rng = random.Random(20271020)
        seen = set()
        for change in changes:
            identity = (change["name"], change["before"], change["after"])
            if identity in seen:
                continue
            seen.add(identity)
            source = Image.open(data / (change["name"] + ".png")).convert("RGB")
            before = Image.open(image_path(change["before"], change["name"])).convert("RGB")
            after = Image.open(image_path(change["after"], change["name"])).convert("RGB")
            assert source.size == before.size == after.size
            w, h = source.size
            size = min(512, w, h)
            positions = [((w - size) // 2, (h - size) // 2)]
            for _ in range(2):
                for attempt in range(50):
                    x, y = rng.randrange(w - size + 1), rng.randrange(h - size + 1)
                    if np.asarray(source.crop((x, y, x + size, y + size))).std() > 6:
                        break
                positions.append((x, y))
            for x, y in positions:
                pid = f"p{len(public):03d}"
                swap = bool(rng.getrandbits(1))
                images = [source, after if swap else before, before if swap else after]
                for label, image in zip(("ref", "a", "b"), images):
                    image.crop((x, y, x + size, y + size)).save(review / "img" / f"{pid}_{label}.webp", lossless=True)
                public.append(dict(id=pid))
                private.append(dict(id=pid, **change, x=x, y=y, size=size,
                                    a="expanded" if swap else "baseline", b="baseline" if swap else "expanded"))
        # Keep answer key outside the page; page only sees anonymous A/B pixels.
        (a.run / "blind_key.json").write_text(json.dumps(private, indent=2))
        html = '''<!doctype html><meta charset="utf-8"><title>AEIC — Blind review</title>
<style>body{font:16px system-ui;background:#f5f5f5;margin:24px;color:#17212b}img{width:100%;max-width:512px;image-rendering:auto}.grid{display:flex;gap:16px;flex-wrap:wrap}.grid>div{flex:1;min-width:260px}button{padding:12px;margin:5px;cursor:pointer}.active{background:#bde9d5}#ref{max-width:400px}p{max-width:950px}</style>
<h1>So sánh chất lượng ảnh A/B</h1><p>Đối chiếu với ảnh gốc. Chọn ảnh giữ cấu trúc, chữ, khuôn mặt và texture tốt hơn; đừng chỉ chọn ảnh sắc nét hơn. Các cặp lấy từ hai phương án phân bổ cùng ngân sách toàn tập; bitrate từng ảnh có thể khác. Chưa có phiếu đánh giá nào được điền sẵn.</p>
<p id="progress"></p><details open><summary>Ảnh gốc</summary><img id="ref"></details>
<div class="grid"><div><h2>A</h2><img id="a"></div><div><h2>B</h2><img id="b"></div></div>
<div id="choices"></div><button onclick="move(-1)">Trước</button><button onclick="move(1)">Sau</button><button onclick="save()">Tải phiếu JSON</button>
<script>const pairs=PAIRS;let i=0;const storage='aeic-full30-20261020';let votes=JSON.parse(localStorage.getItem(storage)||'{}');
function render(){if(!pairs.length){document.getElementById('progress').textContent='Không có lựa chọn thay đổi để chấm.';return;}const id=pairs[i].id;document.getElementById('progress').textContent=`Cặp ${i+1}/${pairs.length} — Đã chấm ${Object.keys(votes).length}`;for(const k of ['ref','a','b'])document.getElementById(k).src=`img/${id}_${k}.webp`;const c=document.getElementById('choices');c.replaceChildren();for(const [v,t] of [['A','A tốt hơn'],['B','B tốt hơn'],['tie','Ngang nhau'],['uncertain','Khó đánh giá']]){const b=document.createElement('button');b.textContent=t;b.className=votes[id]===v?'active':'';b.onclick=()=>{votes[id]=v;localStorage.setItem(storage,JSON.stringify(votes));render()};c.appendChild(b)}}
function move(n){i=Math.max(0,Math.min(pairs.length-1,i+n));render()}function save(){const blob=new Blob([JSON.stringify({study:storage,created:new Date().toISOString(),votes},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='aeic_blind_votes.json';a.click();URL.revokeObjectURL(a.href)}render();</script>'''
        (review / "index.html").write_text(html.replace("PAIRS", json.dumps(public)), encoding="utf-8")
        metric_gate = any(r["delta_qhat"] >= .02 and r["aeic_selected"] >= 3 for r in comparisons)
        decision = dict(metric_gate=metric_gate, threshold="delta Qhat >= 0.02 and >=3 AEIC-selected images at at least one rate",
            human_review="pending", fine_tuning_started=False,
            reason="Await independent visual votes and decoder-cost assessment" if metric_gate else "Metric gain does not meet the predeclared practical threshold",
            blind_pairs=len(public))
        (a.run / "decision.json").write_text(json.dumps(decision, indent=2))
        report("complete", comparisons=comparisons, **decision)
    except Exception as e:
        report("failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
