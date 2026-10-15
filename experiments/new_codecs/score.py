"""Score saved probes with the repo metrics and compare at matched byte budgets."""
import argparse
import json
import math
from pathlib import Path
import sys

from half_vram import install


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--qhat", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--only", nargs="+", default=[])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / "vram.jsonl")
    sys.path.insert(0, str(a.repo / "src"))
    sys.path.insert(0, str(a.repo / "experiments/qhat"))
    sys.path.insert(0, str(a.repo / "experiments/bakeoff"))
    from clic27.eval import metrics as M
    from qhat import Features
    from bakeoff import qhat_score, allocate_one
    qm = json.loads(a.qhat.read_text())
    files = {p.stem: p for p in sorted(a.data.glob("*.png")) if not a.only or any(p.name.startswith(s) for s in a.only)}
    if not files or (a.only and len(files) != len(a.only)):
        raise ValueError("Expected one image for each supplied prefix")
    pixels = sum(M.load(p).shape[0] * M.load(p).shape[1] for p in files.values())
    metric_path = a.out / "metrics.csv"
    rows = M.read_csv(metric_path) if metric_path.exists() else []
    seen = {(r["cand"], r["key"], r["name"]) for r in rows}
    feature = Features("cuda")
    options = {}
    for run in a.runs:
        meta = json.loads((run / "meta.json").read_text())
        cand = meta["model"]
        for q in meta["qualities"]:
            for name in files:
                path = run / f"q{q}" / f"{name}.json"
                if not path.exists():
                    raise ValueError(f"Incomplete run: missing {path}")
                rec = json.loads(path.read_text())
                if rec["status"] != "ok":
                    raise ValueError(f"Failed point: {path}")
                key = (cand, f"q{q}", name)
                if key not in seen:
                    x, y = M.load(files[name]), M.load(path.with_suffix(".png"))
                    f = feature(x, y, only=set(qm["features"]) | {"psnr", "lpips_alex", "dists"})
                    row = dict(cand=cand, key=f"q{q}", name=name, bytes=rec["bytes"],
                               t_dec=rec["decode_s"], **f, score=qhat_score(qm, f))
                    rows.append(row)
                    seen.add(key)
                    M.write_csv(rows, metric_path)
                    print(json.dumps(row), flush=True)
    bm = {(r["cand"], r["key"], r["name"]): r for r in M.read_csv(a.baseline / "metrics.csv")}
    base_rows = []
    for file in sorted((a.baseline / "points").glob("*.csv")):
        for r in M.read_csv(file):
            key = (r["cand"], r["key"], r["name"])
            if r["name"] in files and r["cand"] in {"msillm", "ditic", "ditics", "vtmscc", "codlite"} and key in bm:
                base_rows.append(dict(r, score=qhat_score(qm, bm[key])))
    for r in [*base_rows, *rows]:
        options.setdefault(r["cand"], {}).setdefault(r["name"], []).append(
            (int(r["bytes"]), float(r["score"]), r["key"]))
    old = {r["cand"] for r in base_rows}
    for label, candidates in (("baseline_mix", old), ("expanded_mix", set(options))):
        options[label] = {n: [(b, s, f"{c}/{k}") for c in candidates for b, s, k in options[c].get(n, [])]
                          for n in files}
    summary = []
    allocations = []
    for cand, opts in options.items():
        if set(opts) != set(files):
            raise ValueError(f"{cand} missing images, cannot rank it")
        for rate in (0.075, 0.15, 0.3):
            budget = math.floor(rate * pixels / 8)
            selected = allocate_one(opts, budget)
            row = dict(cand=cand, rate=rate, n=len(files), budget_bytes=budget,
                       feasible=selected is not None)
            if selected is not None:
                row.update(bpp=8 * sum(v[0] for v in selected.values()) / pixels,
                           qhat=sum(v[1] for v in selected.values()) / len(files))
                allocations.extend(dict(cand=cand, rate=rate, name=n, bytes=b, score=s, key=k)
                                   for n, (b, s, k) in selected.items())
            summary.append(row)
    M.write_csv(summary, a.out / "summary.csv")
    M.write_csv(allocations, a.out / "allocation.csv")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    stop.set()


if __name__ == "__main__":
    main()
