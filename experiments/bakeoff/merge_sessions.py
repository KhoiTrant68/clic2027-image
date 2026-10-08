"""Merge bake-off work dirs from several Kaggle sessions and re-run allocate + summary on the union (CPU, no torch).

Each session writes points/<cand>.csv and metrics.csv for the candidates it produced; recon PNGs are not kept,
so this only needs the CSVs. Image names and sizes come from budgets.json (the validation images are not needed).

    python experiments/bakeoff/merge_sessions.py --out results/<date>_bakeoff_merged \n        --qhat results/2026-10-06_qhat_v0/qhat_v0.json results/<session>/bakeoff ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcs", nargs="+", type=Path, help="bakeoff work dirs (each with points/ and metrics.csv)")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qhat", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo / "experiments" / "bakeoff"))
    import bakeoff as B
    M = B.M

    out = a.out
    (out / "points").mkdir(parents=True, exist_ok=True)
    shutil.copy(a.srcs[0] / "budgets.json", out / "budgets.json")
    metrics, mkeys, points = [], set(), {}
    for s in a.srcs:
        for r in (M.read_csv(s / "metrics.csv") if (s / "metrics.csv").exists() else []):
            k = (r["cand"], r["key"], r["name"])
            if k not in mkeys:
                mkeys.add(k)
                metrics.append(r)
        for f in sorted((s / "points").glob("*.csv")):
            for r in M.read_csv(f):
                points.setdefault((r["cand"], r["key"], r["name"]), r)
    kept = [r for k, r in points.items() if k in mkeys]
    for cand in sorted({r["cand"] for r in kept}):
        M.write_csv([r for r in kept if r["cand"] == cand], out / "points" / f"{cand}.csv")
        print(f"  {cand}: {sum(r['cand'] == cand for r in kept)} points, "
              f"{len({r['name'] for r in kept if r['cand'] == cand})} images")
    M.write_csv(metrics, out / "metrics.csv")
    print(f"merged {len(a.srcs)} sessions: {len(kept)} points with metrics, {len(points) - len(kept)} without (dropped)")

    names = sorted(json.loads((out / "budgets.json").read_text())["sizes"])
    B.inputs = lambda _a: {n: None for n in names}
    ns = argparse.Namespace(out=out, qhat=a.qhat, data=None)
    B.allocate(ns)
    B.summary(ns)


if __name__ == "__main__":
    main()
