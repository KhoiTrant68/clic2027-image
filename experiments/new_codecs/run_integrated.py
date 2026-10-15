"""Run PULSE through the main bake-off CLI with isolated codec/metric interpreters.

Never alters the baseline. Creates a fresh output directory, copies its cached
metrics/points, runs actual PULSE coding, then scores and allocates the union.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--qhat", type=Path, required=True)
    ap.add_argument("--codec-python", required=True)
    ap.add_argument("--metric-python", required=True)
    ap.add_argument("--gpu-memory-guard", action="store_true", required=True,
                    help="required: forward half-device guard to both GPU subprocesses")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "points").mkdir()
    candidates = {"codlite", "ditic", "ditics", "msillm", "vtmscc"}
    for candidate in sorted(candidates):
        shutil.copy2(a.baseline / "points" / f"{candidate}.csv", a.out / "points")
    with (a.baseline / "metrics.csv").open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r["cand"] in candidates]
    with (a.out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    common = [str(a.repo / "experiments/bakeoff/bakeoff.py"), "PLACEHOLDER",
              "--out", str(a.out), "--data", str(a.data), "--qhat", str(a.qhat)]
    commands = []
    for step, python in (("prepare", a.metric_python), ("points", a.codec_python),
                         ("metrics", a.metric_python), ("allocate", a.metric_python),
                         ("summary", a.metric_python)):
        args = common.copy()
        args[1] = step
        if step in {"points", "metrics"}:
            args += ["--gpu-memory-guard"]
        if step == "points":
            args += ["--cands", "pulse", "--defer-metrics", "--pulse-bundle", str(a.bundle),
                     "--pulse-runtime", str(a.runtime), "--pulse-q", "0", "1", "2", "3", "4"]
        command = [python, "-u", *args]
        commands.append(command)
        (a.out / "commands.json").write_text(json.dumps(commands, indent=2))
        print("RUN", step, flush=True)
        with (a.out / f"{step}.log").open("w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        if step == "points":
            with (a.out / "points/pulse.csv").open() as f:
                points = list(csv.DictReader(f))
            expected = json.loads((a.out / "budgets.json").read_text())["n_images"] * 5
            if len(points) != expected:
                raise RuntimeError(f"Expected {expected} PULSE points, got {len(points)}")
    print("COMPLETED", a.out, flush=True)


if __name__ == "__main__":
    main()
