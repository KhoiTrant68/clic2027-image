"""Fresh-directory offline decode with process wall time and pixel comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--bakeoff", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--gpu-memory-guard", action="store_true", required=True)
    ap.add_argument("--rate", type=float, default=0.3)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    streams = a.out / "input"
    streams.mkdir()
    with (a.bakeoff / "allocation.csv").open() as f:
        selection = [r for r in csv.DictReader(f) if r["cand"] == "pulse" and float(r["rate"]) == a.rate]
    expected = json.loads((a.bakeoff / "budgets.json").read_text())["n_images"]
    if len(selection) != expected:
        raise ValueError("Missing full-corpus PULSE allocation")
    for r in selection:
        source = a.bakeoff / "streams/pulse" / r["key"] / (r["name"] + ".bin")
        if source.stat().st_size != int(r["bytes"]):
            raise ValueError("Stream byte accounting mismatch")
        shutil.copy2(source, streams / source.name)
    began = time.perf_counter()
    with zipfile.ZipFile(a.archive) as z:
        for info in z.infolist():
            if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts:
                raise ValueError("Unsafe package member")
        z.extractall(a.out / "decoder")
    unpack_s = time.perf_counter() - began
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(a.out / "empty_hf_cache"), PYTHONNOUSERSITE="1", PYTHONPATH="",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [a.python, "-I", str(a.out / "decoder/decode.py"),
               "--input", str(streams), "--output", str(a.out / "images"),
               "--report", str(a.out / "decode.json"), "--gpu-memory-guard"]
    # Isolated Python removes script dir from sys.path; decode inserts its own
    # package directory explicitly before importing the shipped helpers.
    t0 = time.perf_counter()
    with (a.out / "decode.log").open("w") as log:
        subprocess.run(command, cwd=a.out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    process_s = time.perf_counter() - t0
    total_s = time.perf_counter() - began
    import numpy as np
    from PIL import Image
    checks = []
    for r in selection:
        ref = a.bakeoff / "recon/pulse" / r["key"] / (r["name"] + ".png")
        x = np.asarray(Image.open(ref)).astype(np.int16)
        y = np.asarray(Image.open(a.out / "images" / ref.name)).astype(np.int16)
        checks.append(dict(name=r["name"], key=r["key"], equal=bool(np.array_equal(x, y)),
                           max_abs_diff=int(np.max(np.abs(x-y)))))
    report = dict(unpack_s=unpack_s, process_wall_s=process_s, total_wall_s=total_s,
                  input_bytes=sum(p.stat().st_size for p in streams.glob("*.bin")),
                  output_count=len(list((a.out / "images").glob("*.png"))), checks=checks)
    (a.out / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    if not all(r["equal"] for r in checks) or report["output_count"] != expected:
        raise SystemExit("Independent decode verification failed")


if __name__ == "__main__":
    main()
