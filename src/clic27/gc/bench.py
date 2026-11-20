"""Decoder speed / memory benchmark (run it on the target GPU, ideally an L4 inside the devkit-like container).

    python -m clic27.gc.bench --decoder RUN/package/decoder --bs RUN/encode/bpp-0.075/bs.zip [--repeat 2]
        [--synth-dtype fp16|bf16|fp32] [--ent-device cuda|cpu] [--exact f32|f64]

Prints per-image entropy / synthesis time, the whole-set estimate for CLIC (30 images), peak VRAM, and writes
bench.json next to bs.zip. The first pass is a warm-up (CUDA context, kernels) and is reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--bs", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--synth-dtype", default=None)
    ap.add_argument("--ent-device", default=None)
    ap.add_argument("--exact", default="f32", choices=["f32", "f64"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dec = Path(a.decoder)
    sys.path.insert(0, str(dec / "lib"))
    import torch
    from clic27.gc import codec
    codec.EXACT["mode"] = a.exact
    sys.path.insert(0, str(dec))
    from decode_main import load_decoder
    t0 = time.perf_counter()
    model, rt = load_decoder(dec / "model.pt", a.device)
    t_load = time.perf_counter() - t0
    ent = a.ent_device or rt.get("entropy_device", "cuda")
    sd = a.synth_dtype or rt.get("synth_dtype", "fp16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[sd]
    with zipfile.ZipFile(a.bs) as z:
        streams = [(Path(n).stem, z.read(n)) for n in sorted(z.namelist()) if n.endswith(".gc")]
    passes = []
    for r in range(a.repeat + 1):
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows = []
        tp = time.perf_counter()
        for name, buf in streams:
            t = {}
            ts = time.perf_counter()
            codec.decompress(model, buf, dev=a.device, ent_dev=ent, synth_dtype=dtype, timings=t)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            rows.append(dict(name=name, total=time.perf_counter() - ts, **t))
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if a.device.startswith("cuda") else 0.0
        passes.append(dict(wall=time.perf_counter() - tp, peak_gib=peak, rows=rows))
        print(f"pass {r}{' (warm-up)' if r == 0 else ''}: {passes[-1]['wall']:.2f}s for {len(rows)} images, "
              f"peak VRAM {peak:.2f} GiB", flush=True)
    best = min(passes[1:] or passes, key=lambda p: p["wall"])
    n = len(best["rows"])
    ent_t = sum(r["entropy"] for r in best["rows"]) / n
    syn_t = sum(r["synthesis"] for r in best["rows"]) / n
    est30 = best["wall"] / n * 30 + t_load
    res = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               images=n, load_s=t_load, per_image_s=best["wall"] / n, entropy_s=ent_t, synthesis_s=syn_t,
               est_30_images_s=est30, peak_vram_gib=best["peak_gib"], entropy_device=ent, synth_dtype=sd,
               exact=a.exact, passes=passes)
    out = Path(a.bs).with_name("bench.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "passes"}, indent=2))


if __name__ == "__main__":
    main()
