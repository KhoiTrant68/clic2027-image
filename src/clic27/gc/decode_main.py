"""CLIC decoder entry point: bs.zip (or *.gc files in a folder) -> images/<name>.png.

Packaged next to `lib/` (clic27.entropy + clic27.gc) and `model.pt` by clic27.gc.package; the `decode` shell
script calls this file. Needs only python3 + torch + numpy + pillow (the CLIC devkit image has them).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "lib").exists():
    sys.path.insert(0, str(HERE / "lib"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from clic27.gc import codec  # noqa: E402
from clic27.gc.model import GCodec  # noqa: E402


def load_decoder(path, dev):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = GCodec(sd["cfg"]["model"])
    missing, unexpected = m.load_state_dict(sd["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith(("g_a.", "h_a."))]
    if bad or unexpected:
        raise RuntimeError(f"decoder weights mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")
    return m.to(dev).eval(), sd.get("runtime", {})


def streams(inp: Path):
    if inp.is_file():
        with zipfile.ZipFile(inp) as z:
            for n in sorted(z.namelist()):
                if n.endswith(".gc") and not n.startswith("__MACOSX"):
                    yield Path(n).stem, z.read(n)
    else:
        for p in sorted(inp.glob("*.gc")):
            yield p.stem, p.read_bytes()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="bs.zip")
    ap.add_argument("--output", default="images")
    ap.add_argument("--model", default=str(HERE / "model.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ent-device")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    model, rt = load_decoder(a.model, a.device)
    ent = a.ent_device or rt.get("entropy_device", "cuda")
    if ent.startswith("cuda") and not torch.cuda.is_available():
        ent = "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[rt.get("synth_dtype", "fp16")]
    inp = Path(a.input)
    if not inp.exists():  # the server may already have unpacked bs.zip into the working directory
        inp = Path(".")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(2) as pool:  # PNG writing overlaps with the next image's decoding
        futs = []
        for name, buf in streams(inp):
            t = {}
            ts = time.perf_counter()
            img = codec.decompress(model, buf, dev=a.device, ent_dev=ent, synth_dtype=dtype, timings=t)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            t["total"] = time.perf_counter() - ts
            rows.append(dict(name=name, bytes=len(buf), **t))
            futs.append(pool.submit(lambda im, p: Image.fromarray(im).save(p, compress_level=1), img,
                                    out / f"{name}.png"))
        for f in futs:
            f.result()
    rep = dict(images=len(rows), wall=time.perf_counter() - t0, entropy_device=ent, device=a.device, rows=rows)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))
    print(f"decoded {len(rows)} images in {rep['wall']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
