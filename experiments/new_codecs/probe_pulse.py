"""PULSE actual-bitstream probe with a half-VRAM guard; use torch 2.6 venv."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from half_vram import install


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pulse-perceptual", "pulse-l-perceptual"], required=True)
    ap.add_argument("--root", type=Path, default=Path.home() / "clic/research_20261015")
    ap.add_argument("--data", type=Path, default=Path.home() / "clic/work_full_gpu/bakeoff/valid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qualities", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / "vram.jsonl")
    sys.path.insert(0, str(a.root / "pulse_vendor"))
    import torch
    from PIL import Image
    from codec.checkpoint import load_model
    from codec.images import read_image, save_image
    from codec.runtime import configure_runtime, compress_image, decompress_image
    configure_runtime("cuda:0", 2)
    torch.manual_seed(0)
    start = time.monotonic()
    model = load_model(a.root / "weights/pulse" / a.model, "cuda:0")
    torch.cuda.synchronize()
    (a.out / "meta.json").write_text(json.dumps(dict(model=a.model, torch=torch.__version__,
        load_s=time.monotonic() - start, qualities=a.qualities, full_frame=True), indent=2))
    files = [p for p in sorted(a.data.glob("*.png")) if not a.only or any(p.name.startswith(s) for s in a.only)]
    if not files:
        raise ValueError("No selected images")
    for q in a.qualities:
        for p in files:
            dst = a.out / f"q{q}"
            dst.mkdir(exist_ok=True)
            record = dst / (p.stem + ".json")
            if record.exists() and json.loads(record.read_text()).get("status") == "ok":
                continue
            torch.cuda.reset_peak_memory_stats()
            image = read_image(p, "cuda:0")
            h, w = image.shape[-2:]
            data, enc_ms = compress_image(model, image, q)
            del image
            stream = dst / (p.stem + ".pulse")
            stream.write_bytes(data)
            reconstruction, dec_ms = decompress_image(model, stream.read_bytes())
            save_image(reconstruction, dst / p.name)
            del reconstruction
            if Image.open(dst / p.name).size != (w, h):
                raise ValueError("Wrong reconstruction dimensions")
            row = dict(model=a.model, q=q, name=p.stem, status="ok", bytes=len(data), pixels=h*w,
                       bpp=8*len(data)/(h*w), encode_s=enc_ms/1000, decode_s=dec_ms/1000,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       stream_sha256=hashlib.sha256(data).hexdigest())
            record.write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
    stop.set()


if __name__ == "__main__":
    main()
