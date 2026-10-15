"""Standalone offline decoder shipped in the PULSE research deployment package."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import socket
import sys
import time


def main():
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--gpu-memory-guard", action="store_true", required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    a.output.mkdir(parents=True, exist_ok=False)
    a.report.parent.mkdir(parents=True, exist_ok=True)

    # Refuse any Python TCP connection, including accidental checkpoint downloads.
    def offline(*args, **kwargs):
        raise RuntimeError("Network access disabled in offline decoder")
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.create_connection = offline
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        h = hashlib.sha256()
        with (root / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    verified = time.perf_counter()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "runtime"))
    from half_vram import install
    stop = install(a.report.with_suffix(".vram.jsonl"))
    import torch
    from codec.checkpoint import load_model
    from codec.runtime import configure_runtime, decompress_image
    from codec.images import to_uint8
    from PIL import Image
    from pulse_codec import unpack
    configure_runtime("cuda:0", 2)
    load_start = time.perf_counter()
    model = load_model(root / "weights", "cuda:0")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start
    files = sorted(a.input.glob("*.bin"))
    if not files:
        raise ValueError("No bitstreams")
    records = []
    loop_start = time.perf_counter()
    for p in files:
        t0 = time.perf_counter()
        payload = unpack(p.read_bytes())
        read_s = time.perf_counter() - t0
        reconstruction, ms = decompress_image(model, payload)
        h, w = reconstruction.shape[-2:]
        t0 = time.perf_counter()
        pixels = to_uint8(reconstruction)[0].permute(1, 2, 0).cpu().numpy()
        # Decoded PNG bytes are not compressed-stream rate. Avoid spending most
        # of the decode budget on lossless output recompression (35 s/30 images).
        Image.fromarray(pixels).save(a.output / (p.stem + ".png"), compress_level=0)
        records.append(dict(name=p.stem, width=w, height=h, bytes=p.stat().st_size,
                            read_s=read_s, decode_s=ms / 1000,
                            write_s=time.perf_counter() - t0))
        del reconstruction
    report = dict(n=len(files), torch=torch.__version__, network="Python TCP disabled", png_compress_level=0,
                  verify_s=verified-started, load_s=load_s,
                  loop_s=time.perf_counter()-loop_start, inside_process_s=time.perf_counter()-started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, images=records)
    a.report.write_text(json.dumps(report, indent=2))
    stop.set()
    print(json.dumps({k: v for k, v in report.items() if k != "images"}), flush=True)


if __name__ == "__main__":
    main()
