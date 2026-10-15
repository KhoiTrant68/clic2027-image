"""Actual-bitstream AFP-GIC/DC-VIC probe, bounded by half-device VRAM.

Uses public author runtime; quality/timing from this compatibility environment
is not claimed to reproduce their paper. Full-frame only; OOM is a failed point.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import sys
import time

from half_vram import install


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["afp", "dcvic"], required=True)
    ap.add_argument("--root", type=Path, default=Path.home() / "clic/research_20261015")
    ap.add_argument("--data", type=Path, default=Path.home() / "clic/work_full_gpu/bakeoff/valid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qualities", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / "vram.jsonl")
    sys.path.insert(0, str(a.root / "vendor"))
    import numpy as np
    import torch
    from PIL import Image
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    if a.model == "afp":
        runtime = a.root / "AFP_GIC/public_release/runtime"
        sys.path.insert(0, str(runtime))
        import eval_public_release as api
        config = api.load_infer_config(str(runtime / "config/afp_gic_release.yaml"), "cuda:0")
        weights = a.root / "weights/afp_gic_release.pth.tar"
        build, save = api.build_comp_model, api.img_utils.imwrite
    else:
        runtime = a.root / "DC_VIC"
        sys.path.insert(0, str(runtime))
        from src.models import build_comp_model
        from src.utils.options import BaseConfig
        from src.utils.img_utils import imwrite
        import src  # registers model components
        cfg, txt, _ = BaseConfig._file2dict_yaml(str(runtime / "config/dc_vic_patchgan.yaml"))
        cfg.update(is_train=False, device="cuda:0")
        # Released DC-VIC checkpoint includes the complete frozen VQGAN too.
        cfg["subnet"]["vq_model"]["ckpt_path"] = ""
        config = BaseConfig(cfg, cfg_text=txt)
        weights = a.root / "weights/dc_vic_patchgan.pth.tar"
        build, save = build_comp_model, imwrite
    t0 = time.monotonic()
    model = build(config).to("cuda:0")
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    state = {}
    # CompressAI 1.2.4 -> 1.2.8 renamed these ParameterLists. Preserve tensors,
    # then require a strict load: upstream strict=False silently leaves random weights.
    for key, value in ck["comp_model"].items():
        key = key.removeprefix("module.")
        for old, new in (("matrix", "matrices"), ("bias", "biases"), ("factor", "factors")):
            key = re.sub(r"\._" + old + r"(\d+)$", lambda m: "." + new + "." + m[1], key)
        if key in state:
            raise ValueError(f"Duplicate checkpoint key after rename: {key}")
        state[key] = value
    model.load_state_dict(state, strict=True)
    del ck, state
    model.entropy_model_z.update(force=False)
    model.codec_setup()
    model.eval()
    torch.cuda.synchronize()
    meta = {"model": a.model, "torch": torch.__version__, "load_s": time.monotonic() - t0,
            "runtime": str(runtime), "weights": str(weights), "qualities": a.qualities,
            "full_frame": True, "seed": 0}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    files = sorted(a.data.glob("*.png"))
    files = [p for p in files if not a.only or any(p.name.startswith(s) for s in a.only)]
    if not files:
        raise ValueError("No selected input images")
    for q in a.qualities:
        for p in files:
            dst = a.out / f"q{q}"
            dst.mkdir(exist_ok=True)
            record = dst / (p.stem + ".json")
            if record.exists() and json.loads(record.read_text()).get("status") == "ok":
                continue
            row = {"model": a.model, "q": q, "name": p.stem}
            try:
                x = np.array(Image.open(p).convert("RGB"))
                h, w = x.shape[:2]
                tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    encoded = model.compress(tensor, quality_ind=q)
                    torch.cuda.synchronize()
                    enc_s = time.monotonic() - start
                    parts = encoded["string_list"]
                    blob = b"".join(struct.pack("<I", len(part)) + part for part in parts)
                    (dst / (p.stem + ".bin")).write_bytes(blob)
                    # Read back saved bytes; decoding does not receive original tensors.
                    data = (dst / (p.stem + ".bin")).read_bytes()
                    parts, pos = [], 0
                    while pos < len(data):
                        size = struct.unpack_from("<I", data, pos)[0]
                        pos += 4
                        if size <= 0 or pos + size > len(data):
                            raise ValueError("Invalid stream length")
                        parts.append(data[pos:pos + size])
                        pos += size
                    assert pos == len(data) and len(parts) == 3
                    del encoded, tensor
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    reconstruction, _, _ = model.decompress(parts)
                    torch.cuda.synchronize()
                    dec_s = time.monotonic() - start
                    save(str(dst / p.name), reconstruction)
                y = np.array(Image.open(dst / p.name).convert("RGB"))
                if y.shape != x.shape:
                    raise ValueError(f"Wrong output shape {y.shape} != {x.shape}")
                row.update(status="ok", bytes=len(blob), pixels=h * w, bpp=8 * len(blob) / (h * w),
                           encode_s=enc_s, decode_s=dec_s,
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                           stream_sha256=hashlib.sha256(blob).hexdigest())
                del reconstruction
            except torch.OutOfMemoryError as e:
                row.update(status="oom_under_cap", error=str(e))
                record.write_text(json.dumps(row, indent=2))
                print(json.dumps(row), flush=True)
                # Exit rather than continuing with possibly live activation references.
                raise
            record.write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
    stop.set()


if __name__ == "__main__":
    main()
