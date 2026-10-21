"""AEIC pilot: author checkpoints, real streams, independent decoding, no color fix."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from half_vram import install


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpu_check(root):
    """Exercise the actual entropy extension and both codec checkpoints on CPU."""
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    sys.path.insert(0, str(root / "AEIC"))
    from src.codec.codec_practical import PixelCodec
    results = []
    for variant in ("SE", "ME"):
        model = PixelCodec("AEIC-" + variant).eval()
        weight = root / "weights" / f"AEIC_{variant}_ft2.pkl"
        state = torch.load(weight, map_location="cpu")
        codec_state = state["state_dict_codec"]
        extra = set(codec_state) - set(model.state_dict())
        # Author inference drops training-only distillation projections.
        allowed = {f"{module}.project.{kind}" for module in ("g_a", "g_s", "h_a", "h_s")
                   for kind in ("weight", "bias")}
        if not extra <= allowed:
            raise ValueError(f"Unexpected checkpoint keys: {extra - allowed}")
        model.load_state_dict({k: v for k, v in codec_state.items() if k not in extra}, strict=True)
        model.update(force=True)
        with torch.inference_mode():
            x = torch.rand(1, 3, 128, 128) * 2 - 1
            model.entropy_coder.reset()
            model.compress(x)
            stream = model.entropy_coder.get_encoded_stream()
            outputs = []
            for _ in range(2):
                model.entropy_coder.reset()
                model.entropy_coder.set_stream(stream)
                outputs.append(model.decompress((1, model.y_channel // 2, 1, 1), 4, 4))
            assert len(stream) > 0
            assert all(torch.isfinite(t).all() for t in outputs[0])
            assert all(torch.equal(x, y) for x, y in zip(*outputs))
        results.append(dict(variant=variant, weights_sha256=digest(weight),
                            stream_bytes=len(stream), repeat_latents_equal=True,
                            training_only_keys=sorted(extra)))
        print(json.dumps(results[-1]), flush=True)
    (root / "cpu_preflight.json").write_text(json.dumps(results, indent=2))


def queue_pilot(a):
    """Bounded resource wait for this single pilot; never touch other jobs."""
    if a.out.exists():
        raise FileExistsError("Queue output must be a new directory")
    a.out.mkdir(parents=True)
    status = a.out / "status.json"

    def report(state, **details):
        status.write_text(json.dumps(dict(state=state, updated_unix=time.time(),
                                         pid=os.getpid(), **details), indent=2))

    deadline = time.monotonic() + 6 * 3600
    streak = {}
    try:
        report("waiting_for_memory" if a.shared_small else "waiting_for_idle_gpu", expires_unix=time.time() + 6 * 3600)
        while time.monotonic() < deadline:
            raw = subprocess.check_output(["nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=10)
            selected = None
            for line in raw.strip().splitlines():
                gpu, total, used, utilization = [int(v.strip()) for v in line.split(",")]
                available = total - used >= 12288 if a.shared_small else used < 512 and utilization < 5
                streak[gpu] = streak.get(gpu, 0) + 1 if available else 0
                if streak[gpu] >= (1 if a.shared_small else 3) and selected is None:
                    selected = gpu
            if selected is not None:
                break
            time.sleep(60)
        else:
            report("expired_without_free_gpu")
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(selected), HF_HUB_OFFLINE="1")
        if a.shared_small:
            env["AEIC_SHARED_SMALL"] = "1"
        runs = []
        for variant in ("SE", "ME"):
            out = a.out / ("aeic_" + variant.lower() + "_ft2")
            runs.append(out)
            for stage in ("encode", "decode"):
                report("running", gpu=selected, variant=variant, stage=stage)
                cmd = [sys.executable, "-u", __file__, "--stage", stage, "--root", str(a.root),
                       "--variant", variant, "--out", str(out)]
                if stage == "encode":
                    cmd += ["--data", str(a.data), "--only", *a.only]
                with (a.out / f"{variant}_{stage}.log").open("w") as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=3600)
        report("scoring", gpu=selected)
        home = Path.home() / "clic"
        cmd = [str(home / "venv/bin/python"), str(a.root / "score.py"),
               "--repo", str(home / "clic2027-image"), "--baseline", str(home / "work_merge_1013_v0"),
               "--qhat", str(home / "inputs/qhat_v0/qhat/qhat_v0.json"),
               "--data", str(a.data), "--out", str(a.out / "scores"),
               "--runs", *map(str, runs), "--only", *a.only]
        with (a.out / "score.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        report("complete", gpu=selected)
    except Exception as e:
        report("failed", error=str(e))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["encode", "decode", "preflight", "queue", "pair"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--variant", choices=["ME", "SE"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--only", nargs="+", default=[])
    ap.add_argument("--shared-small", action="store_true", help="8 GiB allocator and smaller tiles on a shared GPU")
    a = ap.parse_args()
    if a.stage == "preflight":
        cpu_check(a.root)
        return
    if a.out is None:
        ap.error("--out is required")
    if a.stage == "queue":
        if a.data is None or not a.only:
            ap.error("Pilot queue requires --data and --only")
        queue_pilot(a)
        return
    if a.variant is None:
        ap.error("Codec stages require --variant")
    if a.stage == "pair":
        if a.data is None or a.out.exists():
            ap.error("pair requires --data and a new --out directory")
        a.out.mkdir(parents=True)
        status = a.out / "pair_status.json"
        try:
            for stage in ("encode", "decode"):
                status.write_text(json.dumps(dict(state=stage, pid=os.getpid(), updated_unix=time.time())))
                cmd = [sys.executable, "-u", __file__, "--stage", stage, "--root", str(a.root),
                       "--variant", a.variant, "--out", str(a.out)]
                if stage == "encode":
                    cmd += ["--data", str(a.data)]
                    if a.only:
                        cmd += ["--only", *a.only]
                with (a.out / f"{stage}.log").open("w") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            status.write_text(json.dumps(dict(state="complete", updated_unix=time.time())))
        except Exception as e:
            status.write_text(json.dumps(dict(state="failed", error=str(e), updated_unix=time.time())))
            raise
        return
    if (a.stage == "encode") != (a.data is not None):
        ap.error("Only encode receives --data; decoder must not receive source images")
    runtime = a.root / "AEIC"
    weight = a.root / "weights" / f"AEIC_{a.variant}_ft2.pkl"
    sd = a.root / "weights/sd-turbo"
    vae = a.root / "weights/AdcSR/weight/pretrained/halfDecoder.ckpt"
    for p in (runtime, weight, sd, vae):
        if not p.exists():
            raise FileNotFoundError(p)
    a.out.mkdir(parents=True, exist_ok=True)
    stop = install(a.out / f"{a.stage}.vram.jsonl")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    from AEIC_practical import AEIC
    from my_utils.compress_utils import my_write_body, my_read_body
    args = SimpleNamespace(codec_type="AEIC-" + a.variant, codec_path=str(weight),
        vae_decoder_path=str(vae), lora_rank_unet=32, merge_LoRA=True,
        compile_model=False, use_tiled_vae=True, use_tiled_unet=True,
        latent_tiled_size=192 if a.variant == "SE" else 96,
        latent_tiled_overlap=64 if a.variant == "SE" else 32, vae_decoder_tiled_size=160)
    if os.environ.get("AEIC_SHARED_SMALL") == "1":
        args.latent_tiled_size = 64
        args.latent_tiled_overlap = 16
        args.vae_decoder_tiled_size = 64
    config = dict(vars(args), weights_sha256=digest(weight), vae_sha256=digest(vae),
        torch=torch.__version__, color_fix=False, deterministic=True, dtype="float32")
    meta = a.out / "meta.json"
    if a.stage == "encode":
        if meta.exists():
            raise FileExistsError("Use a new output directory after changes or failure")
        meta.write_text(json.dumps(dict(model="aeic_" + a.variant.lower(), qualities=[2], config=config), indent=2))
    elif json.loads(meta.read_text())["config"] != config:
        raise ValueError("Encode/decode configuration mismatch")
    start = time.monotonic()
    net = AEIC(sd_path=str(sd), args=args).cuda().eval()
    net.unet.enable_xformers_memory_efficient_attention()
    net.codec.update(force=True)
    torch.cuda.synchronize()
    load_s = time.monotonic() - start
    dst = a.out / "q2"
    dst.mkdir(exist_ok=True)
    header = b"AE" + bytes([0 if a.variant == "ME" else 1, 2])

    def decode(path):
        with path.open("rb") as f:
            if f.read(4) != header:
                raise ValueError("Wrong AEIC envelope")
            strings, (h, w) = my_read_body(f)
            if f.read(1) or not (0 < h <= 16384 and 0 < w <= 16384):
                raise ValueError("Invalid stream")
        yh, yw = math.ceil(h / 64) * 2, math.ceil(w / 64) * 2
        shape = (1, net.codec.y_channel // 2, math.ceil(yh / 4), math.ceil(yw / 4))
        net.codec.entropy_coder.reset()
        net.codec.entropy_coder.set_stream(strings)
        image, seconds = net.decompress(shape, yh, yw)
        image = (image[:, :, :h, :w] * .5 + .5).float().cpu()
        if not torch.isfinite(image).all():
            raise ValueError("Nonfinite pixels")
        return to_pil_image(image[0].clamp(0, 1)), seconds, h, w

    records = []
    with torch.inference_mode():
        if a.stage == "encode":
            files = [p for p in sorted(a.data.glob("*.png"))
                     if not a.only or any(p.name.startswith(s) for s in a.only)]
            if not files or (a.only and len(files) != len(a.only)):
                raise ValueError("Expected one image per prefix")
            for p in files:
                t0 = time.monotonic()
                x = to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).cuda() * 2 - 1
                h, w = x.shape[-2:]
                x = F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode="reflect")
                net.codec.entropy_coder.reset()
                enc_s = net.compress(x)
                stream = dst / (p.stem + ".bin")
                with stream.open("wb") as f:
                    f.write(header)
                    my_write_body(f, [h, w], net.codec.entropy_coder.get_encoded_stream())
                del x
                image, dec_s, dh, dw = decode(stream)
                assert (dh, dw) == (h, w)
                image.save(stream.with_suffix(".png"), compress_level=0)
                rec = dict(status="ok", name=p.stem, bytes=stream.stat().st_size,
                    bpp=8 * stream.stat().st_size / (h * w), height=h, width=w,
                    encode_s=enc_s, decode_s=dec_s, wall_s=time.monotonic() - t0,
                    stream_sha256=digest(stream), pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                stream.with_suffix(".json").write_text(json.dumps(rec, indent=2))
                print(json.dumps(rec), flush=True)
                torch.cuda.empty_cache()
        else:
            independent = a.out / "independent"
            independent.mkdir(exist_ok=True)
            streams = sorted(dst.glob("*.bin"))
            if not streams:
                raise ValueError("No streams")
            for stream in streams:
                image, seconds, h, w = decode(stream)
                image.save(independent / (stream.stem + ".png"), compress_level=0)
                expected = json.loads(stream.with_suffix(".json").read_text())
                same = hashlib.sha256(image.tobytes()).hexdigest() == expected["pixel_sha256"]
                records.append(dict(name=stream.stem, same_pixels=same, decode_s=seconds))
                print(json.dumps(records[-1]), flush=True)
                torch.cuda.empty_cache()
            (a.out / "parity.json").write_text(json.dumps(records, indent=2))
            if not all(r["same_pixels"] for r in records):
                raise RuntimeError("Independent decoding failed pixel parity")
    (a.out / f"{a.stage}.timing.json").write_text(json.dumps(dict(load_s=load_s,
        total_s=time.monotonic() - start, max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved()), indent=2))
    stop.set()


if __name__ == "__main__":
    main()
