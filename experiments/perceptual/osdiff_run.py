"""OSDiff (cheesejiang/OSDiff) with real bitstreams, separate encode/decode processes.

Stream = 8-byte header (magic, version, quality, original h/w) + the authors' write_body() payload.
Images are padded like the authors' inference (utils pad to 64) and cropped back; rate is counted on
the original pixel count. The one-step sampler starts from x_T ~ N(0, I): the decoder draws it from a
fixed seed (no bits), so encode-side and fresh-process decodes must match exactly.
SD 2.1-base comes from a safetensors copy (no pickle); OpenCLIP ViT-H/14 from a local safetensors file.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CKPT = {1: 'lambda_1.ckpt', 2: 'lambda_2.ckpt'}
HEADER = struct.Struct('>4sBBHH')
UPSTREAM = '2d772c1b73078257568a8e27894390d69dd1ee61'
NOISE_SEED = 231  # authors' default --seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def build(a):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    import types
    sys.path.insert(0, str(a.upstream))
    # pyiqa is only used for training-time validation metrics; inference needs none of it.
    sys.modules.setdefault('pyiqa', types.ModuleType('pyiqa'))
    work = a.out / 'cwd'  # the model opens a TensorBoard writer in the working directory
    work.mkdir(exist_ok=True)
    os.chdir(work)
    import open_clip.factory as ocf
    _load = ocf.load_state_dict
    # open_clip 2.22 always torch.load()s; read the safetensors copy without unpickling.
    # (The HF local-dir file is a symlink into the cache, so test the resolved target we were given.)
    target = str(a.openclip.resolve())
    ocf.load_state_dict = lambda path, map_location='cpu': load_file(str(path)) if str(Path(path).resolve()) == target else _load(path, map_location)
    from utils.common import instantiate_from_config
    cfg = OmegaConf.load(a.upstream / 'configs/model/osdiff.yaml')
    cfg.params.cond_stage_config.params.version = str(a.openclip)
    cfg.params.calculate_metrics = {}
    # sync_path would torch.load the SD pickle to seed control/discriminator weights for training; the
    # released lambda checkpoint already carries the control module, and the discriminator is unused.
    cfg.params.sync_path = None
    model = instantiate_from_config(cfg)
    state = load_file(str(a.sd))
    lc = torch.load(a.weights_dir / CKPT[a.quality], map_location='cpu', weights_only=True)['state_dict']
    state.update(lc)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Same strict=False as the authors, but only the training-time discriminator and logvar may be absent.
    other = [k for k in missing if not k.startswith('guidance_model.') and k != 'logvar']
    if other:
        raise ValueError(f'Inference weights missing from SD + lambda checkpoint: {other[:10]} (total {len(other)})')
    model.preprocess_model.update(force=True)
    model.freeze()
    model.cuda().eval()
    return model, dict(missing=len(missing), missing_sample=missing[:20], unexpected=len(unexpected),
                       unexpected_sample=unexpected[:20])


def stage(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))
    from half_vram import install
    stop = install(a.out / f'q{a.quality}-{a.stage}.vram.jsonl')
    import numpy as np
    import torch
    import einops
    from PIL import Image
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    t_load = time.perf_counter()
    model, load_report = build(a)
    load_s = time.perf_counter() - t_load
    from utils.utils import write_body, read_body
    from model.spaced_sampler import SpacedSampler
    qdir = a.out / f'q{a.quality}'
    if a.stage == 'encode':
        (qdir / 'streams').mkdir(parents=True, exist_ok=False)
        (qdir / 'direct').mkdir()
        files = sorted(a.data.glob('*.png'))[:a.limit or None]
    else:
        (qdir / 'recon').mkdir(exist_ok=False)
        files = sorted((qdir / 'streams').glob('*.bin'))
    records = []
    for path in files:
        torch.cuda.empty_cache()  # large 2K images fragment the capped allocator otherwise
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if a.stage == 'encode':
                with Image.open(path) as im:
                    pixels = np.array(im.convert('RGB'))
                h, w = pixels.shape[:2]
                x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
                x = torch.nn.functional.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode='reflect' if min(h, w) > 64 else 'replicate')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ref = model.encode_first_stage(x * 2 - 1).mode() * model.scale_factor
                out = model.preprocess_model.compress(ref, None)
                buf = io.BytesIO()
                write_body(buf, out['shape'], out['strings'])
                torch.cuda.synchronize()
                encode_s = time.perf_counter() - t0
                blob = HEADER.pack(b'OSDF', 1, a.quality, h, w) + buf.getvalue()
                stream = qdir / 'streams' / (path.stem + '.bin')
                stream.write_bytes(blob)
                del x, ref
            else:
                stream = path
            blob = stream.read_bytes()
            magic, version, q, h, w = HEADER.unpack_from(blob)
            if magic != b'OSDF' or version != 1 or q != a.quality:
                raise ValueError('Bad stream header')
            hp, wp = h + (-h) % 64, w + (-w) % 64
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            strings, shape = read_body(io.BytesIO(blob[HEADER.size:]))
            c_latent = model.preprocess_model.decompress(strings, shape)
            cond = {'c_latent': [c_latent], 'c_crossattn': [model.get_learned_conditioning([''])]}
            lshape = (1, 4, hp // 8, wp // 8)
            gen = torch.Generator(device='cuda').manual_seed(NOISE_SEED)
            x_T = torch.randn(lshape, device='cuda', dtype=torch.float32, generator=gen)
            samples = SpacedSampler(model, var_type='fixed_small').sample(
                1, lshape, cond, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                cond_fn=None, x_T=x_T)
            img_t = ((model.decode_first_stage(samples) + 1) / 2).clamp(0, 1)[:, :, :h, :w]
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - t0
            if not torch.isfinite(img_t).all():
                raise ValueError('Non-finite output')
            img = (einops.rearrange(img_t, 'b c h w -> b h w c')[0] * 255).round().byte().cpu().numpy()
        Image.fromarray(img).save(qdir / ('direct' if a.stage == 'encode' else 'recon') / (stream.stem + '.png'), compress_level=0)
        rec = dict(name=stream.stem, quality=a.quality, width=w, height=h, bytes=len(blob), bpp=8 * len(blob) / (h * w),
                   stream_sha256=sha(stream), pixel_sha256=hashlib.sha256(img.tobytes()).hexdigest(), decode_s=decode_s,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        if a.stage == 'encode':
            rec['encode_s'] = encode_s
        records.append(rec)
        print(json.dumps(rec), flush=True)
    write_json(a.out / f'q{a.quality}-{a.stage}.json', dict(
        quality=a.quality, checkpoint=CKPT[a.quality], weights_sha256=sha(a.weights_dir / CKPT[a.quality]),
        sd_sha256=sha(a.sd), upstream=UPSTREAM, load_report=load_report, load_s=load_s, noise_seed=NOISE_SEED,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), fp32=True, images=records))
    stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['encode', 'decode'], required=True)
    p.add_argument('--quality', type=int, choices=sorted(CKPT), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--sd', type=Path, required=True)
    p.add_argument('--openclip', type=Path, required=True)
    p.add_argument('--weights-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    for k in ('upstream', 'sd', 'openclip', 'weights_dir', 'out', 'data'):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    stage(a)


if __name__ == '__main__':
    main()
