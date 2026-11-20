"""OSCAR (NeurIPS 2025, jp-guo/OSCAR) with real bitstreams, separate encode/decode processes.

The authors' test script resizes images to multiples of 128 and reports the nominal bpp of the VQ
configuration; nothing is entropy coded or written. Here:
  * the image is edge-padded (replicate) to a multiple of 64 and the output is cropped back;
  * the stream holds a 12-byte header, the 4 per-channel latent norms that OSCAR's decoder rescales by
    (float32; derived from the uncompressed latent, so they are side information and are counted), and
    the VQ indices packed at ceil(log2 K) bits each;
  * the SD-VAE encoder uses the posterior mean instead of a random sample, so encoding is deterministic.
Layout matches experiments/elic so experiments/perceptual/score.py can score it.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEVELS = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}  # q -> index into OSCAR.bpps (0.0430, 0.0625, 0.0781, 0.0937, 0.1250)
HEADER = struct.Struct('>4sBBHH4f')
UPSTREAM = 'cb23f4e1911aba746dc5a0d88b457876a0b06d66'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def pack_bits(values, width):
    import numpy as np
    v = np.asarray(values, dtype=np.uint32)
    bits = ((v[:, None] >> np.arange(width - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def unpack_bits(data, width, count):
    import numpy as np
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:count * width].reshape(count, width)
    return (bits.astype(np.uint32) << np.arange(width - 1, -1, -1, dtype=np.uint32)).sum(1).astype(np.int64)


def build(a):
    import torch
    sys.path.insert(0, str(a.upstream))
    os.chdir(a.upstream)
    from diffusion.oscar import OSCAR
    args = argparse.Namespace(pretrained_model_name_or_path=str(a.sd), lora_rank=16, hyper_dim=320)
    model = OSCAR(args)
    model.set_eval()
    # Author checkpoint is a plain tensor dict (verified loadable with weights_only=True).
    ck = torch.load(a.weights, map_location='cpu', weights_only=True)
    model.load_ckpt(ck)
    chunk_vae_attention(model.vae)
    return model


def chunk_vae_attention(vae, rows=4096):
    """Exact attention in query chunks for the VAE mid-blocks.

    The VAE attention has one 512-wide head, which PyTorch 2.0 runs with the math kernel: a 2K image
    needs a 7.6 GiB score matrix. Each softmax row only depends on its own query, so splitting queries
    gives the same result up to float rounding (unlike spatial tiling, which changes the output).
    """
    import torch.nn.functional as F
    from diffusers.models.attention_processor import AttnProcessor2_0

    class Chunked(AttnProcessor2_0):
        def __call__(self, *args, **kwargs):
            sdpa = F.scaled_dot_product_attention

            def chunked(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
                if attn_mask is not None or is_causal or q.shape[-2] <= rows:
                    return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
                return torch.cat([sdpa(q[..., i:i + rows, :], k, v, dropout_p=dropout_p)
                                  for i in range(0, q.shape[-2], rows)], dim=-2)
            F.scaled_dot_product_attention = chunked
            try:
                return super().__call__(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = sdpa

    import torch
    vae.set_attn_processor(Chunked())


def geometry(model, idx, h, w):
    he = model.hyper_encoders[idx]
    k = he.quantizer.codebook_size
    width = max(1, math.ceil(math.log2(k)))
    hp, wp = h + (-h) % 64, w + (-w) % 64
    ss = he.config.cfg_ss
    down = {64: 8, 32: 16, 16: 32, 8: 64}[ss]
    return k, width, hp, wp, hp // down, wp // down


def stage(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))
    from half_vram import install
    stop = install(a.out / f'q{a.quality}-{a.stage}.vram.jsonl')
    import numpy as np
    import torch
    from PIL import Image
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    t_load = time.perf_counter()
    model = build(a)
    load_s = time.perf_counter() - t_load
    idx = LEVELS[a.quality]
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
        with torch.inference_mode():
            if a.stage == 'encode':
                with Image.open(path) as im:
                    pixels = np.array(im.convert('RGB'))
                h, w = pixels.shape[:2]
                k, width, hp, wp, th, tw = geometry(model, idx, h, w)
                x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
                x = torch.nn.functional.pad(x, (0, wp - w, 0, hp - h), mode='replicate') * 2 - 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                latents = model.vae.encode(x).latent_dist.mean * model.vae.config.scaling_factor
                norms = torch.norm(latents.flatten(2), p=2, dim=-1)[0].float().cpu().tolist()
                he = model.hyper_encoders[idx]
                z = he.backbone(latents).permute(0, 2, 3, 1)
                b, zh, zw, c = z.shape
                if (zh, zw) != (th, tw):
                    raise ValueError(f'Unexpected token grid {(zh, zw)} vs {(th, tw)}')
                _, indices, _ = he.quantizer(z.reshape(b, -1, c))
                torch.cuda.synchronize()
                encode_s = time.perf_counter() - t0
                blob = HEADER.pack(b'OSCR', 1, a.quality, h, w, *norms) + pack_bits(indices[0].cpu().numpy(), width)
                stream = qdir / 'streams' / (path.stem + '.bin')
                stream.write_bytes(blob)
                del x, latents, z
            else:
                stream = path
            blob = stream.read_bytes()
            magic, version, q, h, w, *norms = HEADER.unpack_from(blob)
            if magic != b'OSCR' or version != 1 or q != a.quality:
                raise ValueError('Bad stream header')
            k, width, hp, wp, th, tw = geometry(model, idx, h, w)
            payload = blob[HEADER.size:]
            if len(payload) != math.ceil(th * tw * width / 8):
                raise ValueError('Payload length does not match the token grid')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ids = torch.from_numpy(unpack_bits(payload, width, th * tw)).cuda()[None]
            if int(ids.max()) >= k:
                raise ValueError('Index outside codebook')
            he = model.hyper_encoders[idx]
            zq = he.quantizer.get_output_from_indices(ids).reshape(1, th, tw, -1).permute(0, 3, 1, 2)
            z_hat = he.proj(zq)
            a_z = torch.tensor(norms, device=z_hat.device, dtype=z_hat.dtype).view(1, 4, 1)
            b_, c_, lh, lw = z_hat.shape
            z_hat = (z_hat.flatten(2) * a_z / torch.norm(z_hat.flatten(2), p=2, dim=-1, keepdim=True)).reshape(b_, c_, lh, lw)
            t = model.timesteps[idx]
            pred = model.unet(z_hat, t, encoder_hidden_states=model.embeddings.unsqueeze(0).float()).sample
            x0 = model.noise_scheduler.step(pred, t, z_hat, return_dict=True).pred_original_sample
            out = model.vae.decode(x0 / model.vae.config.scaling_factor).sample.clamp(-1, 1)[:, :, :h, :w]
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - t0
            if not torch.isfinite(out).all():
                raise ValueError('Non-finite output')
            img = ((out[0] + 1) / 2).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(qdir / ('direct' if a.stage == 'encode' else 'recon') / (stream.stem + '.png'), compress_level=0)
        rec = dict(name=stream.stem, quality=a.quality, width=w, height=h, bytes=len(blob), bpp=8 * len(blob) / (h * w),
                   nominal_bpp=float(model.bpps[idx]), stream_sha256=sha(stream),
                   pixel_sha256=hashlib.sha256(img.tobytes()).hexdigest(), decode_s=decode_s,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        if a.stage == 'encode':
            rec['encode_s'] = encode_s
        records.append(rec)
        print(json.dumps(rec), flush=True)
    write_json(a.out / f'q{a.quality}-{a.stage}.json', dict(
        quality=a.quality, level_index=idx, nominal_bpp=float(model.bpps[idx]), upstream=UPSTREAM,
        weights_sha256=sha(a.weights), load_s=load_s, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        fp32=True, vae_posterior='mean', images=records))
    stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['encode', 'decode'], required=True)
    p.add_argument('--quality', type=int, choices=sorted(LEVELS), required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--sd', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    for k in ('upstream', 'sd', 'weights', 'out', 'data'):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    stage(a)


if __name__ == '__main__':
    main()
