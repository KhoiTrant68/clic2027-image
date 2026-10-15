"""Versioned MS-ILLM single-image framing. All dimensions and lengths are transmitted."""
import struct

HEADER = struct.Struct('>2sBB6I')


def pack(q, image_size, padded_size, latent, hyper):
    if q not in range(1, 7) or not latent or not hyper:
        raise ValueError('Invalid quality or empty entropy stream')
    h, w = image_size
    ph, pw = padded_size
    if not (0 < h <= ph <= 65536 and 0 < w <= pw <= 65536 and ph % 64 == pw % 64 == 0):
        raise ValueError('Invalid image dimensions')
    return HEADER.pack(b'MI', 1, q, h, w, ph, pw, len(latent), len(hyper)) + latent + hyper


def unpack(data):
    if len(data) < HEADER.size:
        raise ValueError('Truncated header')
    magic, version, q, h, w, ph, pw, nl, nh = HEADER.unpack_from(data)
    if magic != b'MI' or version != 1 or len(data) != HEADER.size + nl + nh:
        raise ValueError('Invalid framing or payload lengths')
    latent, hyper = data[HEADER.size:HEADER.size+nl], data[HEADER.size+nl:]
    # Reuse validation before exposing dimensions to an allocating decoder.
    pack(q, (h, w), (ph, pw), latent, hyper)
    return q, (h, w), (ph, pw), latent, hyper
