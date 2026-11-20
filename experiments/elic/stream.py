"""Single-image ELIC envelope: quality, original/hyper-latent sizes, z string and 5x2 y strings.

ELIC (VincentChandelier re-implementation) codes y in 5 uneven channel slices, each as an
anchor and a non-anchor checkerboard string, plus one z string: 11 payloads per image.
"""
import struct

HEADER = struct.Struct('>4sBB4H')
SLICES = 5


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(data, pos):
    n = shift = 0
    while True:
        if pos >= len(data) or shift > 28:
            raise ValueError('Truncated/oversized length')
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, pos


def pack(quality, size, shape, strings):
    h, w = size
    zh, zw = shape
    if type(quality) is not int or quality not in range(1, 7):
        raise ValueError('Invalid quality')
    if not (0 < h <= 65535 and 0 < w <= 65535):
        raise ValueError('Invalid original dimensions')
    if (zh, zw) != ((h + 63) // 64, (w + 63) // 64):
        raise ValueError('Latent shape does not match padding to 64 pixels')
    y_strings, z_strings = strings
    if len(z_strings) != 1 or len(y_strings) != SLICES or any(len(s) != 2 for s in y_strings):
        raise ValueError('Expected one z string and 5x(anchor, non-anchor) y strings')
    payloads = [z_strings[0]] + [part[0] for s in y_strings for part in s]
    if any(type(p) is not bytes for p in payloads):
        raise ValueError('Expected one image per entropy string list')
    data = HEADER.pack(b'ELIC', 1, quality, h, w, zh, zw)
    return data + b''.join(_varint(len(p)) + p for p in payloads)


def unpack(data):
    if len(data) < HEADER.size:
        raise ValueError('Truncated ELIC header')
    magic, version, quality, h, w, zh, zw = HEADER.unpack_from(data)
    if magic != b'ELIC' or version != 1:
        raise ValueError('Unknown ELIC magic/version')
    payloads, pos = [], HEADER.size
    for _ in range(1 + 2 * SLICES):
        n, pos = _read_varint(data, pos)
        if pos + n > len(data):
            raise ValueError('Invalid entropy payload length')
        payloads.append(data[pos:pos + n])
        pos += n
    if pos != len(data):
        raise ValueError('Trailing bytes')
    z = [payloads[0]]
    y = [[[payloads[1 + 2 * i]], [payloads[2 + 2 * i]]] for i in range(SLICES)]
    result = quality, (h, w), (zh, zw), [y, z]
    if pack(*result) != data:
        raise ValueError('Non-canonical ELIC stream')
    return result
