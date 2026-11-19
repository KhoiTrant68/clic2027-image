"""Single-image MLIC++ envelope: quality, architecture, original/latent sizes, two strings."""
import struct

HEADER = struct.Struct('>4sBBB4I')
LENGTH = struct.Struct('>I')


def pack(quality, legacy, size, shape, strings):
    h, w = size
    zh, zw = shape
    if type(quality) is not int or quality not in range(1, 7) or type(legacy) is not bool:
        raise ValueError('Invalid quality or architecture flag')
    if not (0 < h <= 65536 and 0 < w <= 65536 and h * w <= 100_000_000):
        raise ValueError('Invalid original dimensions')
    if (zh, zw) != ((h + 63) // 64, (w + 63) // 64):
        raise ValueError('Latent shape does not match padding to 64 pixels')
    if len(strings) != 2 or any(len(group) != 1 or not group[0] for group in strings):
        raise ValueError('Expected two non-empty strings for one image')
    data = HEADER.pack(b'ML++', 1, quality, int(legacy), h, w, zh, zw)
    return data + b''.join(LENGTH.pack(len(group[0])) + group[0] for group in strings)


def unpack(data):
    if len(data) < HEADER.size:
        raise ValueError('Truncated MLIC++ header')
    magic, version, quality, flags, h, w, zh, zw = HEADER.unpack_from(data)
    if magic != b'ML++' or version != 1 or flags not in (0, 1):
        raise ValueError('Unknown MLIC++ magic/version/flags')
    strings, pos = [], HEADER.size
    for _ in range(2):
        if pos + LENGTH.size > len(data):
            raise ValueError('Truncated payload length')
        n = LENGTH.unpack_from(data, pos)[0]
        pos += LENGTH.size
        if not n or pos + n > len(data):
            raise ValueError('Invalid entropy payload length')
        strings.append([data[pos:pos+n]])
        pos += n
    if pos != len(data):
        raise ValueError('Trailing bytes')
    result = quality, bool(flags), (h, w), (zh, zw), strings
    pack(*result)
    return result
