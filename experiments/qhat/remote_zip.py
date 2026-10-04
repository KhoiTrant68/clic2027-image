"""Read single members of a large remote .zip over HTTP range requests (no full download).

The CLIC perceptual crop archives are 46-135 GiB; a Q-hat fit needs only the crops of a few thousand
sampled questions. zipfile only needs a seekable file object, so `HttpFile` serves reads from a small
block cache filled by `Range:` requests. Works with ZIP64 (zipfile handles it).

    with RemoteZip(url) as z:
        names = z.namelist()
        png_bytes = z.read(names[0])
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from collections import OrderedDict

BLOCK = 1 << 20  # 1 MiB


class HttpFile(io.RawIOBase):
    def __init__(self, url: str, block: int = BLOCK, cache_blocks: int = 64, retries: int = 5):
        self.url, self.block, self.cache_blocks, self.retries = url, block, cache_blocks, retries
        self.pos, self.cache, self.bytes_fetched = 0, OrderedDict(), 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")
            if r.status != 206 or not cr:
                raise OSError(f"{url}: server does not support range requests")
            self.size = int(cr.split("/")[-1])

    # -------------------------------------------------------------- io.RawIOBase
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 4 * self.block:  # big member: one direct request, no caching
            out = self._get(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            while len(out) < n:
                b, off = divmod(self.pos + len(out), self.block)
                data = self._block(b)
                out += data[off:off + n - len(out)]
            out = bytes(out)
        self.pos += len(out)
        return out

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    # -------------------------------------------------------------- fetching
    def _block(self, b):
        if b in self.cache:
            self.cache.move_to_end(b)
            return self.cache[b]
        lo = b * self.block
        data = self._get(lo, min(self.size, lo + self.block) - 1)
        self.cache[b] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def _get(self, lo, hi):
        for k in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != hi - lo + 1:
                    raise OSError(f"short read {len(data)} != {hi - lo + 1}")
                self.bytes_fetched += len(data)
                return data
            except OSError:
                if k == self.retries - 1:
                    raise
                time.sleep(2 ** k)


class RemoteZip(zipfile.ZipFile):
    """zipfile.ZipFile over HTTP. Read members in archive order (sort by header offset) to reuse the cache."""

    def __init__(self, url: str, **kw):
        self.http = HttpFile(url, **kw)
        super().__init__(self.http)

    def offsets(self):
        return {i.filename: i.header_offset for i in self.infolist()}
