"""Small, image-disjoint DIV2K pilot; verify mirror files against official ZIP CRCs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

OFFICIAL = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
MIRROR = 'https://huggingface.co/datasets/mAiello00/DIV2K'
FALLBACK = 'https://huggingface.co/datasets/dwchoi/div2k800-ffhq1000'


def get(url, byte_range=None):
    req = urllib.request.Request(url, headers={'Range': 'bytes='+byte_range} if byte_range else {})
    with urllib.request.urlopen(req, timeout=90) as response:
        if byte_range and response.status != 206:
            raise ValueError('Server ignored byte range')
        return response.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'manifest.json').exists():
        raise ValueError('Refuse to overwrite an existing data manifest')
    tail = get(OFFICIAL, '-65536')
    at = tail.rfind(b'PK\x05\x06')
    if at < 0:
        raise ValueError('Missing ZIP central directory')
    eocd = struct.unpack_from('<4s4H2IH', tail, at)
    size, offset = eocd[5:7]
    directory = get(OFFICIAL, f'{offset}-{offset+size-1}')
    # Relocate only the directory offset in a synthetic ZIP; no image extraction.
    end = bytearray(tail[at:])
    struct.pack_into('<I', end, 16, 0)
    with zipfile.ZipFile(io.BytesIO(directory + end)) as z:
        crcs = {Path(i.filename).name: i.CRC for i in z.infolist() if i.filename.endswith('.png')}
    if len(crcs) != 800:
        raise ValueError(f'Wrong official image inventory: {len(crcs)}')
    revision = json.loads(get('https://huggingface.co/api/datasets/mAiello00/DIV2K'))['sha']
    fallback_revision = json.loads(get('https://huggingface.co/api/datasets/dwchoi/div2k800-ffhq1000'))['sha']
    ids = list(range(1, 801))
    random.Random(1984).shuffle(ids)
    assignments = [(split, image_id) for split, group in
                   [('train', ids[:160]), ('validation', ids[160:192]), ('test', ids[192:224])]
                   for image_id in group]

    def fetch(item):
        split, image_id = item
        name = f'{image_id:04d}.png'
        path = a.out / 'images' / name
        path.parent.mkdir(exist_ok=True)
        content = path.read_bytes() if path.exists() else None
        if content is None:
            errors = []
            for mirror, rev in [(MIRROR, revision), (FALLBACK, fallback_revision)]:
                try:
                    content = get(f'{mirror}/resolve/{rev}/DIV2K_train_HR/{name}')
                    break
                except (OSError, urllib.error.HTTPError) as error:
                    errors.append(str(error))
                    time.sleep(1)
            if content is None:
                raise RuntimeError(f'{name}: both mirrors failed: {errors}')
        if zlib.crc32(content) != crcs[name]:
            raise ValueError(f'Mirror differs from official ZIP CRC: {name}')
        path.write_bytes(content)
        return dict(split=split, name=name, crc32=crcs[name], sha256=hashlib.sha256(content).hexdigest())

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, assignments):
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=len(assignments), **row)), flush=True)
    manifest = dict(dataset='DIV2K train HR subset', official=OFFICIAL, mirror=MIRROR,
                    mirror_revision=revision, split_seed=1984, images=rows,
                    fallback_mirror=FALLBACK, fallback_revision=fallback_revision,
                    split_warning='disjoint by source image; pretrained MLIC training overlap not audited',
                    license='academic research only; see https://data.vision.ee.ethz.ch/cvl/DIV2K/')
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
