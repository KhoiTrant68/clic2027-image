"""Flatten the official validation ZIP and verify the existing full30 image inventory."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--allocation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = {r['name'] for r in csv.DictReader(a.allocation.open(encoding='utf-8-sig'))
                if r['cand'] == 'mix' and float(r['rate']) == .075}
    if len(expected) != 30:
        raise ValueError('Expected the established 30-image baseline inventory')
    blobs = {}
    with zipfile.ZipFile(a.zip) as z:
        for member in z.infolist():
            path = Path(member.filename)
            if path.suffix.lower() != '.png' or path.name.startswith('._') or '__MACOSX' in path.parts:
                continue
            if path.stem in blobs or path.stem not in expected:
                raise ValueError(f'Unexpected/duplicate image: {member.filename}')
            blobs[path.stem] = z.read(member)
    if set(blobs) != expected:
        raise ValueError('Downloaded image list differs from baseline')
    records = []
    for name, blob in sorted(blobs.items()):
        with Image.open(io.BytesIO(blob)) as im:
            im.load()
            records.append(dict(name=name, width=im.width, height=im.height,
                                sha256=hashlib.sha256(blob).hexdigest()))
    pixels = sum(r['width'] * r['height'] for r in records)
    if pixels != 86245376:
        raise ValueError(f'Wrong total pixel count: {pixels}')
    a.out.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        (a.out / (name + '.png')).write_bytes(blob)
    (a.out / 'manifest.json').write_text(json.dumps(dict(images=records, pixels=pixels), indent=2))
    print(json.dumps(dict(images=len(records), pixels=pixels, output=str(a.out))))


if __name__ == '__main__':
    main()
