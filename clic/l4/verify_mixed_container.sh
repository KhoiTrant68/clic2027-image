#!/usr/bin/env bash
set -euo pipefail

decoder="$(realpath "${1:?decoder.zip required}")"
bitstream="$(realpath "${2:?bs.zip required}")"
expected="$(realpath "${3:?expected.json required}")"
out="${4:?new output directory required}"
image="${CLIC_IMAGE:-clic-mlicpp:cu118}"
gpu="${CLIC_GPU:-0}"

docker info >/dev/null
docker image inspect "$image" >/dev/null
mkdir "$out"
out="$(realpath "$out")"
unzip -q "$decoder" -d "$out"
test -x "$out/decode"
cp "$bitstream" "$out/bs.zip"
cp "$expected" "$out/expected.json"
mount="type=bind,source=$out,target=/work"

docker run --rm --pull=never --network none --cpus=2 --memory=12g \
  --gpus "device=$gpu" --mount "$mount" -w /work --entrypoint /work/decode \
  "$image" --memory-mib 5120 --report decoded.json

docker run --rm -i --pull=never --network none --mount "$mount,readonly" \
  -w /work --entrypoint python3 "$image" - <<'PY'
import hashlib
import json
from pathlib import Path

from PIL import Image

expected = json.loads(Path('expected.json').read_text())
rows = json.loads(Path('decoded.json').read_text())['images']
files = {f.name: f for f in Path('images').iterdir()}
if set(expected) != set(files) or sorted(row['name'] for row in rows) != sorted(expected):
    raise SystemExit('Decoded image inventory mismatch')
for row in rows:
    with Image.open(files[row['name']]) as image:
        digest = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        if image.size != (row['width'], row['height']) or digest != expected[row['name']]:
            raise SystemExit(f"Pixel or dimension mismatch: {row['name']}")
print(f'Verified {len(rows)}/{len(expected)} PNGs')
PY
