#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "$0")/../.." && pwd)"
artifact=$1
out=$2
python=$3
upstream="$project/refs/repos/MLIC"
revision=4aa9d2a512eb9db382172058e6c278fb0d376e17
source="$artifact/perceptual-full-20261109/clic30-budget-study"
mixed="$artifact/multirate-20261109"

test "$(git -C "$upstream" rev-parse HEAD)" = "$revision"
mkdir "$out"
mkdir -p "$out/decoder/weights" "$out/native"
git clone --quiet --no-hardlinks --single-branch "$upstream" "$out/decoder/upstream"
test "$(git -C "$out/decoder/upstream" rev-parse HEAD)" = "$revision"
for file in clic_mixed_decode.py geometry_codec.py probe.py stream.py; do
  cp "$project/experiments/mlicpp/$file" "$out/decoder/"
done
cp "$mixed/div2k160-q2-decoder-single-vs-mixed-1000steps/mixed/last.pt" "$out/decoder/mixed.pt"
cat > "$out/decoder/decode" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_CUDNN_V8_API_DISABLED=1
exec "${MLIC_PYTHON:-python3}" "$root/clic_mixed_decode.py" "$@"
SH
chmod +x "$out/decoder/decode"
"$python" - "$artifact" "$source" "$mixed" "$out" <<'PY'
import json
from pathlib import Path
import sys
import zipfile
import torch

artifact, source, mixed, out = map(Path, sys.argv[1:])
name = 'mlicpp-mse-q2-legacy.pth.tar'
state = torch.load(artifact/'weights'/name, map_location='cpu', weights_only=False)
torch.save({'state_dict': state['state_dict']}, out/'decoder'/'weights'/name)
del state

original = source/'budgets/bpp-0.075/perceptual.zip'
with zipfile.ZipFile(original) as src, zipfile.ZipFile(out/'native/bs.zip', 'w') as dst:
    for member in src.infolist():
        data = src.read(member)
        if member.filename == 'manifest.json':
            manifest = json.loads(data)
            assert manifest['decoder'] == 1 and len(manifest['images']) == 30
            manifest['decoder'] = 2
            data = json.dumps(manifest, separators=(',', ':')).encode()
        dst.writestr(member, data)
assert (out/'native/bs.zip').stat().st_size == original.stat().st_size

rows = json.loads((mixed/'clic30-frozen-budget-20261111/per_image.json').read_text())
expected = {r['name']: r['pixel_sha256'] for r in rows if r['model']=='mixed' and r['target_bpp']==.075}
assert len(expected) == 30
(out/'expected.json').write_text(json.dumps(expected, indent=2))
root = out/'decoder'
with zipfile.ZipFile(out/'decoder.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(root))
assert (out/'decoder.zip').stat().st_size <= 4_000_000_000
print('decoder bytes:', (out/'decoder.zip').stat().st_size)
print('bs bytes:', (out/'native/bs.zip').stat().st_size)
PY
