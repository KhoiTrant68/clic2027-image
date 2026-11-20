#!/usr/bin/env bash
# Decode a bs.zip with a packaged GC decoder the way the CLIC server does (2 CPUs, 12 GB RAM, one GPU, no
# network), then check the outputs and time the run. Run on a machine with an NVIDIA L4 and Docker.
#
#   bash scripts/gc/l4_verify.sh RUN/package/decoder.zip RUN/encode/bpp-0.075/bs.zip REF_DIR [OUT_DIR]
#
# REF_DIR holds the original PNGs (to check names and sizes; and pixel hashes if REF_DIR/expected.json exists,
# a {"name": sha256(RGB bytes)} map written by a previous decode you trust).
set -euo pipefail
DEC=$(realpath "$1"); BS=$(realpath "$2"); REF=$(realpath "$3"); OUT=${4:-$PWD/l4_verify_$(date +%H%M%S)}
IMG=${GC_IMAGE:-clic-gc:cu124}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
docker image inspect "$IMG" >/dev/null 2>&1 || docker build --platform linux/amd64 -f "$REPO/clic/l4/Dockerfile.gc" -t "$IMG" "$REPO/clic/l4"
mkdir -p "$OUT/run"
rm -rf "$OUT/run/"*
(cd "$OUT/run" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('decoder')" "$DEC")
cp "$BS" "$OUT/run/bs.zip"
chmod +x "$OUT/run/decoder/decode"
START=$(date +%s.%N)
docker run --rm --gpus '"device=0"' --cpus 2 --memory 12g --network none \
  -v "$OUT/run:/work" -w /work "$IMG" bash -c "cd /work && ./decoder/decode --report /work/decode_report.json"
END=$(date +%s.%N)
echo "wall (container incl. start): $(python3 -c "print(round($END-$START,2))") s"
python3 - "$OUT/run" "$REF" <<'PY'
import hashlib, json, sys
from pathlib import Path
from PIL import Image
run, ref = Path(sys.argv[1]), Path(sys.argv[2])
outs = {p.stem: p for p in (run / "images").glob("*.png")}
refs = {p.stem: p for p in ref.rglob("*.png")}
exp = json.loads((ref / "expected.json").read_text()) if (ref / "expected.json").exists() else {}
bad = 0
for name, p in sorted(outs.items()):
    o = Image.open(p).convert("RGB")
    if name in refs and Image.open(refs[name]).size != o.size:
        print("SIZE MISMATCH", name); bad += 1
    if name in exp and hashlib.sha256(o.tobytes()).hexdigest() != exp[name]:
        print("PIXEL HASH MISMATCH", name); bad += 1
rep = json.loads((run / "decode_report.json").read_text())
print(f"images {len(outs)}; decode wall inside container {rep['wall']:.2f}s; "
      f"est. for 30 images {rep['wall'] / max(len(outs), 1) * 30:.1f}s; problems {bad}")
sys.exit(1 if bad else 0)
PY
