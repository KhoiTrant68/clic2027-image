#!/usr/bin/env bash
# Run one perceptual codec over qualities: encode, fresh-process decode, parity check, then metrics.
# Usage: run.sh <oscar|osdiff> <out_dir> <limit|all> <q> [q ...]
# Expects lab09 layout under /mnt/data/khoitm8/clic2027 (see experiments/perceptual/README.md).
set -euo pipefail
MODEL=$1; OUT=$2; LIMIT=$3; shift 3
B=/mnt/data/khoitm8/clic2027
REPO=$B/clic2027-image
P=$B/perc
DATA=$B/data/valid30
source /mnt/data/khoitm8/env.sh
export CUDA_VISIBLE_DEVICES=6 AEIC_SHARED_SMALL=1 SHARED_CAP_MIB=${SHARED_CAP_MIB:-26624} TORCH_CUDNN_V8_API_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
case $MODEL in
  oscar)  PY=$P/env-oscar/bin/python
          ARGS=(--upstream $B/refs/OSCAR --sd $P/sd21-768 --weights $P/oscar/weights/oscar.pkl) ;;
  osdiff) PY=$P/env-osdiff/bin/python
          ARGS=(--upstream $B/refs/OSDiff --sd $P/sd21-512/v2-1_512-ema-pruned.safetensors
                --openclip $P/openclip-h14/open_clip_model.safetensors --weights-dir $P/osdiff/weights/pretrained_weight) ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
export LD_LIBRARY_PATH="$(dirname $PY)/../lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LIM=(); [ "$LIMIT" != all ] && LIM=(--limit $LIMIT)
mkdir -p $OUT
for q in "$@"; do
  # Resume: a stage whose JSON exists finished; a stage that died leaves no JSON (its partial dir is removed).
  if [ ! -f $OUT/q$q-encode.json ]; then
    rm -rf $OUT/q$q
    $PY $REPO/experiments/perceptual/${MODEL}_run.py --stage encode --quality $q "${ARGS[@]}" --out $OUT --data $DATA "${LIM[@]}" > $OUT/q$q-encode.log 2>&1
  fi
  if [ ! -f $OUT/q$q-decode.json ]; then
    rm -rf $OUT/q$q/recon
    $PY $REPO/experiments/perceptual/${MODEL}_run.py --stage decode --quality $q "${ARGS[@]}" --out $OUT > $OUT/q$q-decode.log 2>&1
  fi
  $B/envs/elic/bin/python - $OUT $q <<'EOF'
import json, sys
import numpy as np
from PIL import Image
out, q = sys.argv[1], sys.argv[2]
e = {r['name']: r for r in json.load(open(f'{out}/q{q}-encode.json'))['images']}
d = {r['name']: r for r in json.load(open(f'{out}/q{q}-decode.json'))['images']}
if set(e) != set(d) or any(e[n]['stream_sha256'] != d[n]['stream_sha256'] for n in e):
    raise SystemExit(f'q{q}: stream inventory/hash mismatch')
# Streams must match exactly; decoder float nondeterminism is recorded, not fatal.
diff = {}
for n in e:
    if e[n]['pixel_sha256'] != d[n]['pixel_sha256']:
        x = np.asarray(Image.open(f'{out}/q{q}/direct/{n}.png')).astype(int)
        y = np.asarray(Image.open(f'{out}/q{q}/recon/{n}.png')).astype(int)
        diff[n] = dict(max_abs=int(np.abs(x - y).max()), frac_changed=float((x != y).mean()))
json.dump(dict(quality=q, images=len(e), stream_parity=True, pixel_mismatch=diff), open(f'{out}/q{q}-parity.json', 'w'), indent=2)
print(f'q{q}: {len(e)} images, streams identical, pixel mismatches {len(diff)}, max abs {max([v["max_abs"] for v in diff.values()] or [0])}')
EOF
done
if [ "$LIMIT" = all ]; then
  for i in 1 2 3 4 5; do
    $B/envs/elic/bin/python $REPO/experiments/perceptual/score.py --run $OUT --data $DATA --qualities "$@" >> $OUT/metrics.log 2>&1 && break
    echo "score retry $i"
  done
fi
echo RUN_DONE
