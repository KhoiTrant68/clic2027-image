#!/usr/bin/env bash
# Train the GC-recipe decoder for each quality of one backbone, then decode CLIC30 from the ORIGINAL streams with it
# and score. Resumable: finished trainings (gs.pt), decodes (qN-decode.json) and metric rows are skipped.
#   bash experiments/gc_backbone/run_arm.sh elic "2 3 4 5" 20000
#   bash experiments/gc_backbone/run_arm.sh mlicpp "1 2 3 4" 20000
set -euo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only; elsewhere set GC_ROOT
ARM=$1; QS=$2; STEPS=${3:-20000}
ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=$ROOT/gcb
# nvidia-smi on lab09 can stall for minutes; tolerate ~5 min of missed readings before the guard fails closed.
export VRAM_GUARD_MAX_FAILURES=${VRAM_GUARD_MAX_FAILURES:-30} VRAM_GUARD_TIMEOUT=${VRAM_GUARD_TIMEOUT:-10}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_CUDNN_V8_API_DISABLED=1 AEIC_SHARED_SMALL=1 SHARED_CAP_MIB=${SHARED_CAP_MIB:-20480}
if [[ $ARM == elic ]]; then
  ENV=$ROOT/envs/elic; UP=$ROOT/refs/ELiC-ReImplemetation; W=$ROOT/elic/weights; BASE=$ROOT/elic/full30; EXTRA=()
else
  ENV=$ROOT/envs/mlicpp; UP=$ROOT/refs/MLIC; W=$ROOT/mlicpp/weights; BASE=$ROOT/mlicpp/full30; EXTRA=(--legacy)
fi
export LD_LIBRARY_PATH=$ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=$ENV/bin/python
retry() {  # the VRAM guard exits 75 (memory threshold) or 76 (monitor failed): wait, then resume (last.pt)
  local n rc
  for n in 1 2 3 4 5; do
    set +e; "$@"; rc=$?; set -e
    [[ $rc == 0 ]] && return 0
    [[ $rc == 75 || $rc == 76 ]] || return $rc
    echo "[run_arm] guard stop (exit $rc), attempt $n; retrying in 60 s" >&2
    sleep 60
  done
  return $rc
}
decode() {  # decode CLIC30 from the ORIGINAL streams with the trained g_s: rate is unchanged by construction
  local q=$1 T=$2
  rm -rf "$EVAL/q$q"
  mkdir -p "$EVAL/q$q"
  cp -r "$BASE/q$q/streams" "$EVAL/q$q/streams"
  cp "$BASE/q$q-encode.json" "$EVAL/"
  $PY "$REPO/experiments/$ARM/probe.py" --stage decode --quality "$q" --upstream "$UP" --weights "$W" \
    --out "$EVAL" --synthesis "$T/gs.pt" "${EXTRA[@]}"
}
EVAL=$WORK/eval/$ARM
mkdir -p "$EVAL"
cp -n "$BASE/manifest.json" "$EVAL/" 2>/dev/null || true
for q in $QS; do
  T=$WORK/train/$ARM-q$q
  if [[ ! -f $T/gs.pt ]]; then
    retry $PY "$REPO/experiments/gc_backbone/train_gs.py" --arm "$ARM" --quality "$q" --upstream "$UP" --weights "$W" \
      --data "$ROOT/data/div2k_train" --out "$T" --steps "$STEPS" --tf32  # both arms; see README (check --tf32)
  fi
  if [[ ! -f $EVAL/q$q-decode.json ]]; then
    retry decode "$q" "$T"
  fi
done
retry $PY "$REPO/experiments/perceptual/score.py" --run "$EVAL" --data "$ROOT/data/valid30" --qualities $QS
echo "ARM_DONE $ARM"
