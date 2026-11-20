#!/usr/bin/env bash
# Phase 0 on one GPU: MLIC++ latents for q1 and q2, then every entropy variant, two queues in parallel (one per q).
#   GC_ROOT=... nohup bash experiments/gc_entropy/run.sh [steps] > logs/gce.log 2>&1 &
# Resumable: finished extracts (pool.pt, clic30.pt) and variants (result.json) are skipped, trainers resume last.pt.
set -euo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only
STEPS=${1:-50000}; BATCH=${BATCH:-32}
ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=$ROOT/gce
VARIANTS=${VARIANTS:-"V0 V0f V1 V2 V3 V3s"}
QS=${QS:-"1 2"}
export VRAM_GUARD_MAX_FAILURES=${VRAM_GUARD_MAX_FAILURES:-30} VRAM_GUARD_TIMEOUT=${VRAM_GUARD_TIMEOUT:-10}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 AEIC_SHARED_SMALL=1
ME=$ROOT/envs/mlicpp
retry() {  # VRAM guard: exit 75 (memory threshold) / 76 (monitor failed) -> wait and resume
  local n rc
  for n in 1 2 3 4 5; do
    set +e; "$@"; rc=$?; set -e
    [[ $rc == 0 ]] && return 0
    [[ $rc == 75 || $rc == 76 ]] || return $rc
    echo "[gce] guard stop (exit $rc), attempt $n; retrying in 60 s" >&2
    sleep 60
  done
  return $rc
}
for q in $QS; do
  [[ -f $WORK/q$q/pool.pt && -f $WORK/q$q/clic30.pt ]] && continue
  SHARED_CAP_MIB=20480 TORCH_CUDNN_V8_API_DISABLED=1 \
  LD_LIBRARY_PATH=$ME/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    retry "$ME/bin/python" "$REPO/experiments/gc_entropy/extract.py" --quality "$q" --upstream "$ROOT/refs/MLIC" \
      --weights "$ROOT/mlicpp/weights" --data "$ROOT/data/div2k_train" --valid "$ROOT/data/valid30" --out "$WORK/q$q"
done
echo "extract done $(date)"
queue() {
  local q=$1 v
  for v in $VARIANTS; do
    SHARED_CAP_MIB=${SHARED_CAP_MIB:-10240} retry "$ROOT/envs/gc/bin/python" "$REPO/experiments/gc_entropy/rate_only.py" \
      --variant "$v" --data "$WORK/q$q" --out "$WORK/q$q/$v" --steps "$STEPS" --batch "$BATCH" --eval-every 10000
  done
}
pids=()
for q in $QS; do
  queue "$q" > "$ROOT/logs/gce_q$q.log" 2>&1 &
  pids+=($!)
  sleep 30
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "PHASE0_DONE rc=$rc $(date)"
