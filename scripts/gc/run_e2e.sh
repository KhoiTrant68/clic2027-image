#!/usr/bin/env bash
# End-to-end GC run: env check -> data -> reward -> train -> encode/allocate -> package -> verify -> report.
#
#   GC_ROOT=/data/me bash scripts/gc/run_e2e.sh configs/gc_smoke.json runs/smoke            # everything
#   GC_ROOT=/data/me bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full train encode   # some stages
#
# Every cache (pip, torch hub, HF, tmp) goes under $GC_ROOT; stages resume where they stopped.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
CFG=${1:?config json}
WORK=${2:?work dir}
shift 2
GC_ROOT=${GC_ROOT:?set GC_ROOT}
PY=${GC_PYTHON:-$GC_ROOT/envs/gc/bin/python}
export XDG_CACHE_HOME=$GC_ROOT/cache TORCH_HOME=$GC_ROOT/cache/torch HF_HOME=$GC_ROOT/cache/hf \
       PIP_CACHE_DIR=$GC_ROOT/cache/pip TMPDIR=$GC_ROOT/tmp MPLCONFIGDIR=$GC_ROOT/cache/mpl
mkdir -p "$TMPDIR" "$TORCH_HOME"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
cd "$REPO"
case "$WORK" in /*) ;; *) WORK=$GC_ROOT/$WORK ;; esac
mkdir -p "$WORK"
# Run in the background so a SIGTERM / SIGINT / SIGUSR1 sent to this script (preemption, scancel, Ctrl-C) reaches
# the pipeline, which forwards it to training: last.pt is saved and the script exits 75 ("resume me").
"$PY" -m clic27.gc.pipeline --config "$CFG" --work "$WORK" ${@:+--stages "$@"} > >(tee -a "$WORK/run.log") 2>&1 &
PID=$!
trap 'kill -TERM $PID 2>/dev/null' TERM USR1
trap 'kill -INT $PID 2>/dev/null' INT
set +e
while :; do
  wait $PID; rc=$?
  kill -0 $PID 2>/dev/null || break   # wait returns early when a trap fires; keep waiting until the pipeline exits
done
[[ $rc == 75 ]] && echo "[run_e2e] training stopped early and checkpointed; rerun this exact command to resume" >&2
exit $rc
