#!/usr/bin/env bash
# Finish phase 0 after a queue died (20/11: another user took ~50 GB of GPU 6, so the q2 queue failed to start V2).
# q2 continues now; q1 continues once the original run.sh (still training q1) has exited. Finished variants
# (result.json) are skipped and unfinished ones resume from last.pt. Writes gce/phase0.done, which long.sh waits for.
#   GC_ROOT=... nohup bash experiments/gc_entropy/resume.sh > logs/gce_resume.log 2>&1 &
set -euo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only
ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=$ROOT/gce
STEPS=${STEPS:-50000}; BATCH=${BATCH:-32}
VARIANTS=${VARIANTS:-"V0 V0f V1 V2 V3 V3s"}
export VRAM_GUARD_MAX_FAILURES=${VRAM_GUARD_MAX_FAILURES:-30} VRAM_GUARD_TIMEOUT=${VRAM_GUARD_TIMEOUT:-10}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 AEIC_SHARED_SMALL=1 SHARED_CAP_MIB=${SHARED_CAP_MIB:-5120}
export VRAM_GUARD_WAIT=${VRAM_GUARD_WAIT:-43200}
PY=$ROOT/envs/gc/bin/python
retry() {
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
queue() {
  local q=$1 v
  for v in $VARIANTS; do
    [[ -f $WORK/q$q/$v/result.json ]] && continue
    retry "$PY" "$REPO/experiments/gc_entropy/rate_only.py" --variant "$v" --data "$WORK/q$q" --out "$WORK/q$q/$v" \
      --steps "$STEPS" --batch "$BATCH" --eval-every 10000
  done
}
queue 2 >> "$ROOT/logs/gce_q2.log" 2>&1 &
P2=$!
while pgrep -u "$(id -u)" -f "^bash clic2027-image/experiments/gc_entropy/run.sh" > /dev/null; do sleep 60; done
queue 1 >> "$ROOT/logs/gce_q1.log" 2>&1 &
P1=$!
rc=0
wait $P1 || rc=$?
wait $P2 || rc=$?
if [[ $rc == 0 ]]; then touch "$WORK/phase0.done"; fi
echo "PHASE0_DONE rc=$rc $(date)"
