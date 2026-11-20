#!/usr/bin/env bash
# Design 2.1 (G0 amendment): after the 50k-step phase 0, continue V0 and the best 50k variant of each quality
# (lowest CLIC30 bpp among V1 V2 V3 V3s) to 300k steps in total: fresh Adam, lr 3e-4 (x0.1 for the last 10 %).
#   GC_ROOT=... nohup bash experiments/gc_entropy/long.sh > logs/gce_long.log 2>&1 &
# Waits for $GC_ROOT/gce/phase0.done (written by resume.sh). G0 is then applied verbatim to the long results.
set -euo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only
ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=$ROOT/gce
TOTAL=${TOTAL:-300000}; BATCH=${BATCH:-32}; LR=${LR:-3e-4}
QS=${QS:-"1 2"}
export VRAM_GUARD_MAX_FAILURES=${VRAM_GUARD_MAX_FAILURES:-30} VRAM_GUARD_TIMEOUT=${VRAM_GUARD_TIMEOUT:-10}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 AEIC_SHARED_SMALL=1 SHARED_CAP_MIB=${SHARED_CAP_MIB:-5120}
export VRAM_GUARD_WAIT=${VRAM_GUARD_WAIT:-43200}  # GPU 6 is shared: wait for free VRAM instead of failing
PY=$ROOT/envs/gc/bin/python
until [[ -f $WORK/phase0.done ]]; do sleep 60; done
retry() {
  local n rc
  for n in 1 2 3 4 5; do
    set +e; "$@"; rc=$?; set -e
    [[ $rc == 0 ]] && return 0
    [[ $rc == 75 || $rc == 76 ]] || return $rc
    echo "[gce-long] guard stop (exit $rc), attempt $n; retrying in 60 s" >&2
    sleep 60
  done
  return $rc
}
long() {  # long <q> <variant>
  local q=$1 v=$2 init=$WORK/q$1/$2/model.pt done
  done=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'])" "$WORK/q$q/$v/result.json")
  retry "$PY" "$REPO/experiments/gc_entropy/rate_only.py" --variant "$v" --data "$WORK/q$q" --out "$WORK/q$q/long_$v" \
    --init "$init" --steps $((TOTAL - done)) --batch "$BATCH" --lr "$LR" --eval-every 25000
}
pids=()
for q in $QS; do
  best=$("$PY" - "$WORK/q$q" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
r = {v: json.loads((d / v / 'result.json').read_text())['bpp'] for v in ('V1', 'V2', 'V3', 'V3s')
     if (d / v / 'result.json').exists()}
if len(r) != 4:
    raise SystemExit(f'phase 0 incomplete in {d}: {sorted(r)}')
print(min(r, key=r.get))
EOF
)
  echo "q$q: best 50k variant $best $(date)"
  echo "$best" > "$WORK/q$q/long_pick.txt"
  for v in V0 "$best"; do
    long "$q" "$v" > "$ROOT/logs/gce_long_q${q}_$v.log" 2>&1 &
    pids+=($!)
    sleep 20
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "LONG_DONE rc=$rc $(date)"
