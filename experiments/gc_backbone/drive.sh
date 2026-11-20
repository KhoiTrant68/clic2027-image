#!/usr/bin/env bash
# Whole ablation on lab09 GPU 6: wait for data + MLIC++ baseline streams, smoke both arms, then both arms in parallel.
#   nohup bash experiments/gc_backbone/drive.sh 20000 > logs/gcb_drive.log 2>&1 &
set -euo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only; elsewhere set GC_ROOT
STEPS=${1:-20000}
ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
until [[ -f data/div2k_train/.done ]]; do sleep 60; done
until grep -q '"complete"' mlicpp/full30/status.json 2>/dev/null; do
  grep -q '"failed"' mlicpp/full30/status.json 2>/dev/null && { echo "MLIC++ baseline failed"; exit 1; }
  sleep 60
done
echo "inputs ready $(date)"
for spec in "elic 3 refs/ELiC-ReImplemetation elic/weights" "mlicpp 2 refs/MLIC mlicpp/weights"; do
  set -- $spec
  E=$ROOT/envs/$1
  if [[ ! -f gcb/smoke/$1/done.json ]]; then
    rm -rf "gcb/smoke/$1"
    TORCH_CUDNN_V8_API_DISABLED=1 AEIC_SHARED_SMALL=1 SHARED_CAP_MIB=20480 \
    LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
      "$E/bin/python" "$REPO/experiments/gc_backbone/train_gs.py" --arm "$1" --quality "$2" --upstream "$3" \
      --weights "$4" --data data/div2k_train --out "gcb/smoke/$1" --steps 200
  fi
done
echo "smoke ok $(date)"
bash "$REPO/experiments/gc_backbone/run_arm.sh" elic "2 3 4 5" "$STEPS" > logs/gcb_elic.log 2>&1 &
P1=$!
sleep 120
bash "$REPO/experiments/gc_backbone/run_arm.sh" mlicpp "1 2 3 4" "$STEPS" > logs/gcb_mlicpp.log 2>&1 &
P2=$!
S1=0; S2=0
wait $P1 || S1=$?
wait $P2 || S2=$?
echo "arms finished elic=$S1 mlicpp=$S2 $(date)"
