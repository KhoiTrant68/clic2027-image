#!/usr/bin/env bash
# Step D of plans/GC_v2_architecture_2026-11-23.md section 5 on a machine other than lab09.
#   ./run_stepd.sh DATA_ROOT GPU_ID PYTHON [PARALLEL=1]
# DATA_ROOT holds patches512_div2k/, patches512_clic2020/, valid30/. Runs D0..D5 (rd 100k each; D0/D1 resume from
# resume/ if present), then for each: real streams at 8 levels on the 30 images -> Qhat v0 + LPIPS -> BD-rate.
# Resumable: rerun the same command after any stop; finished runs / evals are skipped.
set -uo pipefail
DATA=$(realpath "$1"); GPU=$2; PY=${3:-python}; PAR=${4:-1}
H=$(cd "$(dirname "$0")" && pwd)
cd "$H"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$H/repo/src TORCH_HOME=$H/weights/torch_home CUBLAS_WORKSPACE_CONFIG=:4096:8
export STEPD_VALID=$DATA/valid30 VRAM_GUARD=off      # dedicated GPU: no lab09 VRAM guard
mkdir -p ~/.cache/clip && cp -n weights/clip/* ~/.cache/clip/   # CLIP-IQA weights where piq looks for them
$PY make_cfgs.py "$DATA" || exit 1
mkdir -p work logs eval
for n in D0 D1; do   # carry over the lab09 checkpoints (D0 21.5k, D1 20.7k steps) once
  [ -d resume/$n/rd ] && [ ! -e work/$n/rd ] && mkdir -p work/$n && cp -r resume/$n/rd work/$n/rd && echo "[stepd] $n resumes from lab09 checkpoint"
done
EV=$H/repo/experiments/gc_v2/stepd_eval.py
one() {
  n=$1
  if [ ! -f work/$n/rd/model.pt ]; then
    echo "[stepd] $n train $(date -u)"
    $PY -m clic27.gc.train --config cfg/$n.json --work work/$n --stage rd >> logs/$n.train.log 2>&1 \
      || { echo "[stepd] $n train stopped rc=$? (rerun to resume)"; return 1; }
  fi
  if [ ! -f eval/$n/metrics.meta.json ]; then
    echo "[stepd] $n eval $(date -u)"
    $PY $EV encode --ckpt work/$n/rd/model.pt --out eval/$n >> logs/$n.encode.log 2>&1 || { echo "[stepd] $n encode failed"; return 1; }
    $PY $H/repo/experiments/perceptual/score.py --run eval/$n --data "$DATA/valid30" --qualities 0 1 2 3 4 5 6 7 \
      >> logs/$n.score.log 2>&1 || { echo "[stepd] $n score failed"; return 1; }
  fi
  echo "[stepd] $n done $(date -u)"
}
bd() {
  tests=""; for m in D1 D2 D3 D4 D5; do [ -f eval/$m/metrics.meta.json ] && tests="$tests eval/$m"; done
  [ -f eval/D0/metrics.meta.json ] && [ -n "$tests" ] && $PY $EV bd --ref eval/D0 --test $tests --json bd_vs_D0.json > bd_vs_D0.md
  : > bd_step.md; prev=D0
  for m in D1 D2 D3 D4 D5; do
    [ -f eval/$prev/metrics.meta.json ] && [ -f eval/$m/metrics.meta.json ] && { $PY $EV bd --ref eval/$prev --test eval/$m >> bd_step.md; echo >> bd_step.md; }
    prev=$m
  done
}
nvidia-smi --query-gpu=index,name,memory.total --format=csv -i "$GPU"
$PY -c "import torch, torchvision, lpips, piq; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
RUNS=(D0 D1 D2 D3 D4 D5); fail=0
for ((i = 0; i < 6; i += PAR)); do
  pids=()
  for ((j = i; j < i + PAR && j < 6; j++)); do one ${RUNS[$j]} & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || fail=1; done
  bd
done
[ $fail -eq 0 ] && echo STEPD_DONE || echo "STEPD_INCOMPLETE: rerun ./run_stepd.sh to resume"
