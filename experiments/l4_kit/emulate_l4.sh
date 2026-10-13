#!/usr/bin/env bash
# Rough L4 stand-in on one L40 of our server (2026-10-14; we have no L4). Run after run_all.sh, same $WORK:
#   - MPS gives the decoder SM% of the L40's 142 SMs (an L4 has 58 at <= 2040 MHz vs 2490: ~34% at full clock);
#   - a second MPS client with HOG% of the SMs streams memory, leaving the decoder about an L4's 300 GB/s;
#   - taskset + OMP_NUM_THREADS give the decoder 2 CPUs (CLIC GPU track: 2 CPUs, 12 GB RAM).
# Not captured: the L4's 72 W cap (clocks drop under sustained load; a lower SM% stands in), its 48 MB L2, its CPU.
# The MPS daemon uses a private pipe directory, so nobody else's CUDA process joins it.
#
# NOT a faithful L4 (calibration 14/10, l4_emu.py calib on the idle L40 #1): MPS caps how many SMs a client may use but
# does not keep clients on separate SMs, so the hog also takes the decoder's SMs. Copy GB/s / bf16 GEMM TFLOPS seen:
# SM 100% 655 / 108; SM 34% alone 667 / 60 (the L40's bandwidth stays); SM 34% + hog 50% 272 / 12 (compute collapses).
# Kept for the record; do not use its timings.
#
#     GPU=1 CPUS=10,11 bash emulate_l4.sh calib 34 0 40 60    # what the decoder sees at SM 34%, hog 0 / 40 / 60%
#     GPU=1 CPUS=10,11 bash emulate_l4.sh speed 34 60         # teacher and student decode timing at SM 34%, hog 60%
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HOME/clic27_l4}
GPU=${GPU:-1} CPUS=${CPUS:-0,1}
cd "$WORK" && . venv/bin/activate
export DITIC_DIR=$WORK/DiT-IC HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
export CUDA_MPS_PIPE_DIRECTORY=$WORK/mps/pipe CUDA_MPS_LOG_DIRECTORY=$WORK/mps/log
M=$WORK/mps && mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
unset CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
export CUDA_VISIBLE_DEVICES=0  # clients number devices within the daemon's list
HOG=
hog_on() {
    rm -f "$M/stop" "$M/ready"
    [ "$1" -gt 0 ] || return 0
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1 python "$KIT/l4_kit/l4_emu.py" hog --stop "$M/stop" --ready "$M/ready" &
    HOG=$!
    until [ -f "$M/ready" ]; do sleep 1; done
}
hog_off() { touch "$M/stop"; [ -z "$HOG" ] || wait "$HOG"; HOG=; }
trap 'hog_off; echo quit | nvidia-cuda-mps-control' EXIT
run() { CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1 OMP_NUM_THREADS=2 taskset -c "$CPUS" "${@:2}"; }
mode=$1 sm=$2
shift 2
H=$(hostname)
OUT=$WORK/emu_$H && mkdir -p "$OUT"
case $mode in
calib)
    for h in "$@"; do
        hog_on "$h"
        run "$sm" python "$KIT/l4_kit/l4_emu.py" calib --tag "sm$sm hog$h" | tee -a "$OUT/calib.log"
        hog_off
    done;;
speed)
    h=$1
    hog_on "$h"
    nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu --format=csv -lms 500 \
        > "$OUT/smi_sm${sm}_hog$h.csv" &
    SMI=$!
    for d in teacher student; do
        run "$sm" python "$KIT/l4_kit/l4_check.py" speed --decoder $d --students "$KIT/students" \
            --streams "$WORK/out_$H/streams_$H.pt" --valid valid --tag "(L4 stand-in: SM $sm%, hog $h%)" \
            --json "$OUT/speed_${d}_sm${sm}_hog$h.json" > "$OUT/speed_${d}_sm${sm}_hog$h.log" 2>&1
        grep "^== " "$OUT/speed_${d}_sm${sm}_hog$h.log" | tee -a "$OUT/speed.log"
    done
    kill "$SMI"
    hog_off;;
esac
