#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_DIR=/home/jovyan/clic2027-envs/mlicpp
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TORCH_CUDNN_V8_API_DISABLED=1
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ENV_DIR/bin/python" -u "$ROOT/experiments/mlicpp/critic_train.py" "$@"
