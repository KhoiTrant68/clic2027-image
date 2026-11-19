#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_DIR=/home/jovyan/clic2027-envs/mlicpp
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"
export TORCH_CUDNN_V8_API_DISABLED=1
export LD_LIBRARY_PATH="$ENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ENV_DIR/bin/python" "$ROOT/experiments/mlicpp/operational_quantizer.py" "$@"
