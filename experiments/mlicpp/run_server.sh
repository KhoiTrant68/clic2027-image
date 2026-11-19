#!/usr/bin/env bash
# Freeze the CUDA library/backend choices for both encoder and fresh decoder.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_DIR=${MLICPP_ENV:-/home/jovyan/clic2027-envs/mlicpp}
NVRTC_DIR="$ENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU index}"
export TORCH_CUDNN_V8_API_DISABLED=1
export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ENV_DIR/bin/python" -c 'import ctypes; ctypes.CDLL("libnvrtc.so")'
exec "$ENV_DIR/bin/python" "$REPO_ROOT/experiments/mlicpp/probe.py" "$@"
