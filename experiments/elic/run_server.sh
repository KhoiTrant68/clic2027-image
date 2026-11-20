#!/usr/bin/env bash
# Freeze CUDA library/backend choices for encoder and fresh decoder (same as experiments/mlicpp).
# Shared GPU: AEIC_SHARED_SMALL=1 caps the allocator at 8 GiB and stops the process at 10 GiB.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ENV_DIR=${ELIC_ENV:?Set ELIC_ENV to the venv/conda prefix}
NVRTC_DIR="$ENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU index}"
export TORCH_CUDNN_V8_API_DISABLED=1
export AEIC_SHARED_SMALL=${AEIC_SHARED_SMALL:-1}
export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ENV_DIR/bin/python" -c 'import ctypes; ctypes.CDLL("libnvrtc.so")'
exec "$ENV_DIR/bin/python" "$REPO_ROOT/experiments/elic/probe.py" "$@"
