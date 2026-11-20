#!/usr/bin/env bash
# Create the Python environment for GC (train + encode + decode) under $GC_ROOT.
#   GC_ROOT=/path/with/space bash scripts/gc/setup_env.sh
# Versions follow the CLIC devkit image (torch 2.6.0 / CUDA 12.4, numpy 1.26.4) so the decoder runs the same code
# path locally and on the server.
set -euo pipefail
GC_ROOT=${GC_ROOT:?set GC_ROOT to a writable folder}
ENV_DIR=${GC_ENV:-$GC_ROOT/envs/gc}
export PIP_CACHE_DIR=$GC_ROOT/cache/pip TMPDIR=$GC_ROOT/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  if command -v conda >/dev/null; then
    conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.12
  else
    python3 -m venv "$ENV_DIR"
  fi
fi
PY=$ENV_DIR/bin/python
# torch 2.6 wheels need Python 3.9-3.13; the tested env is 3.12.
$PY -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if (3, 10) <= v <= (3, 13) else f"Python {v} unsupported: need 3.10-3.13 (tested 3.12)")'
command -v "${CC:-gcc}" >/dev/null || command -v cc >/dev/null || { echo "need a C compiler (gcc) for the rANS library"; exit 1; }
$PY -m pip install --upgrade pip
# Exact versions from the tested env; torch/torchvision resolve from the cu124 index.
$PY -m pip install -r "$(dirname -- "$0")/requirements-gc.lock" --extra-index-url https://download.pytorch.org/whl/cu124
$PY - <<'PYCHECK'
import torch, lpips, piq  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no CUDA here (fine on a login node used only for prefetch); the cu124 wheels need driver >= 525")
elif torch.cuda.get_device_capability() >= (10, 0):
    raise SystemExit(f"GPU {torch.cuda.get_device_name(0)} (sm_{''.join(map(str, torch.cuda.get_device_capability()))}) "
                     "is not supported by torch 2.6 cu124")
PYCHECK
echo "env ready: $ENV_DIR"
