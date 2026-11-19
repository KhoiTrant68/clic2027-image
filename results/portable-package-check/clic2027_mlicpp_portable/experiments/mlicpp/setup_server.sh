#!/usr/bin/env bash
# Isolated inference environment; never install into the shared system Python.
set -euo pipefail
ENV_DIR=${1:?usage: setup_server.sh /absolute/venv/path}
UV=${UV:-/home/jovyan/.local/bin/uv}
BUILD_PREFIX=${BUILD_PREFIX:-/home/jovyan/clic2027-envs/mlicpp-build}
"$UV" pip install --python "$ENV_DIR/bin/python" \
  'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' \
  --index-url https://download.pytorch.org/whl/cu118
"$UV" pip install --python "$ENV_DIR/bin/python" \
  'numpy==1.26.4' 'setuptools<81' wheel
if [[ -x "$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]]; then
  export PATH="$BUILD_PREFIX/bin:$PATH"
  export CC="$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
"${CXX:-c++}" --version
printf 'int main() { return 0; }\n' | "${CXX:-c++}" -x c++ -std=c++11 - -o "$ENV_DIR/compiler-smoke"
"$ENV_DIR/compiler-smoke"
"$UV" pip install --python "$ENV_DIR/bin/python" \
  'compressai==1.2.0b3' 'timm==0.6.13' einops gdown \
  'lpips==0.1.4' 'piq==0.8.0' pytorch-msssim 'nvidia-cuda-nvrtc-cu11==11.8.89'
NVRTC_DIR="$ENV_DIR/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
if [[ ! -e "$NVRTC_DIR/libnvrtc.so" ]]; then
  ln -s libnvrtc.so.11.2 "$NVRTC_DIR/libnvrtc.so"
fi
"$UV" pip freeze --python "$ENV_DIR/bin/python" > "$ENV_DIR/requirements.freeze.txt"
"$ENV_DIR/bin/python" -c 'import torch,compressai; print(torch.__version__, compressai.__version__, torch.cuda.is_available())'
