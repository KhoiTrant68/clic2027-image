#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
python3.10 -m venv .venv
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install 'numpy==1.26.4' 'setuptools<81' wheel
"$PY" -m pip install 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip install 'compressai==1.2.0b3' 'timm==0.6.13' einops gdown 'lpips==0.1.4' 'piq==0.8.0' pytorch-msssim 'nvidia-cuda-nvrtc-cu11==11.8.89' pytest
"$PY" -m pip install -e . --no-deps
NVRTC="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
if [[ ! -e "$NVRTC/libnvrtc.so" ]]; then ln -s libnvrtc.so.11.2 "$NVRTC/libnvrtc.so"; fi
mkdir -p refs/repos
if [[ ! -d refs/repos/MLIC ]]; then git clone https://github.com/JiangWeibeta/MLIC.git refs/repos/MLIC; fi
git -C refs/repos/MLIC checkout --detach 4aa9d2a512eb9db382172058e6c278fb0d376e17
"$PY" -m pip freeze > portable/requirements.installed.txt
"$PY" -c 'import torch,compressai,lpips,piq; print(torch.__version__,compressai.__version__,torch.cuda.is_available())'
