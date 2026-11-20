#!/usr/bin/env bash
# One-time setup for the GC backbone ablation and the entropy phase 0 (MLIC++ latents), on any Linux + NVIDIA box.
#   GC_ROOT=/big/disk/clic27 bash experiments/gc_backbone/setup.sh            # MLIC++ env, repo, weights, DIV2K
#   GC_ROOT=/big/disk/clic27 WITH_ELIC=1 bash experiments/gc_backbone/setup.sh  # + ELIC env, repo, q1-q4 weights
# Layout created under $GC_ROOT: envs/{mlicpp,elic}, refs/{MLIC,ELiC-ReImplemetation}, mlicpp/weights,
# elic/weights, data/div2k_train. CLIC30 (data/valid30) is NOT downloadable: copy it (see BUNDLE_README.md).
# Needs conda (or set CONDA=micromamba), git, curl, unzip; torch wheels come from download.pytorch.org (cu118).
set -euxo pipefail
ENV_SH=${GC_ENV_SH:-/mnt/data/khoitm8/env.sh}; [[ -f $ENV_SH ]] && source "$ENV_SH"  # lab09 only
ROOT=${GC_ROOT:?set GC_ROOT to the work directory}
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
CONDA=${CONDA:-conda}
MLIC_SHA=4aa9d2a512eb9db382172058e6c278fb0d376e17
ELIC_SHA=92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d
mkdir -p "$ROOT"/{envs,refs,data,logs}

make_env() {  # make_env <name> <compressai version>: torch 2.0.1 cu118 stack used for every result in the reports
  local E=$ROOT/envs/$1 P
  [[ -x $E/bin/python ]] && return
  "$CONDA" create -y -p "$E" python=3.10 pip
  P=$E/bin/python
  $P -m pip install "torch==2.0.1+cu118" "torchvision==0.15.2+cu118" --index-url https://download.pytorch.org/whl/cu118
  $P -m pip install "numpy==1.26.4" "setuptools<81" wheel pybind11
  $P -m pip install --no-build-isolation "compressai==$2"
  $P -m pip install "timm==0.6.13" thop ptflops "einops==0.8.2" gdown "lpips==0.1.4" "piq==0.8.0" pytorch-msssim \
    "nvidia-cuda-nvrtc-cu11==11.8.89" scipy pyyaml pillow
  local NV=$E/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
  [[ -e $NV/libnvrtc.so ]] || ln -s libnvrtc.so.11.2 "$NV/libnvrtc.so"
  $P -m pip freeze > "$E/requirements.freeze.txt"
  $P -c "import torch,compressai; print(torch.__version__, compressai.__version__, torch.cuda.is_available())"
}

clone() {  # clone <url> <dir> <sha>
  [[ -d $2 ]] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
  test "$(git -C "$2" rev-parse HEAD)" = "$3"
}

make_env mlicpp 1.2.0b3
clone https://github.com/JiangWeibeta/MLIC.git "$ROOT/refs/MLIC" $MLIC_SHA
bash "$REPO/experiments/mlicpp/download_weights.sh" "$ROOT/mlicpp/weights"
(cd "$ROOT/mlicpp/weights" && sha256sum -c "$REPO/experiments/gc_backbone/weights.sha256" --ignore-missing)

if [[ ${WITH_ELIC:-0} == 1 ]]; then
  make_env elic 1.1.5
  clone https://github.com/VincentChandelier/ELiC-ReImplemetation.git "$ROOT/refs/ELiC-ReImplemetation" $ELIC_SHA
  mkdir -p "$ROOT/elic/weights"
  # q1-q4 = lambda 0.004 / 0.008 / 0.016 / 0.032 (Google Drive ids from the upstream README). The ablation also used
  # q5 = lambda 0.15: take its link from the same README or copy elic-mse-0.15.pth.tar from lab09.
  for pair in 0.004:1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q 0.008:1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL \
              0.016:1MWlYAmpHbWlGtG7MBBTPEew800grY5yC 0.032:1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc; do
    lam=${pair%%:*}; id=${pair##*:}
    [[ -s $ROOT/elic/weights/elic-mse-$lam.pth.tar ]] || "$ROOT/envs/elic/bin/gdown" -q "$id" -O "$ROOT/elic/weights/elic-mse-$lam.pth.tar"
  done
  (cd "$ROOT/elic/weights" && sha256sum -c "$REPO/experiments/gc_backbone/weights.sha256" --ignore-missing)
fi

D=$ROOT/data/div2k_train
if [[ ! -f $D/.done ]]; then
  mkdir -p "$D"
  curl --fail --location --retry 3 -o "$ROOT/data/DIV2K_train_HR.zip" https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  (cd "$D" && unzip -q -j "$ROOT/data/DIV2K_train_HR.zip" '*.png')
  test "$(ls "$D"/*.png | wc -l)" = 800
  rm "$ROOT/data/DIV2K_train_HR.zip"
  touch "$D/.done"
fi
if [[ -d $ROOT/data/valid30 ]]; then
  (cd "$ROOT/data/valid30" && sha256sum -c "$REPO/experiments/gc_backbone/valid30.sha256")
else
  echo "WARNING: $ROOT/data/valid30 missing: copy the 30 CLIC images (BUNDLE_README.md)" >&2
fi
echo SETUP_DONE
