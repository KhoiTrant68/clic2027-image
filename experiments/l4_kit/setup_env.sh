#!/usr/bin/env bash
# Environment for the DiT-IC checks (l4_check.py) on a rented L4 or on our server: a Python 3.12 venv with the CLIC
# devkit's pinned requirements (torch 2.6.0, cuDNN 9.1.0.70: what decides GPU kernel results) plus the few packages
# DiT-IC needs, the DiT-IC checkout, its checkpoints and the 30 validation images. Idempotent.
#
#     WORK=~/clic27_l4 bash setup_env.sh      # QS="q2 q3 q4" for more than the q3 checkpoint (4.3 GB each)
#
# Needs only an NVIDIA driver, curl and git; uv brings its own Python.
set -euo pipefail
WORK=${WORK:-$HOME/clic27_l4}
mkdir -p "$WORK" && cd "$WORK"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d devkit ] || git clone -q --depth 1 https://github.com/clic-challenge/devkit
[ -d venv ] || uv venv -q --python 3.12 venv
. venv/bin/activate
uv pip install -q -r devkit/docker/requirements.txt
printf 'torch==2.6.0\ntorchvision==0.21.0\nnumpy==1.26.4\n' > constraints.txt  # never move what the devkit pins
uv pip install -q -c constraints.txt diffusers==0.41.0 compressai==1.2.8 timm==1.0.30 pyyaml==6.0.3 \
    huggingface_hub==1.33.0 einops==0.8.2
if [ ! -d DiT-IC ]; then
    git clone -q https://github.com/Eric-qi/DiT-IC.git
    git -C DiT-IC checkout -q cd43f5d9761fb34f5224622145629d3ff2b89ca1
fi
HF_HUB_DISABLE_PROGRESS_BARS=1 python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
for q in os.environ.get("QS", "q3").split():
    hf_hub_download("JunqiShi/DiT-IC", f"{q}_merge_ema.pt")  # 4.3 GB each
for f in ("vae/config.json", "transformer/config.json", "scheduler/scheduler_config.json"):
    hf_hub_download("Efficient-Large-Model/Sana_600M_1024px_diffusers", f)  # configs only
EOF
if [ ! -d valid ]; then
    mkdir -p valid_zip && curl -Ls -o valid_zip/v.zip https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip
    python -m zipfile -e valid_zip/v.zip valid_zip && mkdir valid && find valid_zip -name "*.png" -exec mv {} valid/ \; && rm -rf valid_zip
fi
echo "validation images: $(ls valid/*.png | wc -l), devkit $(git -C devkit rev-parse --short HEAD)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), torch.cuda.get_device_name(0))"
