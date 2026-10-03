#!/usr/bin/env bash
# Build the exact CLIC 2027 GPU environment on a rented L4 machine.
#
# Case 1 — full VM with Docker (GCP g2-standard-*, Lambda, AWS g6.*): builds the devkit's
#          docker/Dockerfile.gpu as image "clic-gpu". This is what the server runs.
# Case 2 — container host without Docker (RunPod, Vast): run with --venv; creates a venv with
#          the devkit's pinned requirements.txt (torch 2.6.0 etc.). Close, but not identical.
#
# Usage:  bash setup_l4.sh            (case 1)
#         bash setup_l4.sh --venv     (case 2)
set -euo pipefail

WORK=${WORK:-$HOME/clic}
mkdir -p "$WORK" && cd "$WORK"
[ -d devkit ] || git clone --depth 1 https://github.com/clic-challenge/devkit

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ "${1:-}" = "--venv" ]; then
    python3 -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install -q --upgrade pip
    "$WORK/venv/bin/pip" install -q -r devkit/docker/requirements.txt
    "$WORK/venv/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    echo "venv ready: source $WORK/venv/bin/activate"
    exit 0
fi

if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
    # NVIDIA container toolkit (official apt repo)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi

sudo docker build -t clic-gpu -f devkit/docker/Dockerfile.gpu devkit/docker/
sudo docker run --rm --gpus all clic-gpu python3 -c \
    "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
echo "image clic-gpu ready"
