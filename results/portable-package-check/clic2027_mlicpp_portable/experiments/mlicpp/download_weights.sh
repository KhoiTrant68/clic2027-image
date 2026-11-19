#!/usr/bin/env bash
# Public checkpoint URLs linked by the authors' README; no training.
set -euo pipefail
DEST=${1:?usage: download_weights.sh /absolute/weights/path}
mkdir -p "$DEST"
fetch() {
  local name=$1 id=$2
  if [[ -e "$DEST/$name" ]]; then
    echo "Already present: $name"
    return
  fi
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 1800 \
    "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" \
    --output "$DEST/$name.part"
  mv -- "$DEST/$name.part" "$DEST/$name"
  sha256sum "$DEST/$name"
}
fetch mlicpp-mse-q1-legacy.pth.tar 1rRdM5X5ueWvW5-ZN3gwPoXGECvgIcz-2
fetch mlicpp-mse-q2-legacy.pth.tar 1SlIsurg4JocEBAWrnp1jiEe1EnFgw5AP
fetch mlicpp-mse-q3-legacy.pth.tar 1CKpyUXwifJZ9gb0l_Di6R-TMH9f4UmAU
fetch mlicpp-mse-q4-legacy.pth.tar 1Qp4YBea3GkLSGDyYgo7OeaWZa5_rX8kJ
