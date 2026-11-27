#!/usr/bin/env bash
# Rebuild the step D data from public sources, exactly as on lab09, then check it file by file.
#   ./make_data.sh DATA_ROOT PYTHON
# Downloads ~9.5 GB (DIV2K train HR 3.3 GB, CLIC 2020 professional + mobile train 6 GB, CLIC 2025 image test 116 MB),
# extracts, cuts the 512 px patches with the same code and seeds (clic27.gc.data.prep_patches), and compares every
# PNG against data_checksums.sha256 (made on lab09). Needs ~25 GB free; the zips can be deleted afterwards.
set -euo pipefail
D=$(mkdir -p "$1" && cd "$1" && pwd); PY=${2:-python}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/dl"
get() { [ -s "$D/dl/$2" ] || { echo "download $2"; curl -fL --retry 5 -C - -o "$D/dl/$2.part" "$1" && mv "$D/dl/$2.part" "$D/dl/$2"; }; }
get https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip DIV2K_train_HR.zip
get https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip professional_train_2020.zip
get https://data.vision.ee.ethz.ch/cvl/clic/mobile_train_2020.zip mobile_train_2020.zip
get https://dhldkwazkze5h.cloudfront.net/data/clic2025_image_test.zip clic2025_image_test.zip
( cd "$D/dl" && sha256sum -c - <<'SUMS'
54a2a684b5cbd0db91770a8695efb774fed1f612194fee68bceaf6b1c48e5d5f  professional_train_2020.zip
4bf9d890e6785426fb41cb4f43df3d8cd6db25fafcc76fc1eb629141166d6529  mobile_train_2020.zip
c479a92b7579f716826822fe892abbe583cfa7cd7f07730a14227a98c7870e92  clic2025_image_test.zip
SUMS
)
# DIV2K: 800 PNG flattened into div2k_train/ (0001.png ... 0800.png)
if [ ! -f "$D/div2k_train/0800.png" ]; then
  mkdir -p "$D/div2k_train" && unzip -q -j -o "$D/dl/DIV2K_train_HR.zip" '*.png' -d "$D/div2k_train"; fi
# CLIC 2020 train: professional then mobile into clic2020_train/train/ (1633 PNG; __MACOSX junk skipped)
if [ ! -d "$D/clic2020_train/train" ]; then
  mkdir -p "$D/clic2020_train"
  unzip -q -o "$D/dl/professional_train_2020.zip" 'train/*' -d "$D/clic2020_train"
  unzip -q -o "$D/dl/mobile_train_2020.zip" 'train/*' -d "$D/clic2020_train"; fi
# CLIC 2025 image test = the 30 validation images
if [ ! -d "$D/valid30" ]; then
  mkdir -p "$D/valid30" && unzip -q -j -o "$D/dl/clic2025_image_test.zip" '*.png' -d "$D/valid30"; fi
echo "div2k $(ls "$D/div2k_train" | wc -l)/800, clic2020 $(ls "$D/clic2020_train/train" | wc -l)/1633, valid30 $(ls "$D/valid30"/*.png | wc -l)/30"
# Patches: same parameters as lab09 (DIV2K 4 per image, CLIC 2020 8 per image, 512 px, seed = image index)
cd "$D"
PYTHONPATH="$H/repo/src" $PY -c "
from pathlib import Path
from clic27.gc.data import prep_patches
prep_patches([Path('$D/div2k_train')], Path('$D/patches512_div2k'), per_image=4, size=512, workers=8)
prep_patches([Path('$D/clic2020_train/train')], Path('$D/patches512_clic2020'), per_image=8, size=512, workers=8)
"
echo "patches: div2k $(ls patches512_div2k | wc -l)/3200, clic2020 $(ls patches512_clic2020 | wc -l)/12872"
echo "checking every PNG against lab09 checksums..."
if sha256sum -c --quiet "$H/data_checksums.sha256"; then echo "DATA_OK: identical to lab09"
else echo "DATA_DIFFERS: see the mismatches above (usually a different Pillow version re-sampling the patches)"; exit 1; fi
