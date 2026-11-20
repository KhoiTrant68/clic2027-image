#!/usr/bin/env bash
# Pack everything needed to train / encode / package GC on another machine into one tar.gz.
#   bash scripts/gc/make_bundle.sh [out.tar.gz]
# On the target machine (details: BUNDLE_README.md at the bundle root):
#   tar xzf clic27_gc_*.tar.gz && cd clic27_gc
#   export GC_ROOT=/big/disk/clic27 && bash scripts/gc/setup_env.sh
#   bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full prefetch # on a node WITH Internet
#   bash scripts/gc/run_e2e.sh configs/gc_smoke.json runs/smoke        # 10-15 min check
#   bash scripts/gc/run_e2e.sh configs/gc_full.json runs/full          # after editing data paths / nproc
set -euo pipefail
REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
OUT=${1:-$REPO/dist/clic27_gc_$(date +%Y-%m-%d).tar.gz}
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
# Besides GC itself: the backbone ablation (20/11), the MLIC++-style entropy design and everything its phase 0 needs
# (MLIC++/ELIC probes, latent pool trainer, Q-hat scoring, budget allocation), plus the per-image CLIC30 results the
# tracker and the comparisons read. Data, weights and envs are fetched on the target (scripts/gc/BUNDLE_README.md).
FILES=(
  src/clic27 pyproject.toml configs/gc_smoke.json configs/gc_full.json scripts/gc clic/l4/Dockerfile.gc
  docs/GC_HUONG_DAN.md plans/CLIC2027_win_plan_2026-11-19.md plans/GC_entropy_MLICpp_design_2026-11-20.md
  reports/*.md  # markdown reports only (the tracker links them); PDFs, figures and slides stay out
  tools/blind_score.py tools/BLIND_SCORING.md tools/tracker
  tests/test_gc.py tests/test_entropy.py tests/test_tracker.py
  experiments/qhat experiments/gc_backbone experiments/gc_entropy experiments/mlicpp experiments/elic experiments/perceptual
  experiments/bakeoff experiments/new_codecs/half_vram.py
  results/experiments.json results/tracker results/2026-10-15_new_codecs/qhat_v0.json
  results/2026-10-15_pulse_integration/integrated_bakeoff results/2026-10-08_bakeoff_merged
  results/2026-10-10_bakeoff_v0 results/2026-10-13_bakeoff_v0 results/2026-10-20_aeic_full30/full30_20261020/scores
  results/2026-11-02_mlicpp/full30 results/2026-11-19_elic results/2026-11-20_perceptual/oscar
  results/2026-11-20_perceptual/osdiff results/2026-11-20_gc_backbone results/2026-11-20_gc_entropy
)
for f in "${FILES[@]}"; do [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }; done
# Shell scripts must keep LF endings (a CRLF run_arm.sh failed with pipefail errors on lab09).
if grep -rlI $'\r' --include='*.sh' "${FILES[@]}" 2>/dev/null; then echo "CRLF in the scripts above" >&2; exit 1; fi
tar --exclude="__pycache__" --exclude="*.so" --exclude="*.pyc" \
  --transform "s,^scripts/gc/BUNDLE_README.md$,BUNDLE_README.md," --transform "s,^,clic27_gc/," \
  -czf "$OUT" "${FILES[@]}"  # scripts/gc/BUNDLE_README.md lands at the bundle root
# Relative file name, so `sha256sum -c` works wherever the two files are copied together.
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" | tee "$(basename "$OUT").sha256")
