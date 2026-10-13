#!/usr/bin/env bash
# Everything to run on one rented L4, in one go (setup ~15-20 min, checks ~15 min):
#   1. encode the 30 validation images (DiT-IC q3) here and decode them here (self check);
#   2. decode the bitstreams that came with the kit (streams_*.pt: encoded on our L40) and any passed as arguments
#      (e.g. the other L4's): index / symbol mismatches = is entropy decoding bit-exact across machines;
#   3. CLIC-style decode timing, teacher and distilled student, twice each.
# Results: $WORK/l4_results_<host>.tgz (send it back). On a second L4, pass the first one's streams file:
#     bash run_all.sh /path/to/streams_<first host>.pt
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd)  # kit root: l4_kit/, ditic_distill/, students/, streams_*.pt
WORK=${WORK:-$HOME/clic27_l4}
WORK=$WORK bash "$KIT/l4_kit/setup_env.sh"
cd "$WORK" && . venv/bin/activate
export DITIC_DIR=$WORK/DiT-IC HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
H=$(hostname)
OUT=$WORK/out_$H && mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt" && : > "$OUT/speed.log"
C="python $KIT/l4_kit/l4_check.py"
$C encode --q q3 --valid valid --out "$OUT/streams_$H.pt" 2>&1 | tee "$OUT/encode.log"
$C decode --streams "$OUT/streams_$H.pt" --valid valid --json "$OUT/decode_self.json" 2>&1 | tee "$OUT/decode_self.log"
for s in "$KIT"/streams_*.pt "$@"; do
    [ -f "$s" ] || continue
    n=$(basename "$s" .pt)
    $C decode --streams "$s" --valid valid --json "$OUT/decode_$n.json" 2>&1 | tee "$OUT/decode_$n.log"
done
for d in teacher student teacher student; do
    $C speed --decoder $d --students "$KIT/students" --streams "$OUT/streams_$H.pt" --valid valid \
        --json "$OUT/speed_${d}_$(date +%s).json" 2>&1 | tee -a "$OUT/speed.log"
done
grep -h "^== " "$OUT"/*.log  # DiT-IC itself prints "========mode is ..."
tar czf "$WORK/l4_results_$H.tgz" -C "$WORK" "out_$H"
echo "results: $WORK/l4_results_$H.tgz ($((SECONDS / 60)) min in all)"
