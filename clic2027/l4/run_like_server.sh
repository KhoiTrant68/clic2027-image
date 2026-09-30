#!/usr/bin/env bash
# Run a submission the way the CLIC server does: decoder zip + bs.zip in one directory,
# ./decode as entrypoint, 2 CPUs, 12 GB RAM, one GPU, no network. Prints wall time.
#
# Usage: bash run_like_server.sh decoder.zip bs.zip [outdir]
set -euo pipefail
DEC=$(realpath "$1"); BS=$(realpath "$2"); OUT=${3:-$(mktemp -d)}
mkdir -p "$OUT" && cd "$OUT"
unzip -oq "$DEC" && cp "$BS" bs.zip && chmod +x decode

start=$(date +%s.%N)
sudo docker run --rm --gpus all --cpus=2 --memory=12g --network none \
    -v "$OUT":/work -w /work clic-gpu ./decode > decode.log 2>&1 || { tail -50 decode.log; exit 1; }
end=$(date +%s.%N)

n=$(ls images/*.png 2>/dev/null | wc -l)
printf "decoded %d images in %.1f s -> %s/images\n" "$n" "$(awk "BEGIN{print $end - $start}")" "$OUT"
