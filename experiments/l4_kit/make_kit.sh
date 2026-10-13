#!/usr/bin/env bash
# Pack the kit for a rented L4 (run on our server): these scripts, ditic_distill/common.py, the distilled q3 student
# and bitstreams encoded on the L40 (l4_check.py encode). ~28 MB.
#
#     bash make_kit.sh EXPERIMENTS_DIR STUDENTS_DIR STREAMS_PT OUT.tgz
#     bash make_kit.sh ~/clic/l4kit ~/clic/distill/students ~/clic/clic27_l4/out_server-01/streams_server-01.pt \
#         ~/clic/clic27_l4_kit.tgz
set -euo pipefail
EXP=$1 STUDENTS=$2 STREAMS=$3 OUT=$4
T=$(mktemp -d) && K=$T/clic27_l4_kit
mkdir -p "$K/l4_kit" "$K/ditic_distill" "$K/students"
cp "$EXP"/l4_kit/{l4_check.py,setup_env.sh,run_all.sh,README.md} "$K/l4_kit/"
cp "$EXP/ditic_distill/common.py" "$K/ditic_distill/"
cp "$STUDENTS/q3.pt" "$K/students/"
cp "$STREAMS" "$K/streams_l40.pt"
tar czf "$OUT" -C "$T" clic27_l4_kit && rm -rf "$T"
ls -la "$OUT"
