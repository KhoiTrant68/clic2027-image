#!/usr/bin/env bash
# lab09 wrapper (kept for the commands in the reports): same as setup.sh with the lab09 work directory.
GC_ROOT=${GC_ROOT:-/mnt/data/khoitm8/clic2027} exec bash "$(dirname -- "$0")/setup.sh" "$@"
