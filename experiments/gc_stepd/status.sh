#!/usr/bin/env bash
# Short progress report: last training line of each run, finished evals, BD-rate tables.
cd "$(dirname "$0")"
for n in D0 D1 D2 D3 D4 D5; do
  l=$(grep "^\[rd\] step=" logs/$n.train.log 2>/dev/null | tail -1 | cut -d' ' -f2,4,6)
  e=$([ -f eval/$n/metrics.meta.json ] && echo "eval done" || true)
  echo "$n: ${l:-not started} $e"
done
[ -f bd_vs_D0.md ] && { echo; cat bd_vs_D0.md; }
[ -s bd_step.md ] && { echo; cat bd_step.md; }
