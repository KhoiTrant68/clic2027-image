#!/usr/bin/env bash
# Gather everything to send back for step D into ~/stepd/send/<UTC time>/ (+ a .tar.gz of it).
# Safe to run any time: copies only what exists so far; never touches the run itself.
H=~/stepd/stepd_a30
T=$(date -u +%Y%m%d_%H%M)
O=~/stepd/send/stepd_$T
mkdir -p "$O"
cd "$H" || exit 1
./status.sh > "$O/status.txt" 2>&1
for f in bd_vs_D0.md bd_step.md bd_vs_D0.json; do [ -s "$f" ] && cp "$f" "$O/"; done
for n in D0 D1 D2 D3 D4 D5; do
  [ -f eval/$n/metrics.csv ] && { mkdir -p "$O/eval/$n"; cp eval/$n/metrics.csv eval/$n/metrics.meta.json "$O/eval/$n/" 2>/dev/null; }
  [ -f work/$n/rd/eval.jsonl ] && { mkdir -p "$O/work/$n/rd"; cp work/$n/rd/eval.jsonl "$O/work/$n/rd/"; }
  for k in train encode score; do   # last 50 lines of each log, in case something failed
    [ -f logs/$n.$k.log ] && { mkdir -p "$O/logs_tail"; tail -50 logs/$n.$k.log > "$O/logs_tail/$n.$k.log"; }
  done
done
tail -50 ~/stepd/run_stepd.log > "$O/run_stepd.tail.log" 2>/dev/null
grep -q STEPD_DONE ~/stepd/run_stepd.log 2>/dev/null && echo "STEPD_DONE" > "$O/DONE"
tar czf "$O.tar.gz" -C ~/stepd/send "stepd_$T"
echo "$O"; echo "$O.tar.gz"; (cd "$O" && find . -type f | sort)
