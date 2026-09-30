"""Compare probe.py outputs across devices. Any idx mismatch = a broken bitstream on the server.

python compare.py out/*.npz
"""
import itertools
import sys

import numpy as np


def main(paths):
    runs = {p: np.load(p) for p in paths}
    for p, r in runs.items():
        print(p, r["meta"])
    keys = sorted({k for r in runs.values() for k in r.files if k.endswith("_idx")})
    print(f"\n{'key':24s} {'pair':40s} {'idx mismatch':>14s} {'max rel dscale':>20s}")
    worst = {}
    for (pa, ra), (pb, rb) in itertools.combinations(runs.items(), 2):
        for k in keys:
            if k not in ra.files or k not in rb.files:
                continue  # mode skipped on one of the devices
            a, b = ra[k], rb[k]
            n_bad = int((a != b).sum())
            sk = k.replace("_idx", "_scale")
            rel = ""
            if sk in ra.files and sk in rb.files:
                sa, sb = ra[sk].astype(np.float64), rb[sk].astype(np.float64)
                rel = f"{np.max(np.abs(sa - sb) / np.abs(sa)):.2e}"
            mode = k.split("_", 1)[1]
            worst[mode] = max(worst.get(mode, 0), n_bad)
            print(f"{k:24s} {pa.split('/')[-1] + ' vs ' + pb.split('/')[-1]:40s} {n_bad:>8d}/{a.size:<7d} {rel:>18s}")
    print("\nVerdict (worst case over all pairs and sizes):")
    for mode, n in worst.items():
        print(f"  {mode:16s} {'OK (bit-exact)' if n == 0 else f'BROKEN: {n} mismatched indices'}")


if __name__ == "__main__":
    main(sys.argv[1:])
