# Blind pair set for the rating page: 3 random 512x512 crops per validation image, MS-ILLM vs DiT-IC at the
# points each codec's 0.075 bpp corpus allocation (Q-hat v0) picked. One lossless WebP sprite per pair:
# [original | msillm | ditic], 1536x512.
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
root = Path.home() / "clic"
merge, out = root / "work_merge_1010_v0", Path(sys.argv[1])
(out / "img").mkdir(parents=True, exist_ok=True)
labels = {"ebfd571f": "screen", "bb7344a2": "screen", "86127fbd": "screen", "2a760bf1": "screen", "937476dd": "texture"}
alloc = {(r["cand"], r["name"]): r for r in csv.DictReader(open(merge / "allocation.csv")) if float(r["rate"]) == 0.075}
recon = {"msillm": root / "work_full_gpu/bakeoff/recon/msillm", "ditic": root / "work_new/bakeoff/recon/ditic"}
names = sorted({n for _, n in alloc})
pairs, S = [], 512
for i, n in enumerate(names):
    ref = np.asarray(Image.open(root / "work_full_gpu/bakeoff/valid" / f"{n}.png").convert("RGB"))
    H, W = ref.shape[:2]
    rec = {c: np.asarray(Image.open(recon[c] / alloc[(c, n)]["key"] / f"{n}.png").convert("RGB")) for c in recon}
    rng = np.random.default_rng(2027 + i)
    got = []
    while len(got) < 3:
        for _ in range(50):  # skip near-flat crops (e.g. pure black): nothing to judge there
            y, x = int(rng.integers(0, H - S + 1)), int(rng.integers(0, W - S + 1))
            if ref[y:y + S, x:x + S].std() > 6 and all(abs(y - yy) > S // 2 or abs(x - xx) > S // 2 for yy, xx in got):
                break
        got.append((y, x))
    for j, (y, x) in enumerate(got):
        pid = f"p{len(pairs):02d}"
        sprite = np.concatenate([a[y:y + S, x:x + S] for a in (ref, rec["msillm"], rec["ditic"])], 1)
        Image.fromarray(sprite).save(out / "img" / f"{pid}.webp", lossless=True, method=6)
        px = H * W
        pairs.append(dict(id=pid, image=n[:8], label=labels.get(n[:8], "natural"), x=x, y=y,
                          msillm=dict(key=alloc[("msillm", n)]["key"], bpp=round(int(alloc[("msillm", n)]["bytes"]) * 8 / px, 4)),
                          ditic=dict(key=alloc[("ditic", n)]["key"], bpp=round(int(alloc[("ditic", n)]["bytes"]) * 8 / px, 4))))
json.dump(dict(rate=0.075, crop=S, order=["original", "msillm", "ditic"], allocation="Q-hat v0, 2026-10-10", pairs=pairs),
          open(out / "pairs.json", "w"), indent=1)
print(len(pairs), "pairs")
