"""Build a blind A/B study (tools/blind_score.py) from the decoded images of several runs at one budget.

    python -m clic27.gc.blind --ref VALID_DIR --out STUDY_INPUT --budget 0.15 \
        --run base=RUN_A/encode/bpp-0.15/verify/images --run new=RUN_B/encode/bpp-0.15/verify/images \
        [--crops disagree|random|full] [--per-image 2] [--seed 0]
    python tools/blind_score.py prepare --config STUDY_INPUT/config.json --out STUDY
    python tools/blind_score.py serve --study STUDY

CLIC raters see 512x512 crops, so by default every image contributes `per-image` crops: the crop where the
runs disagree most (largest LPIPS spread, which is where a preference is decided) plus random crops. Any
folder of reconstructions with the same file names can be a "run" (e.g. an older codec for reference).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

CROP = 512


def boxes(H, W, stride=256):
    ys = sorted({min(y, max(H - CROP, 0)) for y in range(0, max(H - CROP, 0) + stride, stride)})
    xs = sorted({min(x, max(W - CROP, 0)) for x in range(0, max(W - CROP, 0) + stride, stride)})
    return [(y, x) for y in ys for x in xs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=folder of decoded PNGs")
    ap.add_argument("--budget", default="")
    ap.add_argument("--crops", default="disagree", choices=["disagree", "random", "full"])
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rnd = random.Random(a.seed)
    runs = dict(r.split("=", 1) for r in a.run)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new folder (avoids mixing studies)")
    refs = {p.stem: p for p in Path(a.ref).rglob("*.png")}
    names = sorted(set(refs).intersection(*[{p.stem for p in Path(d).glob('*.png')} for d in runs.values()]))
    if not names:
        raise SystemExit("no common image names between --ref and the runs")
    scorer = None
    if a.crops == "disagree":
        import torch
        from .losses import LPIPS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = (LPIPS().to(dev), dev)
    for sub in ["reference", *runs]:
        (out / sub).mkdir(parents=True)
    n = 0
    for name in names:
        o = np.array(Image.open(refs[name]).convert("RGB"))
        recs = {k: np.array(Image.open(Path(d) / f"{name}.png").convert("RGB")) for k, d in runs.items()}
        H, W = o.shape[:2]
        if a.crops == "full":
            picks = [(0, 0, H, W)]
        else:
            bs = boxes(H, W)
            picks = []
            if scorer is not None:
                import torch
                lp, dev = scorer
                t = lambda a_: torch.from_numpy(a_).permute(2, 0, 1)[None].float().div(255).to(dev)  # noqa: E731
                spread = []
                with torch.no_grad():
                    for y, x in bs:
                        oc = t(o[y:y + CROP, x:x + CROP])
                        v = [float(lp(t(r[y:y + CROP, x:x + CROP]), oc)) for r in recs.values()]
                        spread.append(max(v) - min(v))
                y, x = bs[int(np.argmax(spread))]
                picks.append((y, x, CROP, CROP))
            rest = [b for b in bs if (b[0], b[1], CROP, CROP) not in picks]
            rnd.shuffle(rest)
            picks += [(y, x, CROP, CROP) for y, x in rest[: max(0, a.per_image - len(picks))]]
        for i, (y, x, h, w) in enumerate(picks):
            fn = f"{name}_{i}.png"
            Image.fromarray(o[y:y + h, x:x + w]).save(out / "reference" / fn)
            for k, r in recs.items():
                Image.fromarray(r[y:y + h, x:x + w]).save(out / k / fn)
            n += 1
    cfg = dict(reference="reference", codecs={k: k for k in runs},
               question=f"So với ảnh gốc, bản nào có chất lượng thị giác tốt hơn?{' (' + a.budget + ' bpp)' if a.budget else ''}")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{n} crops from {len(names)} images -> {out}\nnext: python tools/blind_score.py prepare --config "
          f"{out / 'config.json'} --out <study_dir>")


if __name__ == "__main__":
    main()
