"""Fresh-process decoder check: no original images and no encoder calls."""
import json
from pathlib import Path
import sys

from half_vram import install


def main():
    root = Path.home() / "clic/research_20261015"
    out = root / "fresh_decode"
    out.mkdir(exist_ok=True)
    stop = install(out / "vram.jsonl")
    sys.path.insert(0, str(root / "pulse_vendor"))
    import numpy as np
    from PIL import Image
    from codec.checkpoint import load_model
    from codec.images import save_image
    from codec.runtime import configure_runtime, decompress_image
    configure_runtime("cuda:0", 2)
    model = load_model(root / "weights/pulse/pulse-l-perceptual", "cuda:0")
    rows = []
    for q, prefix in ((0, "02809272"), (1, "937476dd"), (4, "ebfd571f")):
        source = next((root / "full_pulse_large" / f"q{q}").glob(prefix + "*.pulse"))
        recon, ms = decompress_image(model, source.read_bytes())
        target = out / (source.stem + ".png")
        save_image(recon, target)
        old = np.asarray(Image.open(source.with_suffix(".png"))).astype(np.int16)
        new = np.asarray(Image.open(target)).astype(np.int16)
        rows.append(dict(q=q, name=source.stem, same_pixels=bool(np.array_equal(old, new)),
                         max_abs_difference=int(np.max(np.abs(old - new))), decode_ms=ms))
    (out / "checks.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    stop.set()
    if not all(r["same_pixels"] for r in rows):
        raise SystemExit("Fresh decoder mismatch")


if __name__ == "__main__":
    main()
