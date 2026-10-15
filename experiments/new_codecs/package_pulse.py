"""Build a relocatable PULSE decoder kit, excluding Python/Torch system runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--license", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.runtime / "codec", a.out / "runtime/codec",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extensions = list(a.runtime.glob("MLCodec_extensions_cpp*.so"))
    if len(extensions) != 1:
        raise ValueError("Expected exactly one Linux PULSE rANS extension")
    shutil.copy2(extensions[0], a.out / "runtime")
    shutil.copytree(a.bundle, a.out / "weights")
    shutil.copy2(a.license, a.out / "LICENSE.PULSE")
    for src, dst in (("experiments/new_codecs/pulse_decode.py", "decode.py"),
                     ("experiments/new_codecs/half_vram.py", "half_vram.py"),
                     ("experiments/bakeoff/pulse_codec.py", "pulse_codec.py")):
        shutil.copy2(a.repo / src, a.out / dst)
    manifest = dict(runtime="Linux x86_64, Python 3.12, Torch 2.6, NumPy, Pillow; NOT bundled",
                    scope="research deployment probe, not yet a CLIC submission", sha256={})
    for p in sorted(a.out.rglob("*")):
        if p.is_file():
            manifest["sha256"][p.relative_to(a.out).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = a.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(a.out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(a.out).as_posix())
    print(json.dumps(dict(zip_bytes=archive.stat().st_size,
                          unpacked_bytes=sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file()),
                          zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
