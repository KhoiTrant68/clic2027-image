"""Build the CLIC decoder folder / decoder.zip from a trained model.pt.

decoder/
  decode            shell entry (CLIC devkit calls ./decode in a folder holding bs.zip; writes images/*.png)
  decode_main.py
  lib/clic27/{__init__.py, entropy/*.py, gc/*.py, gc/_rans.c, gc/_rans.so}
  model.pt          decoder-side weights only (analysis transforms stripped), runtime settings embedded
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]  # src/clic27

DECODE_SH = """#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$DIR/decode_main.py" --input bs.zip --output images "$@"
"""


def strip_model(src: Path, dst: Path, runtime: dict):
    sd = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: v for k, v in sd["state_dict"].items() if not k.startswith(("g_a.", "h_a."))}
    torch.save(dict(cfg=sd["cfg"], state_dict=keep, runtime=runtime), dst)


def build(model_pt: Path, out_dir: Path, runtime: dict, zip_path: Path | None = None) -> Path:
    from . import rans_c
    if out_dir.exists():
        shutil.rmtree(out_dir)
    lib = out_dir / "lib" / "clic27"
    (lib / "entropy").mkdir(parents=True)
    (lib / "gc").mkdir(parents=True)
    (lib / "__init__.py").write_text("")
    for p in (SRC / "entropy").glob("*.py"):
        shutil.copy2(p, lib / "entropy" / p.name)
    for name in ("__init__.py", "model.py", "layers.py", "codec.py", "rans_c.py", "_rans.c", "decode_main.py"):
        shutil.copy2(SRC / "gc" / name, lib / "gc" / name)
    # The CLIC server has no compiler: a package without _rans.so would fall back to numpy rANS (far too slow).
    try:
        rans_c.build(lib / "gc" / "_rans.so")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[package] cannot build _rans.so ({e}): install gcc (or set CC) and re-run the package stage")
    print("[package] _rans.so built here; it must load on the server's glibc (check with scripts/gc/l4_verify.sh)",
          flush=True)
    shutil.copy2(SRC / "gc" / "decode_main.py", out_dir / "decode_main.py")
    (out_dir / "decode").write_text(DECODE_SH)
    (out_dir / "decode").chmod(0o755)
    strip_model(model_pt, out_dir / "model.pt", runtime)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zi = zipfile.ZipInfo.from_file(p, p.relative_to(out_dir).as_posix())
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    with open(p, "rb") as f:
                        z.writestr(zi, f.read())
    return out_dir


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bs_zip(streams: dict[str, bytes], path: Path) -> int:
    """Stored (uncompressed) zip of <name>.gc files, deterministic timestamps. Returns its size in bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(streams):
            zi = zipfile.ZipInfo(f"{name}.gc", date_time=(2027, 1, 1, 0, 0, 0))
            z.writestr(zi, streams[name])
    return path.stat().st_size
