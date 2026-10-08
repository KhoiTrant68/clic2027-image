"""Bundle the repo code into ONE file, clic27_run.py, plus one-cell notebooks, written OUTSIDE the repo
(default ../clic27-artifacts/run).

    python tools/bundle.py

Running the bundle unpacks the code into ./clic27_repo (once per commit) and starts experiments/pipeline.py
with the same arguments:

    python clic27_run.py                       # default stages (check, qhat, bakeoff, pack)
    python clic27_run.py bakeoff --hours 3
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Build output lives OUTSIDE the repo (default: <repo>/../clic27-artifacts/run); override with --out.
DEFAULT_OUT = REPO.parent / "clic27-artifacts" / "run"
INCLUDE = ["src/clic27/**/*.py", "experiments/pipeline.py", "experiments/s1/*.py",
           "experiments/parity/parity.py", "experiments/qhat/*.py",
           "experiments/bakeoff/*.py", "experiments/clic_b/*.py"]

LAUNCHER = '''#!/usr/bin/env python3
"""clic27 pipeline (CLIC 2027), bundled from commit {commit}. One file: copy it anywhere and run

    python clic27_run.py                    # check, qhat, bakeoff, pack
    python clic27_run.py bakeoff --hours 3  # only some stages
    python clic27_run.py --help

Results: <work>/results.zip (on Kaggle also /kaggle/working/results.zip). Re-running resumes.
"""
import base64, io, os, subprocess, sys, zipfile
from pathlib import Path

COMMIT = "{commit}"
PAYLOAD = "{payload}"


def main():
    root = Path(os.environ.get("CLIC27_ROOT", "clic27_repo")).resolve()
    stamp = root / ".commit"
    if not stamp.exists() or stamp.read_text().strip() != COMMIT:
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
        stamp.write_text(COMMIT)
    sys.exit(subprocess.call([sys.executable, str(root / "experiments" / "pipeline.py"), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
'''


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (outside the repo)")
    DIST = ap.parse_args().out
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    files = sorted({p for pat in INCLUDE for p in REPO.glob(pat) if "__pycache__" not in p.parts})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(REPO).as_posix())
    payload = base64.b64encode(buf.getvalue()).decode()
    DIST.mkdir(parents=True, exist_ok=True)
    src = LAUNCHER.format(commit=commit, payload=payload)
    with open(DIST / "clic27_run.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": "%%writefile clic27_run.py\n" + src},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": "# GPU + Internet on. Change the stages/arguments here if needed.\n!python clic27_run.py"}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(DIST / "clic27_run.ipynb", "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    import clic_notebook
    extra = clic_notebook.write(DIST, src, commit)
    print(f"{len(files)} files, commit {commit}: {DIST / 'clic27_run.py'} ({len(src) / 1e3:.0f} kB), clic27_run.ipynb, "
          + ", ".join(extra))


if __name__ == "__main__":
    main()
