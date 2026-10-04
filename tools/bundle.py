"""Bundle the repo code into ONE file: dist/ratflow_run.py (and the same as a one-cell notebook).

    python tools/bundle.py

Running the bundle unpacks the code into ./ratflow_repo (once per commit) and starts experiments/pipeline.py
with the same arguments:

    python ratflow_run.py                       # default stages, see experiments/pipeline.py
    python ratflow_run.py gonogo1 --hours 3
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
INCLUDE = ["src/ratflow/**/*.py", "experiments/pipeline.py", "experiments/s1/*.py",
           "experiments/quant_noise/measure_quant_noise.py", "experiments/quant_noise/denoiser_gonogo.py",
           "experiments/quant_noise/prepare_gonogo_latents.py", "experiments/ceiling/ceiling.py",
           "experiments/parity/parity.py", "experiments/qhat/*.py",
           "experiments/bakeoff/*.py", "experiments/clic_b/*.py"]

LAUNCHER = '''#!/usr/bin/env python3
"""ratflow pipeline (CVPR 2027 / CLIC 2027), bundled from commit {commit}. One file: copy it anywhere and run

    python ratflow_run.py                    # check, data, cache, gonogo1, s1, s1_eval, pack
    python ratflow_run.py gonogo1 --hours 3  # only some stages
    python ratflow_run.py --help

Results: <work>/results.zip (on Kaggle also /kaggle/working/results.zip). Re-running resumes.
"""
import base64, io, os, subprocess, sys, zipfile
from pathlib import Path

COMMIT = "{commit}"
PAYLOAD = "{payload}"


def main():
    root = Path(os.environ.get("RATFLOW_ROOT", "ratflow_repo")).resolve()
    stamp = root / ".commit"
    if not stamp.exists() or stamp.read_text().strip() != COMMIT:
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
        stamp.write_text(COMMIT)
    sys.exit(subprocess.call([sys.executable, str(root / "experiments" / "pipeline.py"), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
'''


def main():
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    files = sorted({p for pat in INCLUDE for p in REPO.glob(pat) if "__pycache__" not in p.parts})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(REPO).as_posix())
    payload = base64.b64encode(buf.getvalue()).decode()
    DIST.mkdir(exist_ok=True)
    src = LAUNCHER.format(commit=commit, payload=payload)
    with open(DIST / "ratflow_run.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": "%%writefile ratflow_run.py\n" + src},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": "# GPU + Internet on. Change the stages/arguments here if needed.\n!python ratflow_run.py"}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(DIST / "ratflow_run.ipynb", "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{len(files)} files, commit {commit}: {DIST / 'ratflow_run.py'} ({len(src) / 1e3:.0f} kB), ratflow_run.ipynb")


if __name__ == "__main__":
    main()
