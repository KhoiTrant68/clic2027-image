"""Build the self-contained Kaggle notebooks from repo files.

Each notebook embeds the files it needs with %%writefile at their repo paths (src/ratflow/...,
experiments/...), so scripts run exactly as in the repo, and records the git commit it was built from.

    python notebooks/build.py            # all notebooks
    python notebooks/build.py ceiling    # one
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks"
PKG = ["src/ratflow/__init__.py", "src/ratflow/eval/__init__.py", "src/ratflow/eval/metrics.py"]
KAGGLE_T4 = "Accelerator = **GPU T4 x1**, Internet = **On**"
COMMIT_RUN = "Chạy bằng **Save Version → Save & Run All (Commit)** để notebook chạy nền; mỗi bước tự bỏ qua phần đã làm, nên hết giờ thì chạy lại là tiếp tục."

TORCH26 = (
    "%%bash\n"
    "set -e\n"
    "if python -m venv /kaggle/working/v26 2>/dev/null && [ -x /kaggle/working/v26/bin/pip ]; then\n"
    "  /kaggle/working/v26/bin/pip install -q torch==2.6.0 numpy==1.26.4\n"
    "  ln -sf /kaggle/working/v26/bin/python /kaggle/working/py26\n"
    "else\n"
    "  echo 'venv unavailable -> pip --target'\n"
    "  pip install -q --target /kaggle/working/t26 torch==2.6.0 numpy==1.26.4\n"
    "  printf '#!/bin/bash\\nPYTHONPATH=/kaggle/working/t26 exec python \"$@\"\\n' > /kaggle/working/py26 && chmod +x /kaggle/working/py26\n"
    "fi\n"
    "/kaggle/working/py26 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'"
)

VTM_BUILD = (
    "%%bash\n"
    "set -e\n"
    "if [ ! -d vtm ]; then\n"
    "  git clone -q --depth 1 --branch VTM-23.8 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm \\\n"
    "  || git clone -q --depth 1 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm\n"
    "fi\n"
    "cd vtm && git log -1 --format='%h %d %s' && mkdir -p build && cd build\n"
    "cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j$(nproc) EncoderApp 2>&1 | tail -2\n"
    "find ../bin -name 'EncoderApp*' -type f"
)

# name -> dict(title, intro, files, cells); cells are ("md"|"code", source)
SPECS = {
    "determinism_a0": dict(
        title="CLIC 2027 A0: kiểm tra độ tất định của entropy model",
        intro=f"**Cài đặt:** {KAGGLE_T4} (có thể chạy lại với **GPU P100**), rồi bấm Run All. Mất khoảng 5–8 phút.\n\n"
              "**Kết quả:** `a0_<GPU>.zip`, đặt vào `results/determinism/`. Ở mục *Verdict*, `intsim` phải OK.",
        files=["experiments/determinism/probe.py", "experiments/determinism/compare.py"],
        cells=[
            ("code", "import subprocess, torch\n"
                     "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"
                     "GPU = torch.cuda.get_device_name(0).replace('Tesla ', '').replace(' ', '_')\n"
                     "print('GPU tag:', GPU, '| system torch', torch.__version__, torch.version.cuda)"),
            ("md", "## 1. Torch có sẵn của Kaggle: chạy GPU 2 lần, rồi chạy CPU"),
            ("code", "P = 'experiments/determinism/probe.py'\n"
                     "!python {P} --device cuda --tag {GPU}_sys_run1\n"
                     "!python {P} --device cuda --tag {GPU}_sys_run2\n"
                     "!python {P} --device cpu  --tag {GPU}host_sys_cpu"),
            ("md", "## 2. torch 2.6.0 + numpy 1.26.4 (đúng phiên bản của devkit CLIC)"),
            ("code", TORCH26),
            ("code", "!/kaggle/working/py26 {P} --device cuda --tag {GPU}_t26\n"
                     "!/kaggle/working/py26 {P} --device cpu  --tag {GPU}host_t26_cpu"),
            ("md", "## 3. So sánh"),
            ("code", "!python experiments/determinism/compare.py out/*.npz"),
            ("code", "!cd out && zip -q ../a0_{GPU}.zip *.npz && ls -la ../a0_{GPU}.zip"),
        ]),
    "clic_b": dict(
        title="CLIC 2027 nhánh B: DC-AE hỏng ở đâu trên tập validation",
        intro=f"**Cài đặt:** {KAGGLE_T4}. Mất khoảng 40–60 phút.\n\n"
              "Các bước: kiểm kê và OCR; tái tạo bằng trần DC-AE, trần SD-VAE f8 và HEVC-intra ở 0.075/0.15/0.3 bpp; "
              "tính PSNR, MS-SSIM, LPIPS, DISTS, text-PSNR, OCR CER; cắt crop; bảng quyết định B3.\n\n"
              "**Kết quả:** `b_results.zip`, đặt vào `results/clic_b/`.",
        files=PKG + ["experiments/clic_b/b_analysis.py"],
        cells=[
            ("code", "!pip install -q -U diffusers accelerate lpips piq easyocr\n"
                     "!ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx265 && echo 'libx265 OK' || echo 'NO libx265 -> HEVC proxy skipped'"),
            ("code", "B = 'experiments/clic_b/b_analysis.py'\n!python {B} inventory"),
            ("code", "!python {B} recon"),
            ("code", "!python {B} metrics"),
            ("code", "!python {B} crops"),
            ("code", "!python {B} summary"),
            ("code", "from IPython.display import Markdown, display\n"
                     "display(Markdown(open('b_out/summary.md', encoding='utf-8').read()))"),
            ("code", "!cd b_out && zip -qr ../b_results.zip *.csv *.json *.md crops && ls -la ../b_results.zip"),
        ]),
    "clic_b6": dict(
        title="CLIC 2027 B6: VTM thật, VTM-SCC, và DC-AE + residual",
        intro=f"**Cài đặt:** {KAGGLE_T4}. GPU dùng cho DC-AE và OCR, VTM chạy trên 4 CPU. Mất khoảng 2–4 giờ. {COMMIT_RUN}\n\n"
              "**Kết quả:** `b6_results.zip`, đặt vào `results/clic_b/`.",
        files=PKG + ["experiments/clic_b/b_analysis.py", "experiments/clic_b/b6_vtm_residual.py"],
        cells=[
            ("code", "!pip install -q -U diffusers accelerate lpips piq easyocr\n!nproc"),
            ("md", "## Build VTM 23.8 (cùng phiên bản với decoder trong devkit CLIC)"),
            ("code", VTM_BUILD),
            ("code", "B6 = 'experiments/clic_b/b6_vtm_residual.py'\n!python {B6} prepare"),
            ("code", "!python {B6} dcae"),
            ("code", "!python {B6} vtm   # lâu nhất"),
            ("code", "!python {B6} inventory"),
            ("code", "!python {B6} metrics"),
            ("code", "!python {B6} crops"),
            ("code", "!python {B6} summary"),
            ("code", "!cd b6_out && zip -qr ../b6_results.zip *.csv *.json *.md crops && ls -la ../b6_results.zip"),
        ]),
    "ceiling": dict(
        title="CVPR T2: trần DC-AE f32 và SD-VAE f8 trên Kodak, CLIC2020 và DIV2K-val",
        intro=f"**Cài đặt:** {KAGGLE_T4}. Mất khoảng 1.5–2.5 giờ (tải ~2 GB; 552 ảnh × 2 AE; tính metric; FID/KID trên patch). {COMMIT_RUN}\n\n"
              "**Kết quả:** `ceiling_results.zip`, đặt vào `results/ceiling/`.",
        files=PKG + ["experiments/ceiling/ceiling.py"],
        cells=[
            ("code", "!pip install -q -U diffusers accelerate lpips piq torchmetrics torch-fidelity\n"
                     "!nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working | tail -1"),
            ("code", "C = 'experiments/ceiling/ceiling.py'\n!python {C} download"),
            ("code", "!python {C} recon"),
            ("code", "!python {C} metrics"),
            ("code", "!python {C} fid"),
            ("code", "!python {C} summary"),
            ("code", "from IPython.display import Markdown, display\n"
                     "display(Markdown(open('ceiling_out/summary.md', encoding='utf-8').read()))"),
            ("code", "!cd ceiling_out && zip -q ../ceiling_results.zip *.csv *.json *.md && ls -la ../ceiling_results.zip"),
        ]),
}


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def cell(kind, src):
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src}
    if kind != "md":
        c.update(execution_count=None, outputs=[])
    return c


def build(name, spec, rev):
    dirs = sorted({str(Path(f).parent).replace("\\", "/") for f in spec["files"]})
    cells = [cell("md", f"# {spec['title']}\n\n{spec['intro']}\n\n"
                        f"*Sinh bởi `notebooks/build.py` từ commit `{rev}` của repo; đừng sửa tay notebook này.*"),
             cell("code", "!mkdir -p " + " ".join(dirs))]
    for f in spec["files"]:
        cells.append(cell("code", f"%%writefile {f}\n" + (REPO / f).read_text(encoding="utf-8")))
    cells += [cell(k, s) for k, s in spec["cells"]]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = OUT / f"{name}.ipynb"
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on Windows too (.gitattributes)
        f.write(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPECS)
    rev = git_rev()
    for n in names:
        build(n, SPECS[n], rev)
