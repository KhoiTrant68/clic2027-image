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
PKG = sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in (REPO / "src/ratflow").rglob("*.py"))
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
    "parity": dict(
        title="Parity: ratflow.nn / ratflow.entropy so với diffusers, và CPU so với GPU",
        intro=f"**Cài đặt:** {KAGGLE_T4}. Mất khoảng 20–30 phút (tải DC-AE, SANA 0.6B và 1.6B).\n\n"
              "Kiểm tra: DC-AE (thường và theo tile) và SANA DiT viết lại bằng torch thuần phải khớp diffusers; "
              "h_s số nguyên phải khớp từng bit giữa CPU và GPU; nén/giải nén y và z phải khớp hoàn toàn.\n\n"
              "**Kết quả:** `parity_results.zip`, đặt vào `results/parity/`.",
        files=PKG + ["experiments/parity/parity.py"],
        cells=[
            ("code", "!pip install -q -U diffusers accelerate safetensors huggingface_hub\n"
                     "!nvidia-smi --query-gpu=name --format=csv"),
            ("code", "P = 'experiments/parity/parity.py'\n!python {P} dcae"),
            ("code", "!python {P} dit"),
            ("code", "!python {P} entropy"),
            ("code", "!python {P} summary"),
            ("code", "!cd parity_out && zip -q ../parity_results.zip * && ls -la ../parity_results.zip"),
        ]),
    "s1_cache": dict(
        title="CVPR S1 bước 1: cache latent DC-AE cho dữ liệu train",
        intro=f"**Cài đặt:** {KAGGLE_T4}. {COMMIT_RUN}\n\n"
              "Tải DIV2K train (800 ảnh, 3.5 GB) rồi encode toàn ảnh bằng DC-AE (fp16, theo tile), mỗi ảnh lưu một file `.npy` "
              "trong `latents/` (~140 MB). Mất khoảng 30–60 phút.\n\n"
              "**Thêm dữ liệu (không bắt buộc):** gắn dataset Flickr2K hoặc LSDIR của Kaggle vào notebook (Add Input), "
              "rồi điền đường dẫn vào `EXTRA_DIRS` ở cell dưới.\n\n"
              "**Kết quả:** thư mục `latents/` trong Output. Ở notebook `s1_train`, chọn **Add Input → Notebook Output** để dùng nó.\n\n"
              "**Nếu lần trước bị ngắt:** gắn Output của lần đó làm Input; notebook chép các latent đã có và chỉ encode phần còn thiếu.",
        files=PKG + ["experiments/s1/cache_latents.py"],
        cells=[
            ("code", "!pip install -q -U safetensors huggingface_hub\n!nvidia-smi --query-gpu=name --format=csv"),
            ("code", "import glob, os, shutil, torch\n"
                     "assert torch.cuda.is_available(), 'Chưa bật GPU (hoặc hết hạn mức GPU tuần này): trên CPU mất ~4 phút/ảnh'\n"
                     "os.makedirs('latents', exist_ok=True)\n"
                     "prev = glob.glob('/kaggle/input/**/latents/*.npy', recursive=True)  # resume a cut-off run\n"
                     "for p in prev:\n"
                     "    if not os.path.exists('latents/' + os.path.basename(p)):\n"
                     "        shutil.copy(p, 'latents/')\n"
                     "print('resumed', len(prev), 'latents')"),
            ("code", "EXTRA_DIRS = []  # ví dụ: ['/kaggle/input/flickr2k/Flickr2K_HR']\n"
                     "extra = ' '.join(EXTRA_DIRS)\n"
                     "!python experiments/s1/cache_latents.py --download-div2k --images {extra} --out latents --dtype fp16"),
            ("code", "!rm -rf data  # giữ Output gọn: chỉ để lại latents/\n!du -sh latents; ls latents | wc -l"),
        ]),
    "s1_train": dict(
        title="CVPR S1 bước 2: selftest, train codec latent đa rate, eval trên Kodak",
        intro=f"**Cài đặt:** {KAGGLE_T4}. **Add Input → Notebook Output** của `s1_cache`. {COMMIT_RUN}\n\n"
              "1. `selftest`: model ngẫu nhiên, nén trên GPU và giải nén trên CPU. Phải ra **PASS** thì notebook mới chạy tiếp.\n"
              "2. `train`: tối đa 9.5 giờ, tự lưu `s1_out/last.pt` mỗi 5k bước. Nếu bị ngắt, chạy lại sẽ tiếp tục từ checkpoint "
              "(nhớ gắn Output của lần trước làm Input và chép `last.pt` vào `s1_out/`).\n"
              "3. `eval`: 24 ảnh Kodak × 8 mức rate, bitstream thật, so với trần DC-AE.\n\n"
              "**Kết quả:** `s1_results.zip` (log, summary, csv và checkpoint), đặt vào `results/s1/`.",
        files=PKG + ["experiments/s1/train_s1.py", "experiments/s1/eval_s1.py"],
        cells=[
            ("code", "!pip install -q -U safetensors huggingface_hub lpips piq\n!nvidia-smi --query-gpu=name --format=csv"),
            ("code", "import glob, os, json\n"
                     "cands = sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/latents/*.npy', recursive=True)})\n"
                     "LATENTS = cands[0] if cands else 'latents'\n"
                     "print('latents:', LATENTS, len(glob.glob(LATENTS + '/*.npy')), 'files')\n"
                     "prev = glob.glob('/kaggle/input/**/s1_out/last.pt', recursive=True)  # resume from a previous run\n"
                     "if prev and not os.path.exists('s1_out/last.pt'):\n"
                     "    os.makedirs('s1_out', exist_ok=True); os.system(f'cp {prev[0]} s1_out/last.pt'); print('resume from', prev[0])"),
            ("code", "!python experiments/s1/train_s1.py selftest --out s1_out\n"
                     "assert json.load(open('s1_out/selftest.json'))['ok'], 'SELFTEST FAIL: dừng lại, gửi log cho Claude'"),
            ("code", "!python experiments/s1/train_s1.py train --latents {LATENTS} --out s1_out --hours 9.5"),
            ("code", "!python experiments/s1/eval_s1.py --ckpt s1_out/last.pt --dataset kodak --out s1_eval"),
            ("code", "from IPython.display import Markdown, display\n"
                     "display(Markdown(open('s1_eval/summary.md', encoding='utf-8').read()))"),
            ("code", "!zip -qr s1_results.zip s1_out/log.json s1_out/selftest.json s1_out/last.pt s1_eval/*.md s1_eval/*.json s1_eval/*.csv "
                     "&& ls -la s1_results.zip"),
        ]),
    "gonogo1": dict(
        title="CVPR go/no-go 1 (hạn 04/10): decoder biết mô hình nhiễu chính xác có thắng decoder giả định Gaussian?",
        intro=f"**Cài đặt:** {KAGGLE_T4}. **Add Input → Notebook Output** của `s1_cache` (dùng lại latent DIV2K, không encode lại). "
              f"{COMMIT_RUN}\n\n"
              "Tiêu chí (đã duyệt 27/9): trên latent DC-AE thật, mã hóa bằng bộ mã giả lập KLT 2×2 có subtractive dither ở 4 mức 0.01–0.05 bpp. "
              "Cùng một CNN khử nhiễu, train với **nhiễu chính xác** (đều, có dither) phải có MSE latent thấp hơn bản train với **giả định Gaussian** "
              "(kiểu OSCAR) **≥ 5% ở ≥ 2/4 mức rate**, trên Kodak và CLIC2020 professional valid. Mất khoảng 1–1.5 giờ.\n\n"
              "**Kết quả:** `gonogo1_results.zip`, đặt vào `results/gonogo1/`.",
        files=PKG + ["experiments/quant_noise/measure_quant_noise.py", "experiments/quant_noise/denoiser_gonogo.py",
                     "experiments/quant_noise/prepare_gonogo_latents.py"],
        cells=[
            ("code", "!pip install -q -U safetensors huggingface_hub scipy\n!nvidia-smi --query-gpu=name --format=csv"),
            ("code", "import glob, os\n"
                     "cands = sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/latents/*.npy', recursive=True)})\n"
                     "assert cands, 'Chưa gắn Output của s1_cache làm Input'\n"
                     "CACHE = cands[0]; print('cache:', CACHE, len(glob.glob(CACHE + '/*.npy')), 'files')\n"
                     "Q = 'experiments/quant_noise'"),
            ("code", "!python {Q}/prepare_gonogo_latents.py --from-cache {CACHE} --out runs/div2k\n"
                     "!python {Q}/prepare_gonogo_latents.py --kodak --out runs/kodak\n"
                     "!python {Q}/prepare_gonogo_latents.py --clic-valid --out runs/clic2020_valid"),
            ("code", "!cd {Q} && python denoiser_gonogo.py --train ../../runs/div2k --test ../../runs/kodak ../../runs/clic2020_valid "
                     "--out ../../runs/gonogo1 --device cuda"),
            ("code", "from IPython.display import Markdown, display\n"
                     "display(Markdown(open('runs/gonogo1/summary.md', encoding='utf-8').read()))"),
            ("code", "!cd runs/gonogo1 && zip -q ../../gonogo1_results.zip * && ls -la ../../gonogo1_results.zip"),
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
