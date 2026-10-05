"""CLIC week 1-3 runners, written by tools/bundle.py next to ratflow_run.py:

    dist/clic_qhat_bakeoff.ipynb   Kaggle notebook: settings, preflight checks, run, results shown inline
    dist/run_clic.sh               the same on a rented GPU machine (L4 etc.)
"""
from __future__ import annotations

import json
from pathlib import Path

INTRO = """# CLIC 2027: Q̂ v0 + so sánh base (commit `{commit}`)

Notebook này chạy hai việc của plan v2 trong một lần:

1. **`qhat`**: fit Q̂ v0, thước đo thay người chấm, trên dữ liệu chấm theo cặp của CLIC 2021/2022/2024 (khoảng 1.5–2 giờ).
2. **`bakeoff`**: so sánh các base C0 VTM / VTM-SCC, C1 DC-AE + S1 + residual, C2 MS-ILLM, C3 mbt2018 trên 30 ảnh validation,
   ở 0.075 / 0.15 / 0.3 bpp, chia bit cho cả bộ ảnh theo Q̂ (khoảng 4–5 giờ, phần lâu nhất là encode VTM trên CPU).

## Cài đặt trước khi chạy (một lần)

1. **Settings → Accelerator → GPU T4 x1** (P100 cũng được).
2. **Settings → Internet → On** (cần tài khoản đã xác minh số điện thoại).
3. **Add Input → Datasets → New Dataset**: upload file `results/s1_results.zip` từ máy bạn (120 MB, chứa `s1_out/last.pt`
   của S1 lần 1, phủ tới 0.12 bpp). Kaggle tự giải nén. Lần sau chỉ cần Add Input dataset đó.
4. **Chạy dài (> 1 giờ): bấm Save Version → Save & Run All (Commit)**, không chạy tương tác,
   vì phiên tương tác sẽ dừng khi đóng trình duyệt. Kết quả nằm ở tab **Output**: `results.zip`.

**Hết giờ giữa chừng?** Tạo phiên mới, **Add Input → output của lần chạy trước**, rồi chạy lại: các bước đã xong sẽ được bỏ qua.
Gửi `results.zip` cho Claude để phân tích.
"""

CONFIG = """# ---- Cấu hình: chỉ cần sửa ô này ----
HOURS = 11.0            # giới hạn của cả lần chạy; hết giờ thì dừng gọn, đóng gói, Kaggle vẫn lưu Output (Kaggle giết ở 12 giờ)
STAGES = "qhat bakeoff"  # chỉ chạy một phần: "qhat" hoặc "bakeoff"
QHAT_ARGS = ""          # ví dụ "--n 2024t=8000 2022t=2000" để đổi cỡ mẫu
BAKEOFF_ARGS = ""       # ví dụ "--cands msillm mbt" để chạy nhanh các codec học được trước (~30 phút)
BAKEOFF_STEP = "all"    # "points" = chỉ tạo điểm nén (dùng cho phiên CPU chạy VTM)
CPU_ONLY = False        # True = phiên KHÔNG bật GPU (không tốn hạn mức GPU), chỉ hợp với VTM:
                        #   STAGES="bakeoff", BAKEOFF_STEP="points", BAKEOFF_ARGS="--cands vtm420" (phiên khác: vtmscc)
"""

PREFLIGHT = """# Kiểm tra trước khi chạy: GPU, Internet, checkpoint S1, lần chạy trước
import json, socket
from pathlib import Path

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ  <- bật GPU trong Settings")

def online(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False
print("Internet:", "OK" if all(online(h) for h in ("pypi.org", "huggingface.co", "downloads.compression.cc"))
      else "KHÔNG CÓ  <- bật Internet trong Settings")

found = []
for ck in Path("/kaggle/input").rglob("last.pt"):
    log = ck.parent / "log.json"
    lam = max(json.loads(log.read_text())["lambdas"]) if log.exists() else None
    found.append((ck, lam))
    print(f"S1 checkpoint: {ck}  (lambda max = {lam})")
if not any(lam and lam >= 2 for _, lam in found):
    print("CHƯA CÓ checkpoint S1 lần 1 (lambda max = 4). Ứng viên C1 (s1res) sẽ bị bỏ qua; các ứng viên khác vẫn chạy.\\n"
          "-> Add Input: dataset từ results/s1_results.zip")
prev = [p.parent for p in Path("/kaggle/input").rglob("state.json")]
print("Lần chạy trước (sẽ tiếp tục từ đây):", prev or "không có")
"""

RUN = """# Chạy (log hiện trực tiếp bên dưới). Chạy lại ô này sau khi hết giờ: các bước đã xong được bỏ qua.
import shlex, sys
cmd = [sys.executable, "ratflow_run.py", *STAGES.split(), "--hours", str(HOURS)]
if QHAT_ARGS:
    cmd.append("--qhat-args=" + QHAT_ARGS)
if BAKEOFF_ARGS:
    cmd.append("--bakeoff-args=" + BAKEOFF_ARGS)
if BAKEOFF_STEP != "all":
    cmd += ["--bakeoff-step", BAKEOFF_STEP]
if CPU_ONLY:
    cmd.append("--allow-cpu")
line = " ".join(shlex.quote(c) for c in cmd)
print(line)
get_ipython().system(line)
"""

SHOW = """# Kết quả
from pathlib import Path
from IPython.display import Image, Markdown, display

W = Path("/kaggle/working/work")
for f in ["qhat/summary.md", "bakeoff/summary.md"]:
    p = W / f
    display(Markdown(p.read_text(encoding="utf-8") if p.exists() else f"*{f}: chưa có*"))
for p in sorted((W / "bakeoff" / "crops").glob("*.jpg")):
    print(p.name, "(cột đầu là ảnh gốc, các cột sau là từng ứng viên ở cùng ngân sách)")
    display(Image(filename=str(p), width=1400))
z = Path("/kaggle/working/results.zip")
print("results.zip:", f"{z.stat().st_size / 1e6:.1f} MB -> tab Output" if z.exists() else "chưa có")
"""

SHELL = """#!/usr/bin/env bash
# CLIC 2027: Q-hat v0 + base bake-off on a rented GPU machine (commit {commit}).
#   Needs: python3 with torch + CUDA, git, cmake, g++, wget, unzip  (Ubuntu: apt install -y git cmake g++ wget unzip)
#   Put next to this file: ratflow_run.py and s1_results.zip (results/s1_results.zip from the repo checkout)
#
#   bash run_clic.sh                         # everything, resumes when run again
#   HOURS=3 EXTRA='--bakeoff-args="--cands msillm mbt"' bash run_clic.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p inputs
if [ -f s1_results.zip ] && [ ! -d inputs/s1_results ]; then unzip -q s1_results.zip -d inputs/s1_results; fi
eval python3 ratflow_run.py qhat bakeoff --work work --inputs inputs --hours "${{HOURS:-24}}" ${{EXTRA:-}}
echo "Kết quả: $(pwd)/work/results.zip  (gửi file này cho Claude)"
"""


def _cell(kind, src, hidden=False):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
        if hidden:
            c["metadata"] = {"jupyter": {"source_hidden": True}, "collapsed": True}
    return c


def write(dist: Path, launcher_src: str, commit: str):
    cells = [_cell("markdown", INTRO.format(commit=commit)), _cell("code", CONFIG), _cell("code", PREFLIGHT),
             _cell("markdown", "Ô dưới đây chỉ ghi file `ratflow_run.py` (toàn bộ code đã đóng gói); không cần đọc."),
             _cell("code", "%%writefile ratflow_run.py\n" + launcher_src, hidden=True),
             _cell("code", RUN), _cell("code", SHOW)]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(dist / "clic_qhat_bakeoff.ipynb", "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    with open(dist / "run_clic.sh", "w", encoding="utf-8", newline="\n") as f:
        f.write(SHELL.format(commit=commit))
    return ["clic_qhat_bakeoff.ipynb", "run_clic.sh"]
