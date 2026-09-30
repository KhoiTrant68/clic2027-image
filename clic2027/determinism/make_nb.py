import json
from pathlib import Path

root = Path(__file__).resolve().parent
probe = (root / "probe.py").read_text(encoding="utf-8")
compare = (root / "compare.py").read_text(encoding="utf-8")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md(
        "# CLIC 2027: A0, kiểm tra độ tất định của entropy model\n\n"
        "**Cài đặt notebook:** Accelerator = **GPU T4 x1** (lần 2 đổi sang **GPU P100**), Internet = **On**. Sau đó bấm Run All.\n\n"
        "Mỗi lần chạy mất khoảng 5–8 phút, phần lớn là thời gian cài torch 2.6.0.\n\n"
        "**Kết quả:** file `a0_<GPU>.zip` trong Output. Tải file này về để so sánh với kết quả chạy trên L4 sau này.\n\n"
        "**Cách đọc:** cuối notebook có mục *Verdict*.\n"
        "- `intsim` phải OK.\n"
        "- Với các chế độ `float*`, xem chúng có lệch giữa GPU và CPU, và giữa torch hệ thống với torch 2.6.0 không."
    ),
    code(
        "import subprocess, torch\n"
        "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"
        "GPU = torch.cuda.get_device_name(0).replace('Tesla ', '').replace(' ', '_')\n"
        "print('GPU tag:', GPU, '| system torch', torch.__version__, torch.version.cuda)"
    ),
    code("%%writefile probe.py\n" + probe),
    code("%%writefile compare.py\n" + compare),
    md("## 1. Torch có sẵn của Kaggle: chạy GPU 2 lần (kiểm tra độ ổn định trên cùng máy), rồi chạy CPU"),
    code(
        "!python probe.py --device cuda --tag {GPU}_sys_run1\n"
        "!python probe.py --device cuda --tag {GPU}_sys_run2\n"
        "!python probe.py --device cpu  --tag {GPU}host_sys_cpu"
    ),
    md("## 2. torch 2.6.0 + numpy 1.26.4, đúng phiên bản trong `requirements.txt` của devkit CLIC"),
    code(
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
    ),
    code(
        "!/kaggle/working/py26 probe.py --device cuda --tag {GPU}_t26\n"
        "!/kaggle/working/py26 probe.py --device cpu  --tag {GPU}host_t26_cpu"
    ),
    md("## 3. So sánh các lần chạy"),
    code("!python compare.py out/*.npz"),
    code("!cd out && zip -q ../a0_{GPU}.zip *.npz && ls -la ../a0_{GPU}.zip"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = root / "a0_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
