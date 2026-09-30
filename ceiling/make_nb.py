"""Build ceiling_kaggle.ipynb with ceiling.py embedded via %%writefile."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
src = (root / "ceiling.py").read_text(encoding="utf-8")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md(
        "# CVPR T2: trần DC-AE f32 và SD-VAE f8 trên Kodak, CLIC2020 và DIV2K-val\n\n"
        "**Cài đặt:** Accelerator = **GPU T4 x1**, Internet = **On**. Chạy bằng **Save Version → Save & Run All (Commit)**.\n\n"
        "**Thời gian:** khoảng 1.5–2.5 giờ.\n"
        "- Tải ~2 GB dữ liệu.\n"
        "- Encode rồi decode 552 ảnh × 2 autoencoder.\n"
        "- Tính các metric, rồi FID/KID trên patch.\n\n"
        "Mỗi bước tự bỏ qua phần đã làm, nên nếu hết giờ thì chạy lại là tiếp tục được.\n\n"
        "**Kết quả:** tải `ceiling_results.zip` về và đặt vào `ratflow-codec/ceiling/`."
    ),
    code("!pip install -q -U diffusers accelerate lpips piq torchmetrics torch-fidelity\n"
         "!nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working | tail -1"),
    code("%%writefile ceiling.py\n" + src),
    code("!python ceiling.py download"),
    code("!python ceiling.py recon"),
    code("!python ceiling.py metrics"),
    code("!python ceiling.py fid"),
    code("!python ceiling.py summary"),
    code("from IPython.display import Markdown, display\n"
         "display(Markdown(open('ceiling_out/summary.md', encoding='utf-8').read()))"),
    code("!cd ceiling_out && zip -q ../ceiling_results.zip *.csv *.json *.md && ls -la ../ceiling_results.zip\n"
         "# ảnh tái tạo (lớn, không bắt buộc):\n"
         "# !cd ceiling_out && zip -qr ../ceiling_recon.zip recon"),
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = root / "ceiling_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
