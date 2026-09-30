"""Build b6_kaggle.ipynb from b_analysis.py + b6_vtm_residual.py (scripts embedded via %%writefile)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
b_src = (root / "b_analysis.py").read_text(encoding="utf-8")
b6_src = (root / "b6_vtm_residual.py").read_text(encoding="utf-8")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md(
        "# CLIC 2027, B6: VTM thật, VTM-SCC, và DC-AE + residual\n\n"
        "**Cài đặt:** Accelerator = **GPU T4 x1** (GPU dùng cho DC-AE và OCR; VTM chạy trên 4 CPU), Internet = **On**.\n\n"
        "**Thời gian:** khoảng 2–4 giờ, gồm build VTM (~10 phút) và khoảng 250–300 lần encode VTM.\n"
        "- Nên chạy bằng **Save Version → Save & Run All (Commit)**. Notebook sẽ chạy nền, tắt trình duyệt cũng không sao.\n"
        "- Mỗi bước tự bỏ qua kết quả đã có, nên nếu hết giờ thì chạy lại là tiếp tục được.\n\n"
        "**Nội dung:** 7 ảnh (4 screen, 3 natural) × 3 mức bpp. So sánh các cấu hình sau:\n"
        "- `vtm420`: VTM 4:2:0, gần với baseline của CLIC.\n"
        "- `vtmscc`: VTM 4:4:4 có bật các công cụ SCC.\n"
        "- `dcae+res`: trần DC-AE, cộng thêm phần dư được mã hóa bằng VTM với số bit còn lại (giả định base tốn 0.03 bpp).\n\n"
        "**Kết quả:** tải `b6_results.zip` về và đặt vào `analysis/`."
    ),
    code(
        "!pip install -q -U diffusers accelerate lpips piq easyocr\n"
        "!nproc; free -g | head -2; nvidia-smi --query-gpu=name --format=csv,noheader"
    ),
    md("## Build VTM 23.8 (cùng phiên bản với decoder trong devkit CLIC)"),
    code(
        "%%bash\n"
        "set -e\n"
        "if [ ! -d vtm ]; then\n"
        "  git clone -q --depth 1 --branch VTM-23.8 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm \\\n"
        "  || git clone -q --depth 1 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm\n"
        "fi\n"
        "cd vtm && git log -1 --format='%h %d %s' && mkdir -p build && cd build\n"
        "cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j$(nproc) EncoderApp 2>&1 | tail -2\n"
        "find ../bin -name 'EncoderApp*' -type f"
    ),
    code("%%writefile b_analysis.py\n" + b_src),
    code("%%writefile b6_vtm_residual.py\n" + b6_src),
    md("## Các bước"),
    code("!python b6_vtm_residual.py prepare"),
    code("!python b6_vtm_residual.py dcae"),
    code("!python b6_vtm_residual.py vtm   # lâu nhất: ~2–3 giờ trên 4 CPU"),
    code("!python b6_vtm_residual.py inventory"),
    code("!python b6_vtm_residual.py metrics"),
    code("!python b6_vtm_residual.py crops"),
    code("!python b6_vtm_residual.py summary"),
    md("## Xem nhanh"),
    code(
        "from IPython.display import Markdown, Image as Img, display\n"
        "import glob\n"
        "display(Markdown(open('b6_out/summary.md', encoding='utf-8').read()))\n"
        "for p in sorted(glob.glob('b6_out/crops/*.png')):\n"
        "    print(p); display(Img(p))"
    ),
    code(
        "!cd b6_out && zip -qr ../b6_results.zip *.csv *.json *.md crops && ls -la ../b6_results.zip\n"
        "!cd b6_out && zip -qr ../b6_recon.zip recon && ls -la ../b6_recon.zip"
    ),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = root / "b6_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
