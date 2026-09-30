import json
from pathlib import Path

root = Path(__file__).resolve().parent
src = (root / "b_analysis.py").read_text(encoding="utf-8")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md(
        "# CLIC 2027, nhánh B: DC-AE hỏng ở đâu trên tập validation?\n\n"
        "**Cài đặt notebook:** Accelerator = **GPU T4 x1** (hoặc P100), Internet = **On**. Sau đó bấm Run All.\n\n"
        "Ước tính khoảng 40–60 phút.\n\n"
        "**Notebook làm gì:**\n"
        "1. Tải 30 ảnh validation của CLIC.\n"
        "2. Kiểm kê ảnh: kích thước, ngân sách byte, OCR, nhãn gợi ý.\n"
        "3. Tái tạo ảnh bằng: trần DC-AE f32c32, trần SD-VAE f8, và HEVC-intra ở 0.075/0.15/0.3 bpp (dùng thay cho HM/VTM).\n"
        "4. Tính PSNR, MS-SSIM, LPIPS, DISTS, PSNR vùng chữ và CER của OCR.\n"
        "5. Cắt crop so sánh và in bảng quyết định B3.\n\n"
        "**Kết quả:** tải `b_results.zip` (gồm csv, summary.md và crops). File `b_recon.zip` (ảnh tái tạo, lớn) là tùy chọn."
    ),
    code(
        "!pip install -q -U diffusers accelerate lpips piq easyocr\n"
        "!ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx265 && echo 'libx265 OK' || echo 'NO libx265 -> HEVC proxy will be skipped'\n"
        "!nvidia-smi --query-gpu=name,memory.total --format=csv"
    ),
    code("%%writefile b_analysis.py\n" + src),
    md("## Từng bước (bước nào lỗi thì chạy lại riêng bước đó)"),
    code("!python b_analysis.py inventory"),
    code("!python b_analysis.py recon"),
    code("!python b_analysis.py metrics"),
    code("!python b_analysis.py crops"),
    code("!python b_analysis.py summary"),
    md("## Xem nhanh"),
    code(
        "import pandas as pd\n"
        "from IPython.display import Markdown, Image as Img, display\n"
        "inv = pd.read_csv('b_out/inventory.csv')\n"
        "display(inv[['name','H','W','colors_per_px','flat_frac','n_text_boxes','text_area_frac','label_suggested']])\n"
        "display(Markdown(open('b_out/summary.md', encoding='utf-8').read()))\n"
        "for n in inv.sort_values('text_area_frac', ascending=False).name[:6]:\n"
        "    print(n); display(Img(f'b_out/crops/{n}.png'))"
    ),
    md(
        "## Nhãn nội dung (làm bằng tay, khuyến khích)\n\n"
        "Xem các crop và danh sách ảnh ở trên, sửa dict `LABELS` bên dưới (`natural` / `game` / `screen`), rồi chạy lại cell này. `summary.md` sẽ được cập nhật theo nhãn mới."
    ),
    code(
        "LABELS = {\n"
        "    # 'ten_anh': 'game',\n"
        "}\n"
        "if LABELS:\n"
        "    inv = pd.read_csv('b_out/inventory.csv')\n"
        "    inv['label'] = inv.name.map(LABELS).fillna(inv['label_suggested'])\n"
        "    inv.to_csv('b_out/inventory.csv', index=False)\n"
        "    import subprocess\n"
        "    print(subprocess.run(['python', 'b_analysis.py', 'summary'], capture_output=True, text=True).stdout)"
    ),
    code(
        "!cd b_out && zip -qr ../b_results.zip *.csv *.json *.md crops && ls -la ../b_results.zip\n"
        "!cd b_out && zip -qr ../b_recon.zip recon && ls -la ../b_recon.zip"
    ),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = root / "b_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)
