"""Bake-off slides (2026-10-08 merge of sessions A, B, c, d, e) in the style of the 'ratflow-codec · CLIC 2027' deck,
to import into Google Slides (File -> Import slides). Native charts and tables so numbers stay editable."""
import sys

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = sys.argv[1]
BODY, MONO = "Be Vietnam Pro", "JetBrains Mono"
INK, BG, CARD, HEAD, GRID = "13212E", "F5F3EE", "FCFBF8", "E9E4DA", "E4E1DA"
LABEL, BLUE, MUTED, ORANGE_T, BLUE_T = "A9471A", "2F6FA3", "8FA6BA", "F8E3D5", "DCE8F2"
DARK, DARK_LABEL, QUIET = "16202A", "E98A4B", "5B6573"


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(10), Inches(5.625)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def text(slide, x, y, w, h, runs, size=12, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = rgb(o.get("color", color))
        if i:
            para.space_before = Pt(o.get("gap", 4) if paras else 4)
    return tb


def box(slide, x, y, w, h, fill=CARD, line=GRID, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def header(slide, label, title):
    text(slide, 0.6, 0.42, 8.8, 0.2, label.upper(), size=8, color=LABEL, bold=True, name="Section label")
    text(slide, 0.6, 0.62, 8.8, 0.5, title, size=20, bold=True, name="Title")


def footer(slide, src):
    text(slide, 0.6, 5.08, 8.8, 0.36, src, size=7, color=QUIET, name="Source")


def table(slide, x, y, w, rows, col_w, size=8, row_h=0.22, bold_rows=(), tint=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shape.table
    t.first_row = True
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.01)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            fill = HEAD if i == 0 else (tint.get((i, j)) if tint and (i, j) in tint else CARD)
            c.fill.fore_color.rgb = rgb(fill)
            tf = c.text_frame
            tf.paragraphs[0].text = ""
            run = tf.paragraphs[0].add_run()
            run.text = str(v)
            run.font.name = BODY
            run.font.size = Pt(size)
            run.font.bold = i == 0 or i in bold_rows
            run.font.color.rgb = rgb(INK)
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
    shape.name = "Table"
    return shape


# ---------------------------------------------------------------- data (merged bake-off, 2026-10-08)
RATES = ["0.075 bpp", "0.15 bpp", "0.3 bpp"]
QHAT = {  # Q-hat v0, all 30 images, knapsack allocation under the corpus budget
    "VTM 4:2:0": (-2.515, -1.163, -0.152),
    "MS-ILLM": (-0.451, 0.544, 1.383),
    "mbt2018 (MSE)": (None, -2.082, -0.634),
    "MS-ILLM + SD-Turbo": (-1.609, -1.331, None),
}

# ---------------------------------------------------------------- slide 1: headline + Q-hat chart
s = prs.slides.add_slide(BLANK)
bg(s, BG)
header(s, "6 · Bake-off (08/10)", "MS-ILLM dẫn đầu ở cả 3 mức bitrate")
cd = CategoryChartData()
cd.categories = RATES
for k, v in QHAT.items():
    cd.add_series(k, v)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.25), Inches(5.9), Inches(3.85), cd)
gf.name = "Q-hat chart"
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Q̂ v0 trung bình trên 30 ảnh (cao hơn = người chấm thích hơn)"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.name = Pt(10), True, BODY
tp.runs[0].font.color.rgb = rgb(INK)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size, ch.legend.font.name = Pt(8), BODY
ch.legend.font.color.rgb = rgb(INK)
for i, ser in enumerate(ch.series):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb([MUTED, BLUE, "C9C2B3", LABEL][i])
    ser.data_labels.show_value = True
    ser.data_labels.number_format = "0.0"
    ser.data_labels.number_format_is_linked = False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size, ser.data_labels.font.name = Pt(7), BODY
    ser.data_labels.font.color.rgb = rgb(INK)
plot = ch.plots[0]
plot.gap_width, plot.overlap = 60, -5
va = ch.value_axis
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = rgb(GRID)
va.tick_labels.font.size, va.tick_labels.font.name = Pt(8), BODY
va.tick_labels.font.color.rgb = rgb(QUIET)
va.format.line.fill.background()
ca = ch.category_axis
ca.tick_labels.font.size, ca.tick_labels.font.name = Pt(9), BODY
ca.tick_labels.font.color.rgb = rgb(INK)
ca.format.line.color.rgb = rgb(GRID)
from pptx.enum.chart import XL_TICK_LABEL_POSITION  # noqa: E402
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW

# right column: three callouts
cards = [("+2.1", "Q̂ hơn VTM ở 0.075 bpp", "LPIPS 0.094 so với 0.296; PSNR thấp hơn 0.9 dB"),
         ("1.4", "Q̂ ở 0.3 bpp, cao nhất bảng", "VTM chỉ −0.15 dù PSNR cao hơn 2.2 dB"),
         ("2.0 s", "giải mã 1 ảnh trên T4", "≈ 60 s cho 30 ảnh; mục tiêu ≤ 20 s trên L4")]
for i, (big, what, why) in enumerate(cards):
    y = 1.25 + i * 1.3
    box(s, 6.75, y, 2.65, 1.15, name=f"Callout {i + 1}")
    text(s, 6.9, y + 0.08, 2.4, 0.46, big, size=22, bold=True, color=BLUE if i < 2 else LABEL, font=MONO)
    text(s, 6.9, y + 0.52, 2.4, 0.22, what, size=9, bold=True)
    text(s, 6.9, y + 0.75, 2.4, 0.35, why, size=8, color=QUIET)
footer(s, "Nguồn: gộp session A, B, c, d, e (results/2026-10-0{6,8}_*); phân bổ knapsack theo Q̂ dưới ngân sách cả bộ ảnh. "
          "mbt không đạt 0.075; refiner không đạt 0.3.")
s.notes_slide.notes_text_frame.text = (
    "Mỗi ứng viên được phân bổ bit theo Q̂ dưới cùng ngân sách byte cả bộ 30 ảnh. Chênh 1.0 Q̂ ≈ odds 2.7 lần người chấm chọn. "
    "VTM có PSNR cao hơn nhưng LPIPS/DISTS kém 3–4 lần, khớp với Elo 2025 (VTM 1405 so với Vcoder 1929).")

# ---------------------------------------------------------------- slide 2: full table
s = prs.slides.add_slide(BLANK)
bg(s, BG)
header(s, "6 · Bake-off (08/10)", "Bảng đầy đủ: PSNR cao không có nghĩa là thắng")
rows = [["Mức", "Ứng viên", "bpp", "PSNR (dB)", "MS-SSIM", "LPIPS↓", "DISTS↓", "Q̂", "Giải mã T4 (s)"],
        ["0.075", "VTM 4:2:0", "0.0749", "26.01", "0.921", "0.296", "0.178", "−2.52", "CPU"],
        ["", "MS-ILLM", "0.0749", "25.15", "0.914", "0.094", "0.094", "−0.45", "1.95"],
        ["", "MS-ILLM + SD-Turbo", "0.0749", "22.22", "0.842", "0.197", "0.149", "−1.61", "11.13"],
        ["", "mix (chọn theo ảnh)", "0.0749", "23.62", "0.901", "0.096", "0.093", "−0.41", "2.20"],
        ["0.15", "VTM 4:2:0", "0.1498", "28.29", "0.955", "0.214", "0.135", "−1.16", "CPU"],
        ["", "MS-ILLM", "0.1499", "27.16", "0.949", "0.058", "0.061", "0.54", "1.98"],
        ["", "mbt2018 (MSE)", "0.1499", "27.52", "0.949", "0.261", "0.168", "−2.08", "0.76"],
        ["", "mix (chọn theo ảnh)", "0.1497", "26.72", "0.943", "0.065", "0.060", "0.57", "2.20"],
        ["0.3", "VTM 4:2:0", "0.2998", "31.62", "0.973", "0.150", "0.105", "−0.15", "CPU"],
        ["", "MS-ILLM", "0.2997", "29.41", "0.973", "0.034", "0.036", "1.38", "1.99"],
        ["", "mbt2018 (MSE)", "0.2989", "29.51", "0.970", "0.182", "0.121", "−0.63", "0.76"],
        ["", "mix (chọn theo ảnh)", "0.2995", "29.78", "0.973", "0.035", "0.036", "1.41", "1.96"]]
tint = {(r, c): BLUE_T for r in (2, 6, 10) for c in range(1, 9)}
table(s, 0.6, 1.25, 8.8, rows, [0.55, 1.75, 0.8, 0.9, 0.85, 0.8, 0.8, 0.75, 1.6], size=8, row_h=0.255,
      bold_rows=(2, 6, 10), tint=tint)
footer(s, "30 ảnh validation CLIC 2027; PSNR gộp MSE theo số pixel (cách của CLIC). Dòng tô xanh: MS-ILLM. "
          "VTM giải mã trên CPU, không đo thời gian. Q̂ v0 đoán đúng 81% lựa chọn người chấm CLIC 2024.")

# ---------------------------------------------------------------- slide 3: mode switching
s = prs.slides.add_slide(BLANK)
bg(s, BG)
header(s, "6 · Bake-off (08/10)", "Chọn mode theo ảnh: lợi ít, trừ ảnh màn hình")
cd = CategoryChartData()
cd.categories = RATES
cd.add_series("MS-ILLM cho mọi ảnh", (1.208, 1.866, 2.439))
cd.add_series("mix (MS-ILLM + VTM-SCC + S1)", (1.348, 2.259, 2.651))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.25), Inches(4.6), Inches(3.3), cd)
gf.name = "Screen chart"
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Q̂ trên 4 ảnh màn hình"
r0 = ch.chart_title.text_frame.paragraphs[0].runs[0]
r0.font.size, r0.font.bold, r0.font.name, r0.font.color.rgb = Pt(10), True, BODY, rgb(INK)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size, ch.legend.font.name = Pt(8), BODY
for i, ser in enumerate(ch.series):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb([MUTED, BLUE][i])
    ser.data_labels.show_value = True
    ser.data_labels.number_format, ser.data_labels.number_format_is_linked = "0.00", False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size, ser.data_labels.font.name = Pt(8), BODY
ch.plots[0].gap_width = 70
va = ch.value_axis
va.major_gridlines.format.line.color.rgb = rgb(GRID)
va.tick_labels.font.size, va.tick_labels.font.name = Pt(8), BODY
va.format.line.fill.background()
ch.category_axis.tick_labels.font.size = Pt(9)
ch.category_axis.tick_labels.font.name = BODY
ch.category_axis.format.line.color.rgb = rgb(GRID)

text(s, 5.55, 1.25, 3.85, 0.25, "Ảnh nào được giao cho base nào (30 ảnh)", size=10, bold=True)
table(s, 5.55, 1.55, 3.85, [["Mức", "MS-ILLM", "VTM-SCC", "S1 + residual"],
                              ["0.075", "28", "1", "1"], ["0.15", "27", "2", "1"], ["0.3", "28", "2", "0"]],
      [0.85, 1.0, 1.0, 1.0], size=9, row_h=0.26)
box(s, 5.55, 2.8, 3.85, 1.75, fill=CARD, name="Takeaway")
text(s, 5.7, 2.9, 3.6, 1.6, [
    [("Cả bộ ảnh: ", {"bold": True}), ("mix chỉ hơn MS-ILLM +0.03 đến +0.04 Q̂.", {})],
    [("Ảnh màn hình: ", {"bold": True}), ("ở 0.15 bpp PSNR 29.7 → 32.5 dB, Q̂ 1.87 → 2.26 nhờ VTM-SCC.", {})],
    [("Kết luận: ", {"bold": True}), ("giữ mode SCC cho ảnh màn hình (decoder VVC có sẵn trong devkit), "
                                      "nhưng đây không phải đòn quyết định.", {})]], size=9)
footer(s, "VTM-SCC chỉ chạy trên 5 ảnh màn hình/texture (session c) nên chỉ xuất hiện trong mix. S1 + residual thiếu 2/30 ảnh.")

# ---------------------------------------------------------------- slide 4: sessions + status
s = prs.slides.add_slide(BLANK)
bg(s, BG)
header(s, "6 · Bake-off (08/10)", "Năm session Kaggle, một ứng viên chưa chạy được")
rows = [["Session", "Máy", "Ứng viên", "Kết quả"],
        ["A (06/10)", "T4", "Q̂ v0, MS-ILLM, mbt, S1 + residual", "Thiếu 2 ảnh khi hết giờ (MS-ILLM, mbt bù ở e)"],
        ["B (06/10)", "CPU", "VTM 4:2:0", "Chỉ có byte; metrics mất khi pack (đã sửa)"],
        ["c (08/10)", "CPU", "VTM-SCC, 5 ảnh màn hình/texture × 6 QP", "30/30 điểm"],
        ["d (08/10)", "CPU", "VTM 4:2:0, 30 ảnh × 6 QP", "180/180 điểm, 8 giờ"],
        ["e (08/10)", "T4", "MS-ILLM, mbt, SD-Turbo refiner, CoD-Lite", "CoD-Lite lỗi khi nạp model"]]
table(s, 0.6, 1.25, 8.8, rows, [1.1, 0.7, 3.6, 3.4], size=9, row_h=0.3, tint={(5, 3): ORANGE_T})
cards = [("Dừng", "SD-Turbo refiner", "Kém MS-ILLM ~1.2 Q̂, −3 dB PSNR, 11 s/ảnh, decoder 4.5 GB vượt giới hạn 4 GB.", LABEL),
         ("Chạy lại", "CoD-Lite (diffusion, MIT)", "Lỗi đếm tham số (y_embedder.encoder = None), đã có bản sửa. "
                                                         "~1 giờ T4 với --cands codlite.", BLUE),
         ("Kiểm tra", "License MS-ILLM", "Weights công khai là CC-BY-NC. Đọc luật CLIC; hướng dài hạn là tự train "
                                         "codec kiểu MS-ILLM.", INK)]
for i, (tag, what, why, col) in enumerate(cards):
    x = 0.6 + i * 3.0
    box(s, x, 3.3, 2.8, 1.75, name=f"Card {i + 1}")
    text(s, x + 0.15, 3.42, 2.5, 0.25, tag.upper(), size=8, bold=True, color=col)
    text(s, x + 0.15, 3.68, 2.5, 0.3, what, size=11, bold=True)
    text(s, x + 0.15, 4.05, 2.5, 0.95, why, size=8.5, color=QUIET)
footer(s, "Mọi session dùng commit efba80e (+ thay đổi chưa commit) hoặc 137158d; Q̂ v0 giống nhau ở cả 5 session.")

# ---------------------------------------------------------------- slide 5: summary (dark)
s = prs.slides.add_slide(BLANK)
bg(s, DARK)
text(s, 0.6, 0.42, 8.8, 0.2, "TỔNG KẾT BAKE-OFF", size=8, color=DARK_LABEL, bold=True, name="Section label")
text(s, 0.6, 0.62, 8.8, 0.5, "Base: MS-ILLM. Việc tiếp theo là tốc độ và license", size=20, bold=True, color="FFFFFF",
     name="Title")
cols = [("Đã biết", ["MS-ILLM thắng cả 3 mức theo Q̂ (+1.5 đến +2.1 so với VTM)",
                     "Codec MSE (VTM, mbt) PSNR cao nhưng Q̂ thấp, khớp Elo 2025",
                     "DC-AE + S1 và refiner SD-Turbo đều thua xa",
                     "Mode SCC chỉ có lợi cho ảnh màn hình"]),
        ("Rủi ro", ["Giải mã ≈ 60 s/30 ảnh trên T4, mục tiêu ≤ 20 s trên L4",
                    "Weights MS-ILLM là CC-BY-NC",
                    "Q̂ v0 gần như chưa thấy ảnh diffusion",
                    "CoD-Lite chưa được đo"]),
        ("Tiếp theo", ["Chạy lại CoD-Lite (đã sửa lỗi), ~1 giờ T4",
                       "Đo MS-ILLM trên L4: fp16, nạp model, 30 ảnh",
                       "Tự train / fine-tune codec kiểu MS-ILLM theo Q̂",
                       "Quyết base cho từng mức: mốc 27/10"])]
for i, (head, items) in enumerate(cols):
    x = 0.6 + i * 3.0
    box(s, x, 1.4, 2.8, 3.55, fill="1E2B38", line="2C3A48", name=f"Column {i + 1}")
    text(s, x + 0.18, 1.55, 2.45, 0.3, head, size=13, bold=True, color=DARK_LABEL if i == 1 else "FFFFFF")
    tb = text(s, x + 0.18, 1.95, 2.45, 2.9, [[(t, {})] for t in items], size=9.5, color="D5DCE3")
    for p in tb.text_frame.paragraphs:
        p.space_after = Pt(7)

prs.save(OUT)
print("saved", OUT)
