"""Generate the .drawio diagrams of the CLIC 2027 proposal: architecture and encode/decode pipeline."""
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
import sys

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

GREY, BLUE, BLUE_FILL, INK, QUIET = "#9E9E9E", "#1F6FEB", "#DAE8FC", "#1F1F1F", "#666666"
FONT = "fontFamily=Helvetica;fontColor=%s;" % INK
BOX = ("rounded=1;arcSize=8;absoluteArcSize=1;whiteSpace=wrap;html=1;align=left;verticalAlign=top;"
       "spacingLeft=8;spacingTop=2;fontSize=13;fillColor=#FFFFFF;strokeColor=%s;strokeWidth=1.25;" % GREY + FONT)
ACCENT = BOX.replace("fillColor=#FFFFFF;strokeColor=%s;strokeWidth=1.25;" % GREY,
                     "fillColor=%s;strokeColor=%s;strokeWidth=2;" % (BLUE_FILL, BLUE))
CENTER = BOX.replace("align=left;verticalAlign=top;", "align=center;verticalAlign=middle;")
TEXT = "text;html=1;whiteSpace=nowrap;align=left;verticalAlign=middle;fontSize=13;strokeColor=none;fillColor=none;" + FONT
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=6;"
        "strokeColor=%s;strokeWidth=1.25;fontSize=11;fontColor=%s;" % (GREY, QUIET))


def body(name, *lines, bold_last=False):
    out = "<b>%s</b>" % escape(name)
    for i, ln in enumerate(lines):
        last = bold_last and i == len(lines) - 1
        txt = escape(ln)
        out += ("<br><b>%s</b>" % txt) if last else ('<br><font style="font-size:11.5px" color="%s">%s</font>' % (QUIET, txt))
    return out


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 2

    def _id(self):
        self.n += 1
        return "c%d" % self.n

    def vertex(self, value, x, y, w, h, style):
        cid = self._id()
        self.cells.append('<mxCell id="%s" value=%s style=%s vertex="1" parent="1"><mxGeometry x="%g" y="%g" width="%g" height="%g" as="geometry"/></mxCell>'
                          % (cid, quoteattr(value), quoteattr(style), x, y, w, h))
        return cid

    def text(self, value, x, y, w=600, h=20, size=13, bold=False, color=INK, align="left"):
        style = TEXT.replace("fontSize=13;", "fontSize=%g;" % size).replace("fontColor=%s;" % INK, "fontColor=%s;" % color)
        style = style.replace("align=left;", "align=%s;" % align)
        if bold:
            style += "fontStyle=1;"
        return self.vertex(escape(value), x, y, w, h, style)

    def edge(self, src, tgt, exit=None, entry=None, label="", points=()):
        cid = self._id()
        style = EDGE
        if exit:
            style += "exitX=%g;exitY=%g;exitDx=0;exitDy=0;" % exit
        if entry:
            style += "entryX=%g;entryY=%g;entryDx=0;entryDy=0;" % entry
        pts = "".join('<mxPoint x="%g" y="%g"/>' % p for p in points)
        geo = '<mxGeometry relative="1" as="geometry">%s</mxGeometry>' % (('<Array as="points">%s</Array>' % pts) if pts else "")
        self.cells.append('<mxCell id="%s" value=%s style=%s edge="1" parent="1" source="%s" target="%s">%s</mxCell>'
                          % (cid, quoteattr(escape(label)), quoteattr(style), src, tgt, geo))
        return cid

    def save(self, title):
        xml = ('<mxfile host="drawio"><diagram name=%s id="%s"><mxGraphModel dx="%d" dy="%d" grid="1" gridSize="8" guides="1" '
               'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
               '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>\n'
               % (quoteattr(title), self.name, self.w, self.h, self.w, self.h, "".join(self.cells)))
        (OUT / ("%s.drawio" % self.name)).write_text(xml, encoding="utf-8")


# 1. Architecture ---------------------------------------------------------------
d = Diagram("1_kien_truc", 760, 664)
d.vertex("", 24, 204, 712, 436, "rounded=1;arcSize=8;absoluteArcSize=1;html=1;fillColor=none;dashed=1;dashPattern=4 4;strokeColor=#CFCFCF;strokeWidth=1.25;")
d.text("Decoder cố định trên server; encoder chọn mode và chia bit cho từng ảnh", 24, 18, 712, 24, size=15, bold=True)
enc = d.vertex(body("Encoder (chạy ngoài server, không bị đóng băng)",
                    "Chọn mode, chia bit giữa các ảnh và giữa base/residual, tìm seed, tối ưu latent theo Q̂"), 24, 48, 712, 56, BOX)
bs = d.vertex(body("Bitstream bs.zip, tổng byte ≤ ngân sách cả tập",
                   "Header mỗi ảnh: H, W, mode, rate_idx, seed_idx, cờ; sau đó là các stream z, y và stream phụ"), 24, 128, 712, 56, BOX)
d.text("Decoder trên server, đóng băng trước 01/03", 40, 216, 400, 20, bold=True)
d.text("Mode lai: ảnh tự nhiên và hỗn hợp", 40, 242, 300, 18, size=11.5, color=QUIET)
d.text("Mode SCC: ảnh màn hình", 488, 242, 200, 18, size=11.5, color=QUIET)
ent = d.vertex(body("Entropy decode số nguyên (CPU)", "Hyperprior int8 + rANS, khớp từng bit trên mọi thiết bị"), 40, 264, 424, 56, BOX)
br = d.vertex(body("Bridge một bước (SANA-0.6B)", "ŷ có dither → latent sắc nét; seed nhiễu đọc từ header"), 40, 344, 424, 56, BOX)
dc = d.vertex(body("DC-AE decoder", "Latent f32c32 → ảnh base x̂ (hiện mất 4.6 s/ảnh trên T4)"), 40, 424, 424, 56, BOX)
res = d.vertex("<b>Residual có điều kiện theo x̂</b><br><font style=\"font-size:11.5px\">Giải mã phần dư x − x̂; gánh độ trung thực ở 0.15–0.3 bpp</font>",
               40, 504, 424, 56, ACCENT)
vvc = d.vertex(body("VVC-SCC decoder", "VTM 23.8, có sẵn trong devkit", "IBC, palette, 4:4:4"), 488, 264, 232, 72, BOX)
out = d.vertex("<b>Ảnh PNG đầu ra cho từng ảnh</b>", 40, 584, 680, 40, CENTER)
d.edge(enc, bs, (0.5, 1), (0.5, 0))
d.edge(bs, ent, ((440 - 24) / 712, 1), ((440 - 40) / 424, 0))
d.edge(bs, vvc, ((700 - 24) / 712, 1), ((700 - 488) / 232, 0))
d.edge(ent, br, (0.5, 1), (0.5, 0))
d.edge(br, dc, (0.5, 1), (0.5, 0))
d.edge(dc, res, (0.5, 1), (0.5, 0))
d.edge(res, out, (0.5, 1), ((252 - 40) / 680, 0))
d.edge(vvc, out, (0.5, 1), ((604 - 40) / 680, 0))
d.save("Kiến trúc đề xuất")

# 2. Encode-decode pipeline -----------------------------------------------------
d = Diagram("2_pipeline_ma_hoa_giai_ma", 760, 384)
d.text("Chỉ phần quyết định CDF phải khớp từng bit; DiT và DC-AE chạy float", 24, 16, 712, 24, size=15, bold=True)
d.text("Mode lai. Ô tô màu: tính bằng số nguyên, encoder và server phải ra cùng một kết quả", 24, 40, 712, 18, size=11.5, color=QUIET)
d.text("ENCODER (máy của em)", 24, 72, 300, 18, size=11.5, bold=True, color=QUIET)
d.text("DECODER (server L4)", 24, 224, 300, 18, size=11.5, bold=True, color=QUIET)
cols = [24, 171, 318, 465, 612]
BW, BH, r1, r2 = 124, 72, 92, 244
row1 = [("Ảnh x", ["RGB, H × W"], False), ("DC-AE enc", ["latent 32 kênh", "nén 32× / chiều"], False),
        ("g_a, × gain[r]", ["rate r: 8 mức", "bước Δ = 1/gain"], False), ("Dither + round", ["q = round(y + u)", "u chung, từ seed"], True),
        ("rANS encode", ["CDF từ h_s", "mạng int8"], True)]
row2 = [("Residual", ["điều kiện x̂", "cộng phần dư"], False), ("DC-AE decoder", ["latent → ảnh x̂"], False),
        ("Bridge 1 bước", ["từ t₀ = τ(Δ) đến 1", "SANA-0.6B"], False), ("Bỏ dither, g_s", ["ŷ = q − u", "→ z̄ ≈ E[z | bits]"], False),
        ("rANS decode", ["h_s int8 → CDF", "trên CPU"], True)]
ids1 = [d.vertex(body(n, *ls), x, r1, BW, BH, ACCENT if a else BOX) for x, (n, ls, a) in zip(cols, row1)]
ids2 = [d.vertex(body(n, *ls), x, r2, BW, BH, ACCENT if a else BOX) for x, (n, ls, a) in zip(cols, row2)]
for a, b in zip(ids1, ids1[1:]):
    d.edge(a, b, (1, 0.5), (0, 0.5))
d.edge(ids1[-1], ids2[-1], (0.5, 1), (0.5, 0), label="bitstream + seed dither")
for a, b in zip(ids2[::-1], ids2[::-1][1:]):
    d.edge(a, b, (0, 0.5), (1, 0.5))
outp = d.text("Ảnh tái tạo (PNG)", 24, 340, 124, 24, bold=True, align="center")
d.edge(ids2[0], outp, (0.5, 1), (0.5, 0))
d.save("Pipeline mã hóa – giải mã")


print("ok", sorted(p.name for p in OUT.glob("*.drawio")))
