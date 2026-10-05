"""Generate 5_codec_c1_chi_tiet.drawio: base codec C1 (DC-AE + S1 latent codec), layer by layer.

Numbers come from src/ratflow/codec/latent_codec.py, src/ratflow/entropy/{intnet,models,tables}.py,
results/2026-10-03/s1/log.json (N=192, M=128, Z=96, y_stride=2, 8 rates) and the SANA DC-AE f32c32 config.
Usage: python make_codec_c1.py [out_dir]
"""
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
import sys

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
OUT.mkdir(parents=True, exist_ok=True)

# (fill, stroke, title colour, line colour)
GRAY = ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#5F5E5A")
TEAL = ("#E1F5EE", "#0F6E56", "#04342C", "#0F6E56")
PURPLE = ("#EEEDFE", "#534AB7", "#26215C", "#534AB7")
INK, QUIET, EDGE_C = "#1F1F1F", "#5F5E5A", "#888780"
FONT = "fontFamily=Helvetica;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=5;"
        "strokeColor=%s;strokeWidth=1;" % EDGE_C)

cells, n = [], [1]


def _id():
    n[0] += 1
    return "c%d" % n[0]


def box(x, y, w, h, ramp, title=None, lines=(), small_title=False):
    fill, stroke, tc, lc = ramp
    parts = []
    if title:
        parts.append('<font style="font-size:%dpx" color="%s"><b>%s</b></font>' % (12 if small_title else 14, tc, title if title.startswith("<") or "<sub>" in title else escape(title)))
    parts += ['<font style="font-size:12px" color="%s">%s</font>' % (lc, escape(s)) for s in lines]
    style = ("rounded=1;arcSize=6;absoluteArcSize=1;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
             "fillColor=%s;strokeColor=%s;strokeWidth=1;spacing=2;" % (fill, stroke) + FONT)
    cid = _id()
    cells.append('<mxCell id="%s" value=%s style=%s vertex="1" parent="1"><mxGeometry x="%g" y="%g" width="%g" height="%g" as="geometry"/></mxCell>'
                 % (cid, quoteattr("<br>".join(parts)), quoteattr(style), x, y, w, h))
    return cid


def text(s, x, y, w=120, h=18, size=12, bold=False, align="left", color=QUIET):
    style = ("text;html=1;whiteSpace=nowrap;align=%s;verticalAlign=middle;strokeColor=none;fillColor=none;"
             "fontSize=%d;fontColor=%s;%s" % (align, size, color, FONT)) + ("fontStyle=1;" if bold else "")
    cid = _id()
    cells.append('<mxCell id="%s" value=%s style=%s vertex="1" parent="1"><mxGeometry x="%g" y="%g" width="%g" height="%g" as="geometry"/></mxCell>'
                 % (cid, quoteattr(escape(s)), quoteattr(style), x, y, w, h))
    return cid


def edge(src, tgt, exit, entry, points=()):
    style = EDGE + "exitX=%g;exitY=%g;exitDx=0;exitDy=0;entryX=%g;entryY=%g;entryDx=0;entryDy=0;" % (*exit, *entry)
    pts = "".join('<mxPoint x="%g" y="%g"/>' % p for p in points)
    geo = '<mxGeometry relative="1" as="geometry">%s</mxGeometry>' % (('<Array as="points">%s</Array>' % pts) if pts else "")
    cells.append('<mxCell id="%s" value="" style=%s edge="1" parent="1" source="%s" target="%s">%s</mxCell>'
                 % (_id(), quoteattr(style), src, tgt, geo))


DOWN, UP = ((0.5, 1), (0.5, 0)), ((0.5, 0), (0.5, 1))
RIGHT = ((1, 0.5), (0, 0.5))

# Column A: x -> E -> g_a -> x gain[r] -> Q
x_in = text("x: 3 × H", 41, 38, 120, 20, size=14, bold=True, align="center", color=INK)
E = box(30, 68, 142, 80, GRAY, "DC-AE encoder E", ["đóng băng, ↓32", "ResBlock×2 · 3 tầng", "EffViT×3 · 3 tầng"])
ga = box(30, 176, 142, 122, TEAL, "g<sub>a</sub>", ["Conv 3×3, 192", "ResBlock ×2", "Conv 3×3, 192, ↓2", "ResBlock ×2", "Conv 3×3, 128"])
gain = box(30, 324, 142, 24, TEAL, None, ["× gain[r], 8 × 128"])
Q = box(30, 430, 142, 24, PURPLE, None, ["Q: làm tròn"])
text("ℓ: 32 × H/32", 106, 152, 90)
text("y: 128 × H/64", 106, 300, 90)
text("yᵣ", 106, 382, 30)
edge(x_in, E, *DOWN)
edge(E, ga, *DOWN)
edge(ga, gain, *DOWN)
edge(gain, Q, *DOWN)

# Bottom row: AE -> bits y -> AD -> ÷ gain[r]
AEy = box(226, 430, 60, 24, PURPLE, None, ["AE"])
ADy = box(394, 430, 60, 24, PURPLE, None, ["AD"])
igain = box(508, 430, 142, 24, TEAL, None, ["÷ gain[r]"])
text("q", 190, 420, 16, align="center")
text("bits y", 310, 420, 60, align="center")
text("ŷᵣ", 468, 420, 24, align="center")
edge(Q, AEy, *RIGHT)
edge(AEy, ADy, *RIGHT)
edge(ADy, igain, *RIGHT)

# Column D: ÷ gain -> g_s -> D -> x̂ (upwards)
gs = box(508, 176, 142, 138, TEAL, "g<sub>s</sub>", ["Conv 3×3, 32", "ResBlock ×2", "PixelShuffle ↑2", "Conv 3×3, 768", "ResBlock ×2", "Conv 3×3, 192"])
D = box(508, 68, 142, 80, GRAY, "DC-AE decoder D", ["đóng băng, ↑32", "EffViT×3 · 3 tầng", "ResBlock×3 · 3 tầng"])
x_out = text("x̂: 3 × H", 519, 38, 120, 20, size=14, bold=True, align="center", color=INK)
text("ŷ: 128 × H/64", 584, 370, 90)
text("ℓ̂: 32 × H/32", 584, 154, 90)
edge(igain, gs, *UP)
edge(gs, D, *UP)
edge(D, x_out, *UP)

# Column B: h_a (upwards) -> Q -> AE z
ha = box(192, 230, 128, 120, TEAL, "h<sub>a</sub>", ["Conv 5×5, 96, ↓2", "LeakyReLU", "Conv 5×5, 192, ↓2", "LeakyReLU", "Conv 3×3, 192"])
Qz = box(192, 180, 128, 24, PURPLE, None, ["Q: làm tròn"])
AEz = box(226, 60, 60, 24, PURPLE, None, ["AE"])
ADz = box(394, 60, 60, 24, PURPLE, None, ["AD"])
prior = box(278, 2, 124, 40, PURPLE, None, ["Prior phân rã", "96 kênh, 65 bin"])
text("z: 96 × H/256", 261, 208, 90)
text("ẑ", 236, 132, 16, align="center")
text("bits z", 310, 72, 60, align="center")
text("ẑ", 428, 86, 16, align="center")
edge(gain, ha, (0.5, 1), (0.5, 1), points=[(101, 364), (256, 364)])
edge(ha, Qz, *UP)
edge(Qz, AEz, *UP)
edge(AEz, ADz, *RIGHT)
edge(prior, AEz, (0.05, 1), (0.95, 0))
edge(prior, ADz, (0.95, 1), (0.05, 0))

# Column C: h_s (integer, downwards) -> entropy parameters -> AE/AD for y
hs = box(360, 106, 128, 138, PURPLE, "h<sub>s</sub> · số nguyên", ["↑2 nearest", "QConv 5×5, 192", "↑2 nearest", "QConv 5×5, 192", "QConv 3×3, 256", "bias riêng theo r"])
head = box(360, 268, 128, 60, PURPLE, "Tham số entropy", ["σ: 64 scale log", "μ: bước 1/16"])
text("μ, σ", 320, 388, 40, align="center")
edge(ADz, hs, *DOWN)
edge(hs, head, *DOWN)
edge(head, ADy, *DOWN)
edge(head, AEy, (0.5, 1), (0.5, 0), points=[(424, 408), (256, 408)])

# Legend and notes
for i, (ramp, label) in enumerate([(GRAY, "DC-AE, đóng băng"), (TEAL, "S1 học được, float"),
                                   (PURPLE, "số nguyên, khớp từng bit trên mọi máy")]):
    x = [30, 180, 330][i]
    box(x, 476, 14, 14, ramp)
    text(label, x + 20, 474, 260)
for i, s in enumerate(["ResBlock: Conv 3×3 → GELU → Conv 3×3, cộng skip",
                       "QConv: trọng số int8, kích hoạt 16 bit, ReLU, tính bằng float64 trên CPU",
                       "Kích thước ghi là kênh × H/s (W/s tương tự); ảnh 2048 × 1360 cho y cỡ 128 × 32 × 21"]):
    text(s, 30, 504 + 18 * i, 620)

W, H = 680, 572
xml = ('<mxfile host="drawio"><diagram name=%s id="5_codec_c1_chi_tiet"><mxGraphModel dx="%d" dy="%d" grid="1" gridSize="8" '
       'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="%d" pageHeight="%d" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>%s</root></mxGraphModel></diagram></mxfile>\n'
       % (quoteattr("Codec C1 chi tiết"), W, H, W, H, "".join(cells)))
path = OUT / "5_codec_c1_chi_tiet.drawio"
path.write_text(xml, encoding="utf-8")
print("ok", path, len(cells), "cells")
