"""Reproduce supplementary full30 plots and numerical tables from saved CSV only."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/2026-10-20_metrics_comparison/full30_budget_metrics.csv'
ROWS = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
for row in ROWS:
    for key in ('budget_bpp', 'actual_bpp', 'psnr_pooled', 'lpips_alex', 'dists', 'qhat_v0'):
        row[key] = float(row[key])
    row['bytes'] = int(row['bytes'])
    assert int(row['n']) == 30
    assert row['actual_bpp'] <= row['budget_bpp']
    assert abs(row['actual_bpp'] - 8 * row['bytes'] / 86245376) < 1e-12

NAMES = dict(vtm420='VTM 4:2:0', vtmscc='VTM-SCC', mbt='MBT2018-mean',
             turbo='MS-ILLM + SD-Turbo', msillm='MS-ILLM', codlite='CoD-Lite',
             crdr='CRDR', ditic='DiT-IC', ditics='DiT-IC student', pulse='PULSE-L',
             mix='Mix có PULSE-L', expanded_mix='Mix có PULSE-L + AEIC')
COLORS = ['#8B8F97', '#637084', '#A98C70', '#BE719B', '#2379B5', '#DD8D26',
          '#8A5BB1', '#159C9C', '#86B2A8', '#509543', '#B35647', '#162E4E']
IDS = {m: i + 1 for i, m in enumerate(NAMES)}
COLOR = dict(zip(NAMES, COLORS))
FONT = Path('C:/Windows/Fonts/arial.ttf')
def font(size):
    return ImageFont.truetype(str(FONT), size)

def dominates(a, b):
    return (a['bytes'] <= b['bytes'] and a['qhat_v0'] >= b['qhat_v0']
            and (a['bytes'] < b['bytes'] or a['qhat_v0'] > b['qhat_v0']))

frontier = [r for r in ROWS if not any(dominates(s, r) for s in ROWS)]
unique_frontier = sorted({(r['bytes'], r['qhat_v0']) for r in frontier})
assert len(unique_frontier) == 10
high = [r for r in ROWS if r['budget_bpp'] == .3]
def get(method, budget):
    return next(r for r in ROWS if r['method'] == method and r['budget_bpp'] == budget)
assert dominates(get('mix', .3), get('pulse', .3))
assert dominates(get('msillm', .3), get('crdr', .3))
assert abs(get('mix', .3)['qhat_v0'] - get('pulse', .3)['qhat_v0'] - .0744362174787492) < 1e-12

def chart(filename, title, subtitle, rows, xkey, ykey, xlim, ylim, xlabel, ylabel, xticks, yticks, pareto=False):
    im = Image.new('RGB', (1600, 1120), '#FCFCFA')
    d = ImageDraw.Draw(im)
    d.text((65, 24), title, font=font(32), fill='#172B44')
    d.text((65, 70), subtitle, font=font(21), fill='#586577')
    left, top, right, bottom = 145, 135, 1515, 790
    def xy(x, y):
        return (left + (x-xlim[0])/(xlim[1]-xlim[0])*(right-left),
                bottom - (y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top))
    for x in xticks:
        px, _ = xy(x, ylim[0])
        d.line((px, top, px, bottom), fill='#DFE4E8')
        d.text((px, bottom+14), f'{x:g}', font=font(21), fill='#586577', anchor='mt')
    for y in yticks:
        _, py = xy(xlim[0], y)
        d.line((left, py, right, py), fill='#DFE4E8')
        d.text((left-15, py), f'{y:g}', font=font(21), fill='#586577', anchor='rm')
    d.text((left, 105), ylabel, font=font(23), fill='#172B44')
    d.text(((left+right)/2, 850), xlabel, font=font(23), fill='#172B44', anchor='mt')
    if pareto:
        pts = [xy(8*b/86245376, q) for b, q in unique_frontier]
        d.line(pts, fill='#172B44', width=3)
    seen = set()
    for r in sorted(rows, key=lambda r: r in frontier) if pareto else rows:
        key = (r[xkey], r[ykey])
        if key in seen:
            continue
        seen.add(key)
        px, py = xy(*key)
        c = COLOR[r['method']]
        radius = 10
        d.ellipse((px-radius, py-radius, px+radius, py+radius), fill=c, outline='white', width=2)
        if pareto and r in frontier:
            d.ellipse((px-15, py-15, px+15, py+15), outline='#172B44', width=2)
        if not pareto:
            # Number labels use a separate legend; close points remain identifiable by color.
            offsets = {'msillm': (-24, 23), 'pulse': (22, -15), 'mix': (28, -25),
                       'crdr': (25, 16), 'ditics': (28, 17), 'ditic': (-25, -20)}
            if xkey == 'lpips_alex':
                offsets.update(msillm=(-25, 26), pulse=(-35, -16), mix=(43, 16), crdr=(28, -20))
            dx, dy = offsets.get(r['method'], (20, -18))
            label = '11/12' if r['method'] == 'mix' else str(IDS[r['method']])
            d.text((px+dx, py+dy), label, font=font(20), fill=c, anchor='mm')
    for i, (m, name) in enumerate(NAMES.items()):
        xx, yy = 65 + (i % 3)*510, 910 + (i//3)*39
        d.ellipse((xx, yy+4, xx+16, yy+20), fill=COLOR[m])
        d.text((xx+26, yy), f'{IDS[m]}. {name}', font=font(20), fill='#172B44')
    d.text((65, 1080), 'Nguồn: full30_budget_metrics.csv | 30 ảnh | Kết quả mix trùng nhau chỉ vẽ một lần', font=font(19), fill='#586577')
    im.save(OUT / filename)

chart('08_rate_qhat_pareto.png', 'Biên Pareto giữa bitrate và điểm mô hình preference',
      'Vòng viền: điểm không bị chi phối. Đường nối chỉ hướng dẫn mắt, không xác lập bitrate trung gian.',
      ROWS, 'actual_bpp', 'qhat_v0', (.065, .31), (-2.7, 1.7), 'Bitrate thực (bpp; thấp hơn tốt hơn)',
      'Q̂ v0 (cao hơn tốt hơn)', [.075, .1, .15, .2, .25, .3], [-2.5, -2, -1, 0, 1, 1.5], True)
chart('09_psnr_qhat.png', 'Độ trung thực pixel và điểm mô hình preference',
      'Ngân sách 0.3 bpp; DiT-IC, student và Turbo chỉ dùng lần lượt 0.096, 0.098 và 0.146 bpp.',
      high, 'psnr_pooled', 'qhat_v0', (21, 33), (-1.6, 1.8), 'PSNR gộp (dB; cao hơn tốt hơn)',
      'Q̂ v0 (cao hơn tốt hơn)', [22, 24, 26, 28, 30, 32], [-1.5, -1, -.5, 0, .5, 1, 1.5])
chart('10_lpips_dists.png', 'Khoảng cách cảm nhận: LPIPS-Alex và DISTS',
      'Ngân sách 0.3 bpp; bitrate thực khác nhau. Góc dưới bên trái tốt hơn theo cả hai khoảng cách.',
      high, 'lpips_alex', 'dists', (.02, .19), (.02, .155), 'LPIPS-Alex (thấp hơn tốt hơn)',
      'DISTS (thấp hơn tốt hơn)', [.025, .05, .075, .1, .125, .15, .175], [.025, .05, .075, .1, .125, .15])

# A table plot keeps all three budgets readable without hiding missing observations.
im = Image.new('RGB', (1600, 1010), '#FCFCFA')
d = ImageDraw.Draw(im)
d.text((65, 25), 'Mức tăng Q̂ so với VTM-SCC ở từng ngân sách', font=font(32), fill='#172B44')
d.text((65, 75), 'Chênh lệch điểm tuyệt đối; không phải phần trăm tăng hay xác suất ưa thích của người xem.', font=font(23), fill='#586577')
for j, b in enumerate((.075, .15, .3)):
    d.text((770+j*280, 145), f'Ngân sách {b:g} bpp', font=font(24), fill='#172B44', anchor='mm')
for i, (m, name) in enumerate(NAMES.items()):
    y = 205+i*58
    d.text((65, y), name, font=font(25), fill=COLOR[m])
    for j, b in enumerate((.075, .15, .3)):
        candidates = [r for r in ROWS if r['method'] == m and r['budget_bpp'] == b]
        if candidates:
            delta = candidates[0]['qhat_v0'] - get('vtmscc', b)['qhat_v0']
            value, color = f'{delta:+.4f}', '#21624C' if delta > 0 else '#8B4950' if delta < 0 else '#586577'
        else:
            value, color = 'Không vừa', '#586577'
        d.text((770+j*280, y), value, font=font(26), fill=color, anchor='mt')
d.text((65, 930), 'Không vừa: chưa có điểm khả thi đủ 30 ảnh. Loại DC-AE vì chỉ có metric cho 28/30 ảnh.', font=font(22), fill='#586577')
d.text((65, 970), 'Nguồn: full30_budget_metrics.csv | Tính trước khi làm tròn | Bitrate thực khác nhau giữa các phương pháp', font=font(21), fill='#586577')
im.save(OUT / '11_qhat_gain.png')

summary = {'source': str(SOURCE.relative_to(ROOT)), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'frontier': [{k: r[k] for k in ('method', 'budget_bpp', 'bytes', 'actual_bpp', 'qhat_v0')} for r in frontier]}
(OUT / 'supplement_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('FRONTIER:')
for b, q in unique_frontier:
    names = sorted({NAMES[r['method']] for r in frontier if (r['bytes'], r['qhat_v0']) == (b, q)})
    print(f"| {' / '.join(names)} | {b:,} | {8*b/86245376:.8f} | {q:.4f} |")
print('GAIN:')
for m, name in NAMES.items():
    vals = []
    for b in (.075, .15, .3):
        candidates = [r for r in ROWS if r['method'] == m and r['budget_bpp'] == b]
        vals.append(f"{candidates[0]['qhat_v0'] - get('vtmscc', b)['qhat_v0']:+.4f}" if candidates else 'N/A')
    print('| ' + ' | '.join([name] + vals) + ' |')
for m in ('msillm', 'pulse', 'mix'):
    r, base = get(m, .3), get('vtmscc', .3)
    print(m, 'PSNR delta', r['psnr_pooled']-base['psnr_pooled'],
          'LPIPS reduction %', 100*(1-r['lpips_alex']/base['lpips_alex']),
          'DISTS reduction %', 100*(1-r['dists']/base['dists']))
