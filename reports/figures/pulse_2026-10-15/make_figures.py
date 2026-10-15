"""Regenerate report figures from saved measurements; requires matplotlib."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, default=root / 'results/2026-10-13_bakeoff_v0/summary.json')
p.add_argument('--checks', type=Path, default=root / 'results/2026-10-15_pulse_integration/integration_checks.json')
p.add_argument('--allocation', type=Path, default=root / 'results/2026-10-15_pulse_integration/integrated_bakeoff/allocation.csv')
p.add_argument('--out', type=Path, default=Path(__file__).parent)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
baseline = json.loads(a.baseline.read_text(encoding='utf-8'))
checks = json.loads(a.checks.read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
blue, teal, orange, gray = '#285A95', '#008577', '#D87826', '#8793A1'

def save(fig, name):
    fig.savefig(a.out / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(a.out / (name + '.svg'), bbox_inches='tight')
    svg = a.out / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)

rates = [.075, .15, .3]
old = np.array([baseline[f'mix@{r}/all']['qhat'] for r in rates])
new = np.array([checks['quality'][f'mix@{r}/all']['qhat'] for r in rates])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
x = np.arange(3)
for values, offset, label, color in [(old, -.18, 'Mix baseline', blue), (new, .18, 'Mix + PULSE-L q0–q4', teal)]:
    bars = axes[0].bar(x + offset, values, .34, label=label, color=color)
    axes[0].bar_label(bars, fmt='%.4f', fontsize=9, padding=4)
axes[0].set(xticks=x, xticklabels=rates, xlabel='Budget (bpp)', ylabel='Q̂ v0 — cao hơn tốt hơn', ylim=(0, 1.8), title='Chất lượng của mix trên 30 ảnh')
axes[0].legend(loc='upper left', fontsize=9)
bars = axes[1].bar(x, new-old, color=teal, width=.5)
axes[1].bar_label(bars, fmt='+%.4f', padding=5)
axes[1].set(xticks=x, xticklabels=rates, xlabel='Budget (bpp)', ylabel='ΔQ̂ so với baseline', ylim=(0, .095), title='Mức tăng tập trung ở 0.15–0.3 bpp')
fig.supxlabel('Q̂ là metric đại diện; chưa xác nhận bằng chấm mù. Không có error bar vì chưa đo độ bất định.', fontsize=10)
save(fig, 'quality')

with a.allocation.open(encoding='utf-8', newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['cand'] == 'mix' and float(r['rate']) == .3]
counts = Counter(r['point_cand'] for r in rows)
assert len(rows) == 30 and counts == {'pulse': 22, 'msillm': 5, 'vtmscc': 2, 'ditic': 1}
fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
bars = ax.barh(['PULSE-L', 'MS-ILLM', 'VTM-SCC', 'DiT-IC'], [22, 5, 2, 1], color=[teal, blue, orange, gray])
ax.bar_label(bars, labels=[f'{n}/30 ảnh ({n/30:.1%})' for n in [22, 5, 2, 1]], padding=8)
ax.invert_yaxis()
ax.set(xlim=(0, 30), xlabel='Số ảnh được allocator chọn', title='Mix tại 0.3 bpp: PULSE-L được chọn cho 22/30 ảnh')
fig.supxlabel('Tỷ trọng theo số ảnh, không phải tỷ trọng byte. Tổng stream: 3,231,292 byte (0.299730 bpp).', fontsize=10)
save(fig, 'allocation')

fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained', gridspec_kw={'width_ratios': [1.3, 1]})
versions = [checks['offline'][v] for v in ['v1', 'v2']]
totals = np.array([v['total_wall_s'] for v in versions])
parts = [('Giải nén ZIP', 'unpack_s', gray), ('Nạp model', 'load_s', blue),
         ('API decode', 'decode_sum_s', teal), ('Chuyển/ghi PNG', 'png_write_sum_s', orange)]
left = np.zeros(2)
for label, key, color in parts:
    values = np.array([v[key] for v in versions])
    axes[0].barh([0, 1], values, left=left, label=label, color=color)
    left += values
other = totals-left
assert np.all(other >= 0)
axes[0].barh([0, 1], other, left=left, label='Phần còn lại¹', color='#D5DAE0')
for i, total in enumerate(totals):
    axes[0].text(total+.4, i, f'{total:.2f} s', va='center', fontweight='bold')
axes[0].set(yticks=[0, 1], yticklabels=['V1: PNG mặc định', 'V2: PNG level 0'], xlim=(0, 49), xlabel='Giây / 30 ảnh', title='Giảm thời gian bằng cách tối ưu ghi PNG')
axes[0].invert_yaxis()
axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=9)
# Use decimal GB consistently with the user's 30 GB allowance.
vram = np.array([checks['integrated_max_device_used_mib'], versions[1]['max_device_used_mib']]) * 2**20 / 1e9
bars = axes[1].barh([0, 1], vram, color=[blue, teal], height=.55)
axes[1].bar_label(bars, fmt='%.2f GB', padding=5)
axes[1].axvline(30, color='#A83C36', linestyle='--')
axes[1].text(29.5, -.48, 'Hạn mức: 30 GB/GPU', ha='right', color='#A83C36', fontsize=10)
axes[1].set(yticks=[0, 1], yticklabels=['Bake-off + metric', 'Offline V2'], xlim=(0, 32), ylim=(1.6, -.7), xlabel='Tổng VRAM thiết bị (GB thập phân)', title='VRAM thấp hơn hạn mức cho phép')
fig.supxlabel('L40; một lần chạy mỗi bản; 30/30 ảnh khớp pixel. Chưa đo toàn mix hoặc L4.\n¹ Phần còn lại = tổng wall time trừ các mục đo; gồm khởi động, import, checksum và overhead khác.', fontsize=10)
save(fig, 'runtime_memory')
print('Generated 3 figures in PNG and SVG:', a.out)
