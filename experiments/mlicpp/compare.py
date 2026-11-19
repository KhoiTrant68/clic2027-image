"""Compare full30 MLIC++ points with the frozen PULSE mix using the existing allocator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
from bakeoff import allocate_one, qhat_score
from clic27.eval import metrics as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.out / 'status.json').read_text())
    if status.get('state') != 'complete' or not status.get('fresh_pixel_parity') or not status.get('metrics'):
        raise ValueError('Comparison requires completed metrics and fresh-process parity')
    manifest = json.loads((a.out / 'manifest.json').read_text())
    names = sorted(r['name'] for r in manifest['images'])
    if len(names) != 30:
        raise ValueError('Budget comparison requires all 30 images')
    pixels = {r['name']: r['height'] * r['width'] for r in manifest['images']}
    if sum(pixels.values()) != 86245376:
        raise ValueError('Wrong full30 pixel count')
    qhat_path = ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json'
    if hashlib.sha256(qhat_path.read_bytes()).hexdigest() != manifest['qhat_sha256']:
        raise ValueError('Qhat changed since evaluation')
    qhat = json.loads(qhat_path.read_text())
    rows = M.read_csv(a.out / 'metrics.csv')
    qualities = manifest['qualities']
    if len(rows) != len(names) * len(qualities) or {(r['name'], int(r['quality'])) for r in rows} != {(n, q) for n in names for q in qualities}:
        raise ValueError('Incomplete/duplicate MLIC++ metric grid')
    base = ROOT / 'results/2026-10-15_pulse_integration/integrated_bakeoff'
    baseline = [r for r in M.read_csv(base / 'allocation.csv') if r['cand'] == 'mix']
    baseline_metrics = {(r['cand'], r['key'], r['name']): r for r in M.read_csv(base / 'metrics.csv')}
    mlic = {(r['name'], 'mlicpp/q'+str(r['quality'])): r for r in rows}
    baseline_pool, pool_metrics = {n: [] for n in names}, {}
    for path in sorted((base / 'points').glob('*.csv')):
        for point in M.read_csv(path):
            key = point['cand'], point['key'], point['name']
            if point['name'] not in baseline_pool or key not in baseline_metrics:
                continue
            metric = baseline_metrics[key]
            score = qhat_score(qhat, metric)
            if not math.isfinite(score):
                continue
            tag = 'reference/' + point['cand'] + '/' + point['key']
            baseline_pool[point['name']].append((int(point['bytes']), score, tag))
            pool_metrics[point['name'], tag] = metric
    if any(not options for options in baseline_pool.values()):
        raise ValueError('Reference point pool is incomplete')
    results, selections = [], []

    def stats(method, selected, catalog, budget=None):
        met = [catalog[n, option[2]] for n, option in selected.items()]
        total = sum(option[0] for option in selected.values())
        if budget is not None and total > math.floor(budget * sum(pixels.values()) / 8):
            raise ValueError('Allocation exceeds actual byte budget')
        result = dict(method=method, budget_bpp=budget, n=len(met), bytes=total,
                      actual_bpp=8*total/sum(pixels.values()),
                      psnr_pooled=M.pooled_psnr([float(r['mse']) for r in met], [pixels[n] for n in selected]),
                      qhat_v0=sum(option[1] for option in selected.values())/len(selected),
                      mlic_images=sum(option[2].startswith('mlicpp/') for option in selected.values()))
        for key in ['msssim', 'lpips_alex', 'lpips_vgg', 'dists', 'clipiqa']:
            result[key] = sum(float(r[key]) for r in met)/len(met)
        return result

    for q in qualities:
        chosen = {n: (int(mlic[n, f'mlicpp/q{q}']['bytes']), float(mlic[n, f'mlicpp/q{q}']['qhat_v0']), f'mlicpp/q{q}') for n in names}
        results.append(stats(f'mlicpp_q{q}', chosen, mlic))
    rate_status = []
    for rate in (.075, .15, .3):
        budget = math.floor(rate * sum(pixels.values())/8)
        source = [r for r in baseline if float(r['rate']) == rate]
        if len(source) != 30 or {r['name'] for r in source} != set(names):
            raise ValueError('Baseline does not cover the same 30 images')
        catalog = dict(mlic) | pool_metrics
        fixed = {}
        for row in source:
            key = (row['point_cand'], row['key'], row['name'])
            metric = baseline_metrics[key]
            if not math.isclose(qhat_score(qhat, metric), float(row['score']), abs_tol=1e-6):
                raise ValueError('Baseline scores use a different Qhat/protocol')
            catalog[row['name'], 'baseline'] = metric
            fixed[row['name']] = int(row['bytes']), float(row['score']), 'baseline'
        base_stats = stats('frozen_pulse_mix', fixed, catalog, rate)
        results.append(base_stats)
        options = {n: [(int(mlic[n, f'mlicpp/q{q}']['bytes']), float(mlic[n, f'mlicpp/q{q}']['qhat_v0']), f'mlicpp/q{q}')
                       for q in qualities] for n in names}
        selected = allocate_one(options, budget)
        rate_status.append(dict(budget_bpp=rate, mlic_only_feasible=selected is not None))
        if selected is not None:
            results.append(stats('mlicpp_allocated', selected, catalog, rate))
        reference = allocate_one(baseline_pool, budget)
        if reference is None or sum(v[1] for v in reference.values()) < sum(v[1] for v in fixed.values()):
            reference = fixed
        reference_stats = stats('reoptimized_reference_pool', reference, catalog, rate)
        results.append(reference_stats)
        expanded = allocate_one({n: options[n] + baseline_pool[n] for n in names}, budget)
        # Preserve the feasible frozen allocation if rounded DP loses score.
        if expanded is None or sum(v[1] for v in expanded.values()) < sum(v[1] for v in reference.values()):
            expanded = reference
        result = stats('pulse_mix_plus_mlicpp', expanded, catalog, rate)
        result['delta_qhat_vs_frozen'] = result['qhat_v0'] - base_stats['qhat_v0']
        result['delta_qhat_vs_reoptimized'] = result['qhat_v0'] - reference_stats['qhat_v0']
        results.append(result)
        selections.extend(dict(budget_bpp=rate, name=n, bytes=v[0], score=v[1], selected=v[2]) for n, v in expanded.items())
    M.write_csv(results, a.out / 'comparison.csv')
    M.write_csv(selections, a.out / 'expanded_allocation.csv')
    report = dict(results=results, feasibility=rate_status,
                  scope='same-image retrospective comparison; baseline byte metadata is historical and has not been re-audited',
                  baseline=str(base.relative_to(ROOT)), qhat_sha256=manifest['qhat_sha256'])
    (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# Thử nghiệm MLIC++ trên 30 ảnh', '',
             'Checkpoint MSE chính thức, kiến trúc legacy theo hướng dẫn tác giả; không huấn luyện lại.', '',
             '| Phương án | Trần bpp | bpp thực | PSNR gộp | LPIPS-Alex ↓ | DISTS ↓ | Q̂ ↑ | Ảnh dùng MLIC++ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        cap = '-' if row['budget_bpp'] is None else str(row['budget_bpp'])
        lines.append(f"| {row['method']} | {cap} | {row['actual_bpp']:.5f} | {row['psnr_pooled']:.2f} | {row['lpips_alex']:.4f} | {row['dists']:.4f} | {row['qhat_v0']:.4f} | {row['mlic_images']} |")
    lines += ['', '## Độ phủ bitrate', '']
    for row in rate_status:
        lines.append(f"- {row['budget_bpp']} bpp: " + ('có allocation MLIC++ đủ 30 ảnh.' if row['mlic_only_feasible'] else 'các checkpoint MLIC++ đã thử chưa đủ thấp để nén toàn tập trong ngân sách.'))
    lines += ['', '## Phạm vi kết luận', '',
              '- Stream MLIC++ tính cả header thực 31 byte/ảnh, không dùng entropy ước lượng.',
              '- Đã kiểm hash pixel giữa giải mã trong tiến trình encode và một tiến trình decoder mới trên cùng GPU; phần cứng cụ thể được ghi trong qN-decode.json. Chưa kiểm L4 hoặc khác máy.',
              '- Baseline là mix PULSE-L lịch sử. Metadata của các nhánh cũ chưa được kiểm lại toàn bộ; đây chưa phải so sánh hai gói nộp cuối.',
              '- Mix mở rộng được tối ưu lại trên toàn bộ point pool baseline và MLIC++; không chỉ thay thế các điểm đã chọn trước đó. Hàng reoptimized_reference_pool là đối chứng cùng bộ phân bổ, chưa thêm MLIC++.',
              '- Dùng cùng module metric, tile và Q̂ v0 của repo. Môi trường PyTorch/CompressAI khác lần benchmark cũ; chênh lệch nhỏ cần được kiểm chứng lại.',
              '- Không có chấm mù trong thử nghiệm này; Q̂ là mô hình sàng lọc.',
              '- Thời gian từng ảnh trong q*-decode.json là decode API, process_times.json gồm nạp mô hình và ghi PNG. Không suy ra tốc độ toàn mix bằng cách cộng các API riêng.', '']
    (a.out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
