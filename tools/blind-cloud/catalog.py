"""Build the organizer's results catalog from measured experiment artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    'actual_bpp': ('actual_bpp', 'actual_package_bpp', 'bpp'),
    'target_bpp': ('target_bpp', 'budget_bpp'),
    'images': ('images', 'n'),
    'bytes': ('package_bytes', 'bytes'),
    'psnr_pooled': ('psnr_pooled', 'pooled_psnr'),
    'psnr_mean': ('psnr_mean', 'psnr'),
    'lpips': ('lpips_alex', 'lpips'),
    'lpips_vgg': ('lpips_vgg',),
    'dists': ('dists',),
    'msssim': ('msssim', 'ms_ssim'),
    'qhat': ('qhat_v0', 'qhat'),
    'clipiqa': ('clipiqa', 'clip_iqa'),
    'decode_s': ('decode_s',),
}
QUALITY_FIELDS = {'psnr_pooled', 'psnr_mean', 'lpips', 'lpips_vgg', 'dists', 'msssim', 'qhat', 'clipiqa'}


def numbers(row):
    result = {}
    for key, alternatives in FIELDS.items():
        for alternative in alternatives:
            value = row.get(alternative)
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                try:
                    value = float(value)
                except ValueError:
                    continue
                if math.isfinite(value):
                    result[key] = value
                    break
    return result


def json_records(value, prefix=''):
    if isinstance(value, list):
        for index, entry in enumerate(value, 1):
            yield from json_records(entry, f'{prefix}/{index}')
    elif isinstance(value, dict):
        record = numbers(value)
        if QUALITY_FIELDS.intersection(record):
            yield prefix.strip('/'), value
        for key, entry in value.items():
            if isinstance(entry, (list, dict)):
                yield from json_records(entry, f'{prefix}/{key}')


def make_catalog(inputs, image_details=None):
    rows, inventory, loaded = [], [], set()
    details, history_cache = {}, {}

    def add(raw, source, dataset, protocol, availability, label=None):
        fields = numbers(raw)
        if not QUALITY_FIELDS.intersection(fields):
            return
        model = label or raw.get('display') or raw.get('model') or raw.get('method') or raw.get('arm') or raw.get('cand') or raw.get('name')
        if not isinstance(model, str) or not model:
            return
        setting = ', '.join(f'{key} {raw[key]}' for key in ['quality', 'factor', 'seed'] if key in raw)
        rows.append(dict(model=model, dataset=dataset, protocol=protocol, availability=availability,
                         source=source, setting=setting, **fields))
        return rows[-1]

    def attach_history(row, output):
        allocation = ROOT / row.get('source', '')
        metrics = allocation.with_name('metrics.csv')
        if not allocation.is_file() or not metrics.is_file():
            return
        if allocation not in history_cache:
            with metrics.open(encoding='utf-8-sig', newline='') as handle:
                measurements = {(r.get('cand'), r.get('key'), r.get('name')): r for r in csv.DictReader(handle)}
            with allocation.open(encoding='utf-8-sig', newline='') as handle:
                history_cache[allocation] = (measurements, list(csv.DictReader(handle)))
        measurements, selected = history_cache[allocation]
        records = []
        for choice in selected:
            if choice.get('cand') != row['method']:
                continue
            target = row.get('budget_bpp')
            if target and float(choice.get('rate') or -1) != float(target):
                continue
            key = (choice.get('point_cand', choice['cand']), choice.get('key'), choice.get('name'))
            metric = measurements.get(key)
            if metric is None:
                continue
            records.append(dict(source=choice['name'], **numbers({**metric, 'bytes': choice.get('bytes')})))
        if not records or len(records) != output.get('images') or len({r['source'] for r in records}) != len(records):
            return
        key = f'historical/{len(details)}'
        details[key] = records
        output['details'] = key

    report = ROOT / 'results/2026-10-20_metrics_comparison'
    for filename, dataset in [('full30_budget_metrics.csv', 'CLIC30 · ảnh đầy đủ'),
                              ('aeic_fixed_metrics.csv', 'CLIC30 · ảnh đầy đủ'),
                              ('pilot6_budget_metrics.csv', 'CLIC · pilot 6 ảnh')]:
        path = report / filename
        loaded.add(path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                output = add(row, path.relative_to(ROOT).as_posix(), dataset,
                             'Ngân sách toàn tập · lịch sử' if row.get('budget_bpp') else 'Điểm đo · lịch sử',
                             'Có số liệu · chưa xác minh ảnh')
                if output:
                    attach_history(row, output)

    path = ROOT / 'results/2026-11-02_mlicpp/full30/comparison.json'
    loaded.add(path)
    for row in json.loads(path.read_text(encoding='utf-8'))['results']:
        add(row, path.relative_to(ROOT).as_posix(), 'CLIC30 · ảnh đầy đủ', 'MLIC++ · đối chiếu 02/11',
            'Có số liệu · kiểm ảnh trước khi ghép')
    inventory.append({'source': 'reports/Metric_comparison.md · DC-AE + S1 + residual VTM',
                      'status': 'Thiếu metric (28/30 ảnh); không xếp hạng cùng các kết quả đủ 30 ảnh'})

    sources = {
        'clic30-budget': ('CLIC30 · ảnh đầy đủ', 'Cùng bitstream · phân bổ 09/11'),
        'clic30-extended': ('CLIC30 · ảnh đầy đủ', 'Cùng bitstream · phân bổ 09/11'),
        'div2k32-budget': ('DIV2K32 · development đầy đủ', 'Cùng bitstream · phân bổ 09/11'),
        'div2k32-grid': ('DIV2K32 · development đầy đủ', 'Lưới lượng tử'),
        'div2k-crop-multirate': ('DIV2K32 · development crop 256', 'Cùng bitstream · single/mixed'),
        'fullrd-server': ('DIV2K32 · test campaign', 'Full RD · server'),
        'lab09': ('DIV2K32 · test campaign', 'Full RD · lab09'),
    }
    for name, (dataset, protocol) in sources.items():
        path = inputs / f'{name}.json'
        if not path.exists():
            inventory.append({'source': path.name, 'status': 'Chờ dữ liệu'}); continue
        availability = ('Có ảnh và bitstream' if name in {'clic30-budget', 'clic30-extended'}
                        else 'Có số liệu · chưa xác minh ảnh')
        for row in json.loads(path.read_text(encoding='utf-8-sig')):
            if 'metrics' in row:
                if not row.get('feasible'):
                    continue
                for model, metric in row['metrics'].items():
                    add({**row, **metric}, path.name, dataset, protocol, availability, model)
            else:
                add(row, path.name, dataset, protocol, availability)

    # Keep older metrics discoverable with their source as the comparison boundary.
    for path in sorted((ROOT / 'results').rglob('*')):
        if not path.is_file() or path in loaded or 'portable-package-check' in path.parts:
            continue
        if path.name not in {'summary.json', 'comparison.json', 'matched_rate.json', 'metrics.csv',
                             'summary.csv', 'comparison.csv', 'per_image.csv', 'allocation.csv', 'budgets.json'}:
            continue
        source = path.relative_to(ROOT).as_posix()
        previous = len(rows)
        try:
            if path.name in {'summary.json', 'comparison.json'}:
                for label, row in json_records(json.loads(path.read_text(encoding='utf-8-sig'))):
                    add(row, source, 'Khác · kiểm bộ ảnh theo nguồn', source, 'Có số liệu · kiểm giao thức',
                        None if any(isinstance(row.get(key), str) for key in ('model', 'method', 'arm', 'cand', 'display')) else label)
            elif path.name in {'summary.csv', 'comparison.csv'} and not path.with_suffix('.json').exists():
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    for row in csv.DictReader(handle):
                        if row.get('feasible', '').lower() not in {'false', '0', 'no'}:
                            row.setdefault('target_bpp', row.get('rate'))
                            add(row, source, 'Khác · kiểm bộ ảnh theo nguồn', source, 'Có số liệu · kiểm giao thức')
        except (ValueError, UnicodeError, OSError):
            inventory.append({'source': source, 'status': 'Cần đọc định dạng nguồn'}); continue
        inventory.append({'source': source, 'status': f'{len(rows)-previous} điểm' if len(rows) > previous else 'Artifact chi tiết'})
    clic_decoders = {'original': 'MLIC++ gốc', 'perceptual': 'MLIC++ perceptual',
                     'single': 'MLIC++ single', 'mixed': 'MLIC++ mixed'}
    if image_details:
        for row in json.loads(image_details.read_text(encoding='utf-8')):
            key = f"{row['model']}@{row['target_bpp']:g}"
            details.setdefault(key, []).append(row)
    for row in rows:
        if row['source'] in {'clic30-budget.json', 'clic30-extended.json'}:
            if image_details:
                key = f"{row['model']}@{row['target_bpp']:g}"
                if len(details.get(key, [])) != 30:
                    raise ValueError(f'Incomplete image metrics for {key}')
                row['details'] = key
            row['model'] = clic_decoders[row['model']]
    return {'version': 1, 'rows': rows, 'inventory': inventory, 'details': details}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image-details', type=Path)
    args = parser.parse_args()
    catalog = make_catalog(args.inputs, args.image_details)
    args.out.write_text(json.dumps(catalog, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'{len(catalog["rows"])} metric rows, {len(catalog["inventory"])} inventory entries')
