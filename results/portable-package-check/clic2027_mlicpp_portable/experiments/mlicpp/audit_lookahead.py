"""CPU-only evidence audit; no Torch, source images or codec initialization."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from stream import unpack


def audit(root):
    checked = 0
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        checked += 1
    rows = json.loads((root / 'metrics.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    selections = []
    for name in sorted({r['name'] for r in rows}):
        group = [r for r in rows if r['name'] == name]
        base = group[0]
        assert base['candidate'] == 0 and len(group) == 16
        for row in group:
            blob = (root / 'streams' / row['stream']).read_bytes()
            assert len(blob) == row['bytes']
            assert hashlib.sha256(blob).hexdigest() == row['stream_sha256']
            q, legacy, size, shape, strings = unpack(blob)
            assert q == 2 and legacy and size == (256, 256)
            assert row['bpp'] == 8 * len(blob) / (size[0] * size[1])
            assert abs(row['j'] - (row['bpp'] + config['lambda_mse'] * row['mse'])) < 1e-12
        eligible = [r for r in group if r['current_cdf_delta_bpp'] is not None]
        local_score = lambda r: r['current_cdf_delta_bpp'] + config['lambda_mse'] * (r['mse'] - base['mse'])
        full_score = lambda r: local_score(r) + r['future_cdf_delta_bpp']
        selections.append(dict(name=name, eligible=len(eligible),
            baseline=base, best_all=min(group, key=lambda r:r['j']),
            best_eligible=min(eligible, key=lambda r:r['j']),
            myopic=min(eligible, key=local_score), full_cdf=min(eligible, key=full_score)))
    modes = ['baseline', 'best_all', 'best_eligible', 'myopic', 'full_cdf']
    means = {mode:{key:float(np.mean([r[mode][key] for r in selections]))
                   for key in ['bpp', 'mse', 'j']} for mode in modes}
    rng = np.random.default_rng(config['seed'])
    ix = rng.integers(0, len(selections), (2000, len(selections)))
    comparisons = {}
    for mode, ref in [('best_all', 'baseline'), ('best_eligible', 'myopic'), ('full_cdf', 'myopic')]:
        delta = np.array([r[mode]['j'] - r[ref]['j'] for r in selections])
        comparisons[mode + '_minus_' + ref] = dict(delta_j=float(delta.mean()),
            relative_percent=float(delta.mean() / means[ref]['j'] * 100),
            ci95=np.quantile(delta[ix].mean(1), [.025, .975]).tolist())
    fresh = json.loads((root / 'fresh_decode.json').read_text())
    assert fresh['passed'] == fresh['expected'] == len(rows) == 64
    result = dict(verified_files=checked, verified_streams=len(rows), fresh_decode=fresh,
        means=means, comparisons=comparisons,
        excluded_from_cdf_diagnostic=sum(16-r['eligible'] for r in selections),
        changed_choice=sum(r['myopic']['candidate'] != r['full_cdf']['candidate'] for r in selections),
        max_suffix_symbols_changed=max(r['suffix_symbols_changed'] for r in rows),
        selected=[dict(name=r['name'], eligible=r['eligible'],
                       **{mode:r[mode]['candidate'] for mode in modes}) for r in selections],
        note='Post-run audit: matched candidate sets for CDF comparison; no new model selection or training.')
    (root / 'local_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
