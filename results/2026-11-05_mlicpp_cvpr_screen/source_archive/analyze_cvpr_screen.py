"""Audit completed screens; paired image/family bootstrap, no model selection."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    return dict(delta=float(values.mean()), ci95=np.quantile(values[indices].mean(1), [.025, .975]).tolist())


def main(root):
    glyph = root / 'glyph-minimalpairs-q1q2'
    projection = root / 'projection-fresh64-validationmatched'
    for run in [glyph, projection]:
        assert read(run / 'status.json')['state'] == 'complete'
        for name, expected in read(run / 'sources.sha256.json').items():
            archive = root / 'source_archive' / name
            source = archive if archive.exists() else Path(__file__).with_name(name)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
    assert read(glyph / 'q1.parity.json')['matches'] == 192
    assert read(glyph / 'q2.parity.json')['matches'] == 384
    assert read(projection / 'fresh.json')['matches'] == 896
    rows = read(projection / 'test_metrics.json')
    names = sorted(r['name'] for r in rows if r['method'] == 'base')
    methods = sorted({r['method'] for r in rows})
    by = {(r['method'], r['name']): r for r in rows}
    assert len(names) == len(set(names)) == 64
    assert len(by) == len(rows) == len(methods)*len(names)
    for name in names:
        assert len({by[m, name]['bytes'] for m in methods}) == 1
    selected = read(projection / 'selected.json')
    assert all(r['control']['alpha'] == 1. for r in selected)
    indices = np.random.default_rng(2037).integers(0, 64, size=(10000, 64))
    comparisons = []
    contrasts = [(r['name'], r['name']+'-matched-blend') for r in selected]
    contrasts += [(f'moment-{b}', f'random-{b}') for b in [8, 16, 32]]
    for treatment, control in contrasts:
        for metric in ['lpips', 'dists', 'edge', 'mse']:
            delta = [by[treatment, n][metric]-by[control, n][metric] for n in names]
            comparisons.append(dict(treatment=treatment, control=control, metric=metric, **interval(delta, indices)))
    source = [r for r in read(glyph / 'data.json')['rows'] if r['variant'] in ('a', 'b') and r['source_ocr_correct']]
    q2 = {r['name']:r for r in read(glyph / 'q2.json')['records']}
    transitions = Counter((q2[r['name']]['base_correct'], q2[r['name']]['perceptual_correct']) for r in source)
    harmed, recovered = transitions[True, False], transitions[False, True]
    discordant = harmed+recovered
    p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(harmed, recovered)+1))/2**discordant) if discordant else 1.
    # Cluster a/b together: bootstrap 64 families, retaining the source-only eligibility mask.
    groups = np.zeros((64, 2))
    for row in source:
        groups[row['family'], 0] += int(q2[row['name']]['perceptual_correct'])-int(q2[row['name']]['base_correct'])
        groups[row['family'], 1] += 1
    samples = groups[indices].sum(1)
    strata = []
    for field in ['font', 'size', 'contrast', 'background']:
        for value in sorted({r[field] for r in source}):
            subset = [r for r in source if r[field] == value]
            strata.append(dict(field=field, value=value, eligible=len(subset), **{m:sum(q2[r['name']][m+'_correct'] for r in subset) for m in ['base','perceptual']}))
    result = dict(bootstrap_replicates=10000, seed=2037,
        inference='Exploratory pointwise intervals; no multiplicity correction, not a confirmatory study.',
        projection=comparisons,
        validation_unmatched=[r['name'] for r in selected if not r['matched_validation']],
        glyph_q2=dict(eligible=len(source), harmed=harmed, recovered=recovered,
            both_correct=transitions[True,True], both_wrong=transitions[False,False],
            accuracy_delta=(recovered-harmed)/len(source),
            family_cluster_ci95=np.quantile(samples[:,0]/samples[:,1], [.025,.975]).tolist(),
            image_mcnemar_exact_p=p, mcnemar_caveat='Images within a family are dependent; this p-value is descriptive only.',
            strata=strata), fresh_outputs_verified=1472, source_hashes_verified=True)
    (root / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
