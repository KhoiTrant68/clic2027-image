"""Fixed 1-NN glyph readout across fonts. Diagnostic, never a deployment decoder.

No test-selected hyperparameters. Central-glyph labels, not whole-word OCR.
Privileged rendering boxes apply equally to all pixel probes; symbol probe uses
the entire actual y/z sequence. Probe success/failure does not establish sufficiency.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def pixels(path, row):
    # Exact generator cell geometry; fixed four-class alphabet and source bbox.
    # Cell pitch can be recovered from annotated x0 and the 256px centered layout.
    box = row['bbox']
    x0 = box[0]+8
    cell = (256-2*x0)//5
    y0 = (256-row['size'])//2
    with Image.open(path) as image:
        image = image.convert('L').crop((x0+2*cell-2, y0-2, x0+3*cell+2, y0+row['size']+2))
        return np.asarray(image.resize((32,32), Image.Resampling.BICUBIC), dtype=np.float32).ravel()/255


def nearest(train, test, labels):
    # Train-only centering, per-example normalization, fixed Euclidean 1-NN.
    center = train.mean(0)
    def normalize(value):
        value = value-center
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
    a, b = normalize(train), normalize(test)
    distances = np.maximum(0, (b*b).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*b@a.T)
    return labels[np.argmin(distances, axis=1)]


def main(root, out):
    if out.exists():
        raise ValueError('Refuse to replace a diagnostic result')
    assert read(root/'status.json')['state']=='complete'
    rows = [r for r in read(root/'data.json')['rows'] if r['variant'] in ['a','b']]
    fonts = sorted({r['font'] for r in rows})
    assert len(rows)==128 and len(fonts)==2
    results, details = [], []
    labels = np.asarray([r['label'][2] for r in rows])
    eligible = np.asarray([r['source_ocr_correct'] for r in rows])
    for q in [1,2]:
        records = {r['name']:r for r in read(root/f'q{q}.json')['records']}
        arrays = {'source_pixels':np.stack([pixels(root/'source'/(r['name']+'.png'),r) for r in rows])}
        for mode in ['base']+(['perceptual'] if q==2 else []):
            arrays[mode+'_pixels'] = np.stack([pixels(root/f'q{q}'/mode/(r['name']+'.png'),r) for r in rows])
        symbols = []
        for row in rows:
            with np.load(root/f'q{q}/symbols'/(row['name']+'.npz')) as codes:
                symbols.append(np.concatenate([codes['y'], codes['z']]).astype(np.float32))
        arrays['actual_symbols'] = np.stack(symbols)
        for font in fonts:
            test = np.asarray([r['font']==font for r in rows])
            train = ~test
            for method, values in arrays.items():
                predicted = nearest(values[train], values[test], labels[train])
                mask = eligible[test]
                results.append(dict(quality=q, test_font=font, method=method,
                    n=int(test.sum()), correct=int((predicted==labels[test]).sum()),
                    source_ocr_eligible=int(mask.sum()), correct_eligible=int((predicted[mask]==labels[test][mask]).sum())))
                for row, prediction in zip(np.asarray(rows, dtype=object)[test], predicted):
                    details.append(dict(quality=q, test_font=font, method=method, name=row['name'],
                        label=row['label'][2], prediction=str(prediction), source_ocr_correct=row['source_ocr_correct']))
            for mode in ['base']+(['perceptual'] if q==2 else []):
                subset = np.asarray(rows, dtype=object)[test]
                predicted = np.asarray([records[r['name']][mode+'_ocr'][2] if len(records[r['name']][mode+'_ocr'])==5 else '?' for r in subset])
                mask = eligible[test]
                results.append(dict(quality=q, test_font=font, method=mode+'_ocr_center',
                    n=int(test.sum()), correct=int((predicted==labels[test]).sum()),
                    source_ocr_eligible=int(mask.sum()), correct_eligible=int((predicted[mask]==labels[test][mask]).sum())))
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'predictions.json').write_text(json.dumps(details,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(dict(
        train_test='Leave one font out; no nuisance variants; 64 train and 64 test per fold.',
        frozen_readout='1-NN Euclidean; train-only feature centering then per-vector L2 normalization.',
        target='Four central glyph classes 6/8/E/F; chance 25% on balanced full set.',
        pixel_access='Privileged generator geometry, same for source/base/perceptual probes.',
        symbol_access='Whole actual entropy y/z symbols, unequal feature dimension vs pixel controls.',
        limitations='Two fonts, tiny synthetic distribution, known alignment, no statistical equivalence of representations. Neither failure nor success proves information loss/sufficiency.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    main(args.root,args.out)
