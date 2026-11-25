"""Materialize only missing opaque images for the CLIC30 frozen-stream round."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    for name in ['budgets', 'extended', 'out']:
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if read(args.extended / 'status.json')['state'] != 'complete':
        raise ValueError('Full-resolution decoder evaluation is not complete')
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.out / 'asset'
    assets.mkdir()
    old = args.budgets / 'bpp-0.15/study/assets'
    existing = {hashlib.sha256(path.read_bytes()).hexdigest(): path.stem for path in old.glob('*.png')}
    original_ids = set(existing.values())
    extra = {(r['target_bpp'], r['model'], r['name']): r for r in read(args.extended / 'per_image.json')}
    manifest = []
    details = []
    grid = args.budgets.parent / 'grid'
    metrics = {}
    for model in ['original', 'perceptual']:
        for factor in [0.5, 1, 2, 4]:
            metrics.update({(model, factor, r['name']): r for r in read(grid / f'{model}-factor{factor:g}/metrics.json')})

    def asset(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in existing:
            return existing[digest]
        key = digest[:48]
        target = assets / f'{key}.png'
        if target.exists() or key in original_ids:
            raise ValueError('Opaque asset ID collision')
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)
        existing[digest] = key
        return key

    references = {}
    for rate in [0.075, 0.15, 0.3]:
        folder = args.budgets / f'bpp-{rate:g}'
        study = folder / 'study'
        selections = {row['name']: row for row in read(folder / 'selection.json')}
        for image in read(study / 'private_manifest.json')['images']:
            name = image['source']
            selection = selections.pop(name)
            ref = study / 'assets' / f"{image['reference']}.png"
            reference = asset(ref)
            with Image.open(ref) as img:
                size = img.size
            if size[0]*size[1] != selection['pixels'] or (name in references and references[name] != reference):
                raise ValueError(f'Reference differs across rates: {name}')
            references[name] = reference
            models = {model: asset(study / 'assets' / f'{opaque}.png') for model, opaque in image['codecs'].items()}
            measured = {}
            for model in models:
                row = metrics[model, selection['factor'], name]
                if row['stream_sha256'] != selection['stream_sha256']:
                    raise ValueError(f'Original study stream differs: {name}, {model}')
                measured[model] = row
            for model in ['single', 'mixed']:
                row = extra.pop((rate, model, name))
                if any(row[key] != selection[key] for key in ['factor', 'pixels', 'bytes', 'stream_sha256']):
                    raise ValueError(f'Mismatched stream: {name}, {model}, {rate}')
                path = args.extended / row['recon']
                with Image.open(path) as img:
                    if img.size != size or hashlib.sha256(img.convert('RGB').tobytes()).hexdigest() != row['pixel_sha256']:
                        raise ValueError(f'Invalid reconstruction: {path}')
                models[model] = asset(path)
                measured[model] = row
            if set(models) != {'original', 'perceptual', 'single', 'mixed'}:
                raise ValueError('Unexpected model list')
            manifest.append({'dataset': 'clic30-full', 'target_bpp': rate, 'source': name,
                             'reference': reference, 'models': models, **{k: selection[k] for k in ['bytes', 'pixels', 'stream_sha256']}})
            for model, row in measured.items():
                details.append(dict(model=model, target_bpp=rate, source=name, reference=reference, asset=models[model],
                                    **{key: row[key] for key in ['psnr', 'lpips', 'dists', 'bytes', 'pixels', 'stream_sha256']}))
        if selections:
            raise ValueError('Study and selection image sets differ')
    if len(manifest) != 90 or len(references) != 30 or extra:
        raise ValueError('Missing result or unexpected excess data')
    (args.out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (args.out / 'per_image.json').write_text(json.dumps(details, allow_nan=False), encoding='utf-8')
    print(f'{len(manifest)} image/rate rows, {len(list(assets.iterdir()))} extra assets')


if __name__ == '__main__':
    main()
