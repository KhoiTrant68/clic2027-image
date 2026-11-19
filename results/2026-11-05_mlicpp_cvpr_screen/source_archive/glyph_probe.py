"""Controlled semantic minimal pairs: actual code aliases vs decoder/OCR errors.

Diagnostic only, not a proposed collision attack, new codec, or source-legibility proof.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe import ROOT, load, settings, sha, write_json
from stream import pack, unpack

OCR = Path('/home/jovyan/clic2027-envs/ocr')
PILOT = Path('/home/jovyan/clic2027-artifacts/mlicpp/pilot20261104/div2k224-q2-risk-adapter-seed1984-rational')


def ocr(path, box, work):
    with Image.open(path) as image:
        crop = image.convert('RGB').crop(box)
        crop = crop.resize((crop.width*4, crop.height*4), Image.Resampling.BICUBIC)
        crop.save(work)
    env = dict(os.environ, OMP_THREAD_LIMIT='1', TESSDATA_PREFIX=str(OCR / 'share/tessdata'))
    result = subprocess.run([str(OCR / 'bin/tesseract'), str(work), 'stdout', '-l', 'eng',
        '--psm', '7', '--oem', '1', '-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'],
        capture_output=True, text=True, check=True, timeout=30, env=env)
    return re.sub('[^A-Z0-9]', '', result.stdout.upper())


def generate(out):
    fonts = [OCR / 'fonts/DejaVuSans.ttf', OCR / 'fonts/Inconsolata-Regular.ttf']
    if any(not p.is_file() for p in fonts) or not (OCR / 'bin/tesseract').is_file():
        raise ValueError('Missing pinned OCR environment/fonts')
    if (out / 'data.json').exists():
        raise ValueError('Refuse to replace generated data')
    (out / 'source').mkdir()
    (out / 'ocr_work').mkdir()
    rows = []
    combinations = itertools.product(range(2), [10, 14, 20, 28], [64, 192], ['flat', 'texture'], [('6', '8'), ('E', 'F')])
    for family, (font_id, size, contrast, background, characters) in enumerate(combinations):
        font = ImageFont.truetype(str(fonts[font_id]), size)
        cell = int(np.ceil(max(font.getlength(c) for c in 'ABDEFG68')))+1
        x0, y0 = (256-5*cell)//2, (256-size)//2
        rng = np.random.default_rng(2037+family)
        bg = np.full((256, 256, 3), 224, dtype='uint8')
        if background == 'texture':
            small = Image.fromarray(rng.integers(0, 25, size=(16, 16), dtype='uint8'))
            noise = np.asarray(small.resize((256, 256), Image.Resampling.BICUBIC), dtype='int16')-12
            bg = np.clip(bg.astype('int16')+noise[..., None], 0, 255).astype('uint8')
        for variant, character, shift in [('a', characters[0], 0), ('b', characters[1], 0), ('nuisance', characters[0], 1)]:
            text = 'AB'+character+'DE'
            image = Image.fromarray(bg.copy())
            draw = ImageDraw.Draw(image)
            for j, c in enumerate(text):
                draw.text((x0+j*cell+shift, y0), c, font=font, fill=(224-contrast,)*3, anchor='lt')
            name = f'{family:03d}-{variant}'
            path = out / 'source' / (name+'.png')
            image.save(path)
            box = [max(0, x0-8), max(0, y0-8), min(256, x0+5*cell+10), min(256, y0+size+12)]
            recognized = ocr(path, box, out / 'ocr_work' / (name+'-source.png'))
            row = dict(name=name, family=family, variant=variant, label=text, font=fonts[font_id].name,
                       size=size, contrast=contrast, background=background, bbox=box,
                       source_ocr=recognized, source_ocr_correct=recognized==text, sha256=sha(path))
            rows.append(row)
        if family % 8 == 0:
            print(json.dumps(dict(stage='generate', families=family+1)), flush=True)
    write_json(out / 'data.json', dict(seed=2037, rows=rows, fonts={p.name: sha(p) for p in fonts},
        warning='Procedural glyphs; OCR-correct originals are an operational filter, not human legibility certification',
        renderer='fixed character cells; a/b differs only at one glyph; nuisance translates all glyphs by one pixel',
        ocr_version=subprocess.check_output([str(OCR / 'bin/tesseract'), '--version'], text=True).splitlines()[0]))


def args_for(out):
    return SimpleNamespace(out=out, upstream=ROOT / 'refs/repos/MLIC',
        weights=Path('/home/jovyan/clic2027-artifacts/mlicpp/weights'), legacy=True)


def setup(out, label):
    torch, install = settings()
    stop = install(out / (label+'.vram.jsonl'))
    return torch, stop


def encode(out, quality):
    torch, stop = setup(out, f'encode-q{quality}')
    model, meta = load(args_for(out), quality)
    folder = out / f'q{quality}'
    folder.mkdir(exist_ok=False)
    for sub in ['streams', 'base', 'symbols'] + (['perceptual'] if quality == 2 else []):
        (folder / sub).mkdir()
    data = json.loads((out / 'data.json').read_text())['rows']
    traces = {'y': [], 'z': []}
    original_y, original_z = model.gaussian_conditional.quantize, model.entropy_bottleneck.quantize

    def wrapped(original, key):
        def call(value, mode, means=None):
            result = original(value, mode, means)
            if mode == 'symbols':
                traces[key].append(result.detach().cpu().numpy().astype('int32').ravel())
            return result
        return call
    model.gaussian_conditional.quantize = wrapped(original_y, 'y')
    model.entropy_bottleneck.quantize = wrapped(original_z, 'z')
    adapter = None
    if quality == 2:
        from pilot_model import Adapter
        adapter = Adapter(hidden=32, amplitude=.25).cuda().eval()
        adapter.load_state_dict(torch.load(PILOT / 'mean/last.pt', map_location='cpu', weights_only=True)['adapter'])
    records = []
    started = time.perf_counter()
    for source in data:
        path = out / 'source' / (source['name']+'.png')
        if sha(path) != source['sha256']:
            raise ValueError('Source changed')
        pixels = np.array(Image.open(path).convert('RGB'))
        x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().cuda()/255
        traces['y'].clear()
        traces['z'].clear()
        model.update_resolutions(16, 16)
        with torch.no_grad():
            encoded = model.compress(x)
        if len(traces['y']) != 20 or len(traces['z']) != 1:
            raise ValueError(f'Unexpected actual symbol trace: {len(traces["y"])}, {len(traces["z"])}')
        symbols_y = np.concatenate(traces['y'])
        symbols_z = np.concatenate(traces['z'])
        np.savez_compressed(folder / 'symbols' / (source['name']+'.npz'), y=symbols_y, z=symbols_z)
        blob = pack(quality, True, (256, 256), tuple(encoded['shape']), encoded['strings'])
        (folder / 'streams' / (source['name']+'.bin')).write_bytes(blob)
        values = dict(name=source['name'], bytes=len(blob), bpp=8*len(blob)/65536,
            stream_sha256=hashlib.sha256(blob).hexdigest(),
            symbol_sha256=hashlib.sha256(symbols_z.tobytes()+symbols_y.tobytes()).hexdigest(),
            y_symbols=int(symbols_y.size), z_symbols=int(symbols_z.size))
        for mode in ['base'] + (['perceptual'] if adapter is not None else []):
            hook = None
            if mode == 'perceptual':
                hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda module, args: (adapter(args[0])[0],))
            model.update_resolutions(16, 16)
            with torch.no_grad():
                output = model.decompress(encoded['strings'], encoded['shape'])['x_hat'].clamp(0, 1)
            if hook is not None:
                hook.remove()
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite decode')
            reconstructed = output[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            dest = folder / mode / (source['name']+'.png')
            Image.fromarray(reconstructed).save(dest)
            values[mode+'_pixel_sha256'] = hashlib.sha256(reconstructed.tobytes()).hexdigest()
            recognized = ocr(dest, source['bbox'], out / 'ocr_work' / (source['name']+f'-q{quality}-{mode}.png'))
            values[mode+'_ocr'] = recognized
            values[mode+'_correct'] = recognized == source['label']
        records.append(values)
        write_json(out / f'q{quality}.progress.json', dict(completed=len(records), total=len(data),
            seconds=time.perf_counter()-started, last=values))
        if len(records) % 24 == 0:
            print(json.dumps(dict(quality=quality, completed=len(records))), flush=True)
    model.gaussian_conditional.quantize = original_y
    model.entropy_bottleneck.quantize = original_z
    meta.update(records=records, direct_streams=True, actual_symbols=True,
        adapter_sha256=sha(PILOT / 'mean/last.pt') if adapter is not None else None)
    write_json(out / f'q{quality}.json', meta)
    stop.set()


def verify(out, quality):
    torch, stop = setup(out, f'verify-q{quality}')
    model, _ = load(args_for(out), quality)
    rows = json.loads((out / f'q{quality}.json').read_text())['records']
    adapter = None
    if quality == 2:
        from pilot_model import Adapter
        adapter = Adapter(hidden=32, amplitude=.25).cuda().eval()
        adapter.load_state_dict(torch.load(PILOT / 'mean/last.pt', map_location='cpu', weights_only=True)['adapter'])
    matches = 0
    for row in rows:
        blob = (out / f'q{quality}/streams' / (row['name']+'.bin')).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row['stream_sha256']:
            raise ValueError('Stream hash changed')
        q, legacy, size, shape, strings = unpack(blob)
        if q != quality or not legacy or size != (256, 256):
            raise ValueError('Unexpected stream header')
        for mode in ['base'] + (['perceptual'] if adapter is not None else []):
            hook = None
            if adapter is not None and mode == 'perceptual':
                hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda module, args: (adapter(args[0])[0],))
            model.update_resolutions(16, 16)
            with torch.no_grad():
                pixels = model.decompress(strings, shape)['x_hat'].clamp(0, 1)[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            if hook is not None:
                hook.remove()
            if hashlib.sha256(pixels.tobytes()).hexdigest() != row[mode+'_pixel_sha256']:
                raise ValueError(f'Fresh decode mismatch: q{quality}/{row["name"]}/{mode}')
            matches += 1
    write_json(out / f'q{quality}.parity.json', dict(matches=matches, images=len(rows), fresh_process=True))
    stop.set()


def analyze(out):
    data = json.loads((out / 'data.json').read_text())['rows']
    families = defaultdict(dict)
    for row in data:
        families[row['family']][row['variant']] = row
    summary, pairs = [], []
    for q in [1, 2]:
        records = {r['name']: r for r in json.loads((out / f'q{q}.json').read_text())['records']}
        parity = json.loads((out / f'q{q}.parity.json').read_text())
        if parity['images'] != len(data):
            raise ValueError('Incomplete parity')
        for mode in ['base'] + (['perceptual'] if q == 2 else []):
            eligible = [r for r in data if r['variant'] in ('a', 'b') and r['source_ocr_correct']]
            both_eligible = 0
            exact_collisions = 0
            symbol_collisions = 0
            for family, group in families.items():
                a, b, nuisance = [group[k] for k in ['a', 'b', 'nuisance']]
                ra, rb, rn = [records[r['name']] for r in [a, b, nuisance]]
                sa = np.load(out / f'q{q}/symbols' / (a['name']+'.npz'))
                sb = np.load(out / f'q{q}/symbols' / (b['name']+'.npz'))
                sn = np.load(out / f'q{q}/symbols' / (nuisance['name']+'.npz'))
                ok = a['source_ocr_correct'] and b['source_ocr_correct']
                both_eligible += int(ok)
                exact = ra['stream_sha256'] == rb['stream_sha256']
                same_symbols = ra['symbol_sha256'] == rb['symbol_sha256']
                exact_collisions += int(ok and exact)
                symbol_collisions += int(ok and same_symbols)
                pairs.append(dict(quality=q, method=mode, family=family, source_pair_ocr_correct=ok,
                    exact_stream_collision=exact, exact_symbol_collision=same_symbols,
                    semantic_y_changes=int((sa['y'] != sb['y']).sum()), semantic_z_changes=int((sa['z'] != sb['z']).sum()),
                    nuisance_y_changes=int((sa['y'] != sn['y']).sum()), nuisance_z_changes=int((sa['z'] != sn['z']).sum()),
                    reconstructed_pair_same_text=ra[mode+'_ocr']==rb[mode+'_ocr'],
                    a_correct=ra[mode+'_correct'], b_correct=rb[mode+'_correct'],
                    size=a['size'], contrast=a['contrast'], background=a['background'], font=a['font']))
            summary.append(dict(quality=q, method=mode, total_images=len(data),
                semantic_originals=len([r for r in data if r['variant'] in ('a', 'b')]),
                source_ocr_correct=len(eligible), source_pair_ocr_correct=both_eligible,
                exact_stream_collisions_on_eligible_pairs=exact_collisions,
                exact_symbol_collisions_on_eligible_pairs=symbol_collisions,
                reconstruction_correct_on_eligible=sum(records[r['name']][mode+'_correct'] for r in eligible),
                actual_bpp=sum(r['bytes'] for r in records.values())*8/(len(records)*65536)))
    write_json(out / 'summary.json', summary)
    write_json(out / 'pairs.json', pairs)
    print(json.dumps(summary, indent=2))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    frozen = {name: sha(Path(__file__).with_name(name)) for name in ['glyph_probe.py', 'pilot_model.py', 'probe.py', 'stream.py']}
    write_json(out / 'sources.sha256.json', frozen)
    stages = [('generate', None), ('encode', 1), ('encode', 2), ('verify', 1), ('verify', 2), ('analyze', None)]
    for stage, q in stages:
        write_json(out / 'status.json', dict(state='running', stage=stage, quality=q))
        cmd = [sys.executable, __file__, '--out', str(out), '--stage', stage]
        if q:
            cmd += ['--quality', str(q)]
        with (out / (stage+f'-q{q}.log')).open('w') as log:
            result = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result:
            write_json(out / 'status.json', dict(state='failed', stage=stage, quality=q, exit_code=result))
            raise RuntimeError(f'{stage}/{q}: {result}')
    if any(sha(Path(__file__).with_name(name)) != value for name, value in frozen.items()):
        raise ValueError('Source changed during probe')
    write_json(out / 'status.json', dict(state='complete', qualities=[1, 2], diagnostic_only=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['run', 'generate', 'encode', 'verify', 'analyze'], required=True)
    p.add_argument('--quality', type=int, choices=[1, 2])
    args = p.parse_args()
    if args.stage in ('encode', 'verify'):
        if args.quality is None:
            p.error('quality required')
        globals()[args.stage](args.out, args.quality)
    else:
        globals()[args.stage](args.out)
