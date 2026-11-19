"""Falsify a hard-invariant hypothesis against validation-matched scalar blending.

Hard projections are prior-art controls, not claimed inventions. All test methods
use the same actual q2 stream; parameters are fixed on the old validation split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from hard_constraints import project
from probe import ROOT, load, settings, sha, write_json
from stream import pack, unpack

ART = Path('/home/jovyan/clic2027-artifacts')
OLD_CACHE = ART / 'mlicpp/pilot20261104/div2k224-q2-crop256'
ADAPTER = ART / 'mlicpp/pilot20261104/div2k224-q2-risk-adapter-seed1984-rational/mean/last.pt'
DATA = ART / 'data/div2k/cvpr-holdout64'


def setup(out, label):
    torch, install = settings()
    return torch, install(out / (label+'.vram.jsonl'))


def arguments(out):
    return SimpleNamespace(out=out, upstream=ROOT / 'refs/repos/MLIC',
        weights=ART / 'mlicpp/weights', legacy=True, cache=OLD_CACHE)


def adapter_model(torch):
    from pilot_model import Adapter
    adapter = Adapter(hidden=32, amplitude=.25).cuda().eval()
    adapter.load_state_dict(torch.load(ADAPTER, map_location='cpu', weights_only=True)['adapter'])
    return adapter


def uint8(value):
    return value[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def prepare(out):
    torch, stop = setup(out, 'prepare')
    from pilot import make_tail
    a = arguments(out)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    old = json.loads((ART / 'data/div2k/pilot224/manifest.json').read_text())
    if {r['sha256'] for r in manifest['images']} & {r['sha256'] for r in old['images']}:
        raise ValueError('New test images overlap previous 224-image subset')
    if len(manifest['images']) != 64:
        raise ValueError('Require 64 fresh source images')
    for split in ['validation', 'test']:
        for method in ['original', 'base', 'perceptual']:
            (out / split / method).mkdir(parents=True)
    (out / 'test/streams').mkdir()
    adapter = adapter_model(torch)
    tail = make_tail(a, torch)
    cached = json.loads((OLD_CACHE / 'manifest.json').read_text())
    records = []
    for r in [r for r in cached['records'] if r['split'] == 'validation']:
        sample = torch.load(OLD_CACHE / 'validation' / (r['name']+'.pt'), weights_only=True)
        with torch.no_grad():
            y = tail(adapter(sample['feature'].cuda())[0])
        arrays = dict(original=sample['original'][0].permute(1, 2, 0).numpy(),
                      base=sample['base'][0].permute(1, 2, 0).numpy(), perceptual=uint8(y))
        for key, image in arrays.items():
            Image.fromarray(image).save(out / 'validation' / key / (r['name']+'.png'))
        records.append(dict(split='validation', name=r['name'], bytes=r['bytes'], pixels=65536))
    del tail
    model, meta = load(a, 2)
    for i, r in enumerate(manifest['images']):
        path = DATA / 'images' / r['name']
        if sha(path) != r['sha256']:
            raise ValueError('Holdout image hash changed')
        with Image.open(path) as image:
            image = image.convert('RGB')
            rng = random.Random('2037/'+r['name'])
            left, top = rng.randrange(image.width-255), rng.randrange(image.height-255)
            original = np.array(image.crop((left, top, left+256, top+256)))
        x = torch.from_numpy(original.copy()).permute(2, 0, 1)[None].float().cuda()/255
        model.update_resolutions(16, 16)
        with torch.no_grad():
            encoded = model.compress(x)
        blob = pack(2, True, (256, 256), tuple(encoded['shape']), encoded['strings'])
        name = path.stem+'-fresh'
        (out / 'test/streams' / (name+'.bin')).write_bytes(blob)
        arrays = dict(original=original)
        for method in ['base', 'perceptual']:
            hook = None
            if method == 'perceptual':
                hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda module, args: (adapter(args[0])[0],))
            model.update_resolutions(16, 16)
            with torch.no_grad():
                arrays[method] = uint8(model.decompress(encoded['strings'], encoded['shape'])['x_hat'])
            if hook is not None:
                hook.remove()
        for key, image in arrays.items():
            Image.fromarray(image).save(out / 'test' / key / (name+'.png'))
        records.append(dict(split='test', name=name, bytes=len(blob), pixels=65536,
            stream_sha256=hashlib.sha256(blob).hexdigest(), source_sha256=r['sha256'], crop=[left, top]))
        write_json(out / 'prepare.progress.json', dict(test_images=i+1, total=64))
    meta.update(records=records, adapter_sha256=sha(ADAPTER), fresh_data_sha256=sha(DATA / 'manifest.json'),
        cache_sha256=sha(OLD_CACHE / 'manifest.json'), warning='posthoc hard-invariant test, not a new trained model')
    write_json(out / 'manifest.json', meta)
    stop.set()


def images(out, split, name):
    return {key: np.array(Image.open(out / split / key / (name+'.png')).convert('RGB'))
            for key in ['original', 'base', 'perceptual']}


def apply(value, method):
    if method['kind'] == 'blend':
        return np.rint(value['base']*(1-method['alpha'])+value['perceptual']*method['alpha']).astype('uint8')
    return project(value['perceptual'], value['base'], method['block'], method['kind']=='random')


def edge_error(a, b):
    # RGB finite-difference error, not the grayscale Sobel surrogate used to train.
    delta = a.astype('float64')/255-b.astype('float64')/255
    return float((np.abs(np.diff(delta, axis=0)).mean()+np.abs(np.diff(delta, axis=1)).mean())/2)


def select(out):
    torch, stop = setup(out, 'validation')
    sys.path.insert(0, str(ROOT / 'src'))
    from clic27.eval import metrics as M
    metric = M.Perceptual('cuda')
    validation = [r for r in json.loads((out / 'manifest.json').read_text())['records'] if r['split']=='validation']
    pairs = [images(out, 'validation', r['name']) for r in validation]
    measured = []
    def score(method):
        lps, edges = [], []
        for value in pairs:
            reconstructed = apply(value, method)
            with torch.no_grad():
                lps.append(metric.lpips(value['original'], reconstructed))
            edges.append(edge_error(value['original'], reconstructed))
        row = dict(**method, lpips=float(np.mean(lps)), edge=float(np.mean(edges)))
        measured.append(row)
        return row
    curve = [score(dict(kind='blend', alpha=i/10)) for i in range(11)]
    selected = []
    for kind in ['moment', 'random']:
        for block in [8, 16, 32]:
            target = score(dict(kind=kind, block=block))
            options = list(curve)
            # Refine every coarse crossing; do not assume global monotonicity.
            crossings = [(a, b) for a, b in zip(curve[:-1], curve[1:])
                         if (a['lpips']-target['lpips'])*(b['lpips']-target['lpips']) <= 0]
            for low, high in crossings:
                for _ in range(5):
                    mid = score(dict(kind='blend', alpha=(low['alpha']+high['alpha'])/2))
                    options.append(mid)
                    if (low['lpips']-target['lpips'])*(mid['lpips']-target['lpips']) <= 0:
                        high = mid
                    else:
                        low = mid
            feasible = [r for r in options if r['lpips'] <= target['lpips']*1.002]
            control = min(feasible, key=lambda r: r['edge']) if feasible else min(options, key=lambda r:r['lpips'])
            selected.append(dict(name=f'{kind}-{block}', method=dict(kind=kind, block=block),
                control=dict(kind='blend', alpha=control['alpha']), matched_validation=bool(feasible),
                validation_target=target, validation_control=control))
    write_json(out / 'validation_sweep.json', measured)
    write_json(out / 'selected.json', selected)
    print(json.dumps(selected, indent=2), flush=True)
    stop.set()


def evaluate(out):
    torch, stop = setup(out, 'test')
    sys.path.insert(0, str(ROOT / 'src'))
    from clic27.eval import metrics as M
    metric = M.Perceptual('cuda')
    records = [r for r in json.loads((out / 'manifest.json').read_text())['records'] if r['split']=='test']
    methods = {'base':dict(kind='blend', alpha=0.), 'perceptual':dict(kind='blend', alpha=1.)}
    for row in json.loads((out / 'selected.json').read_text()):
        methods[row['name']] = row['method']
        methods[row['name']+'-matched-blend'] = row['control']
    all_rows, summary = [], []
    for name, method in methods.items():
        folder = out / 'recon' / name
        folder.mkdir(parents=True)
        rows = []
        for record in records:
            value = images(out, 'test', record['name'])
            reconstructed = apply(value, method)
            Image.fromarray(reconstructed).save(folder / (record['name']+'.png'))
            with torch.no_grad():
                metrics = metric.all(value['original'], reconstructed)
            rows.append(dict(method=name, name=record['name'], bytes=record['bytes'], pixels=record['pixels'],
                edge=edge_error(value['original'], reconstructed), pixel_sha256=hashlib.sha256(reconstructed.tobytes()).hexdigest(), **metrics))
        all_rows.extend(rows)
        result = dict(method=name, n=len(rows), actual_bpp=8*sum(r['bytes'] for r in rows)/sum(r['pixels'] for r in rows),
            psnr_pooled=M.pooled_psnr([r['mse'] for r in rows], [r['pixels'] for r in rows]),
            **{key:float(np.mean([r[key] for r in rows])) for key in ['lpips', 'dists', 'edge', 'msssim']})
        summary.append(result)
        write_json(out / 'test.progress.json', result)
        print(json.dumps(result), flush=True)
    write_json(out / 'test_metrics.json', all_rows)
    write_json(out / 'summary.json', summary)
    M.write_csv(summary, out / 'summary.csv')
    stop.set()


def verify(out):
    torch, stop = setup(out, 'fresh')
    model, _ = load(arguments(out), 2)
    adapter = adapter_model(torch)
    expected = {(r['method'], r['name']):r['pixel_sha256'] for r in json.loads((out / 'test_metrics.json').read_text())}
    methods = {'base':dict(kind='blend', alpha=0.), 'perceptual':dict(kind='blend', alpha=1.)}
    for row in json.loads((out / 'selected.json').read_text()):
        methods[row['name']] = row['method']
        methods[row['name']+'-matched-blend'] = row['control']
    matches = 0
    for path in sorted((out / 'test/streams').glob('*.bin')):
        q, legacy, size, shape, strings = unpack(path.read_bytes())
        if q != 2 or not legacy or size != (256,256):
            raise ValueError('Wrong stream')
        value = {}
        for mode in ['base', 'perceptual']:
            hook = None
            if mode == 'perceptual':
                hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda module, args:(adapter(args[0])[0],))
            model.update_resolutions(16,16)
            with torch.no_grad():
                value[mode] = uint8(model.decompress(strings, shape)['x_hat'])
            if hook is not None:
                hook.remove()
        for name, method in methods.items():
            result = apply(value, method)
            if hashlib.sha256(result.tobytes()).hexdigest() != expected[name,path.stem]:
                raise ValueError(f'Fresh pixel mismatch: {name}/{path.stem}')
            matches += 1
    if matches != len(expected):
        raise ValueError('Incomplete fresh parity')
    write_json(out / 'fresh.json', dict(matches=matches, fresh_process=True, source_images_not_used=True))
    stop.set()


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    frozen = {name:sha(Path(__file__).with_name(name)) for name in ['projection_probe.py','hard_constraints.py','probe.py','pilot.py','pilot_model.py','stream.py']}
    write_json(out / 'sources.sha256.json', frozen)
    for stage in ['prepare','select','evaluate','verify']:
        write_json(out / 'status.json', dict(state='running',stage=stage))
        with (out / (stage+'.log')).open('w') as log:
            code = subprocess.call([sys.executable,__file__,'--out',str(out),'--stage',stage],stdout=log,stderr=subprocess.STDOUT)
        if code:
            write_json(out / 'status.json',dict(state='failed',stage=stage,exit_code=code))
            raise RuntimeError(f'{stage} failed: {code}')
    if any(sha(Path(__file__).with_name(name))!=value for name,value in frozen.items()):
        raise ValueError('Source changed during run')
    write_json(out / 'status.json',dict(state='complete',fresh_pixel_parity=True,test_images=64))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['run','prepare','select','evaluate','verify'],required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out)
