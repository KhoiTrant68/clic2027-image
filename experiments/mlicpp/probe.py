"""Pinned official MLIC++ inference: real streams, fresh decode and shared Qhat metrics."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from stream import pack, unpack

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = '4aa9d2a512eb9db382172058e6c278fb0d376e17'
LAMBDAS = {1: .0018, 2: .0035, 3: .0067, 4: .0130}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def verify_parity(out, q, names):
    encoder = json.loads((out / f'q{q}-encode.json').read_text())['images']
    decoder = json.loads((out / f'q{q}-decode.json').read_text())['images']
    direct, fresh = {r['name']: r for r in encoder}, {r['name']: r for r in decoder}
    if len(direct) != len(encoder) or len(fresh) != len(decoder) or set(direct) != set(names) or set(fresh) != set(names):
        raise ValueError(f'q{q}: incomplete/duplicate image inventory')
    if any(direct[n][key] != fresh[n][key] for n in direct for key in ('pixel_sha256', 'stream_sha256')):
        raise ValueError(f'q{q}: fresh-process pixel/stream parity failed')


def settings():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))
    from half_vram import install
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch, install


def load(a, q):
    import torch
    revision = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError(f'Unexpected upstream revision: {revision}')
    sys.path.insert(0, str(a.upstream / 'MLIC++'))
    from config.config import model_config
    import models.mlicpp as mlic
    # The authors explicitly require these classes for the released 2023 weights.
    if a.legacy:
        mlic.SynthesisTransform = mlic.SynthesisTransformOld
        mlic.LatentResidualPrediction = mlic.LatentResidualPredictionOld
    checkpoint = a.weights / f'mlicpp-mse-q{q}-legacy.pth.tar'
    if not a.legacy:
        raise ValueError('This experiment uses the official legacy q1-q4 grid; pass --legacy')
    started = time.perf_counter()
    checkpoint_hash = sha(checkpoint)
    # Official training archives include collections.Counter, unsupported by
    # torch 2.0's weights-only loader. Load only the pinned authors' downloads.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k, v in ck['state_dict'].items()}
    if len(state) != len(ck['state_dict']):
        raise ValueError('Duplicate state keys after stripping module prefix')
    model = mlic.MLICPlusPlus(model_config())
    # Upstream load_state_dict delegates to strict=True; mismatches must raise.
    model.load_state_dict(state)
    del ck, state
    device = getattr(a, 'device', 'cuda')
    model.eval().to(device)
    model.update(force=False)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    meta = dict(quality=q, lambda_mse=LAMBDAS[q], legacy=a.legacy, checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash, upstream_sha=revision,
                cudnn_v8_api_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0'),
                parameters=sum(p.numel() for p in model.parameters()), load_s=time.perf_counter()-started)
    return model, meta


def gpu_stage(a):
    torch, install = settings()
    stop = install(a.out / f'q{a.quality}-{a.stage}.vram.jsonl')
    import numpy as np
    from PIL import Image
    qdir = a.out / f'q{a.quality}'
    start = time.perf_counter()
    model, meta = load(a, a.quality)
    if getattr(a, 'synthesis', None):
        # GC backbone ablation: same streams, decoder g_s replaced (experiments/gc_backbone/train_gs.py).
        if a.stage != 'decode':
            raise ValueError('--synthesis only applies to the decode stage')
        ck = torch.load(a.synthesis, map_location='cpu')
        if ck['quality'] != a.quality:
            raise ValueError('synthesis trained for another quality')
        model.g_s.load_state_dict(ck['g_s'])
        model.g_s.cuda().eval()
        meta.update(synthesis=str(a.synthesis), synthesis_sha256=sha(a.synthesis))
    records = []
    if a.stage == 'encode':
        (qdir / 'streams').mkdir(parents=True, exist_ok=False)
        (qdir / 'direct').mkdir()
        files = sorted(a.data.glob('*.png'))
        if a.limit:
            files = files[:a.limit]
        if not files:
            raise ValueError('No input images')
    else:
        (qdir / 'recon').mkdir(exist_ok=False)
        files = sorted((qdir / 'streams').glob('*.bin'))
        if not files:
            raise ValueError('No saved streams')
    for path in files:
        torch.cuda.reset_peak_memory_stats()
        if a.stage == 'encode':
            with Image.open(path) as im:
                pixels = np.array(im.convert('RGB'))
            h, w = pixels.shape[:2]
            x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
            x = torch.nn.functional.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode='constant', value=0)
            model.update_resolutions(16, 16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                encoded = model.compress(x)
            torch.cuda.synchronize()
            encode_s = time.perf_counter() - t0
            blob = pack(a.quality, a.legacy, (h, w), tuple(encoded['shape']), encoded['strings'])
            stream = qdir / 'streams' / (path.stem + '.bin')
            stream.write_bytes(blob)
            del x, encoded
        else:
            stream = path
            blob = stream.read_bytes()
        q, legacy, (h, w), shape, strings = unpack(stream.read_bytes())
        if q != a.quality or legacy != a.legacy:
            raise ValueError('Stream architecture/quality differs from loaded model')
        # Match official testing: clear resolution caches to prevent encoder state leakage.
        model.update_resolutions(16, 16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            decoded = model.decompress(strings, shape)['x_hat'][:, :, :h, :w]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
        if tuple(decoded.shape) != (1, 3, h, w) or not torch.isfinite(decoded).all():
            raise ValueError(f'Invalid decoded tensor: image={path.stem}, shape={tuple(decoded.shape)}, '
                             f'expected={(1, 3, h, w)}, nonfinite={int((~torch.isfinite(decoded)).sum())}')
        t0 = time.perf_counter()
        pixels = decoded[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(pixels).save(qdir / ('direct' if a.stage == 'encode' else 'recon') / (path.stem + '.png'),
                                     compress_level=0)
        record = dict(name=path.stem, quality=q, width=w, height=h, bytes=len(blob), bpp=8*len(blob)/(h*w),
                      stream_sha256=sha(stream), pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                      decode_s=decode_s, write_s=time.perf_counter()-t0,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if a.stage == 'encode':
            record['encode_s'] = encode_s
        records.append(record)
        write_json(a.out / f'q{a.quality}-{a.stage}.progress.json', dict(completed=len(records), total=len(files), last=record))
        print(json.dumps(record), flush=True)
        del decoded
    meta.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0), fp32=True, deterministic=True, tf32=False,
                png_compress_level=0, network='checkpoint local; network not isolated by OS',
                process_inside_wall_s=time.perf_counter()-start, images=records)
    write_json(a.out / f'q{a.quality}-{a.stage}.json', meta)
    stop.set()


def score(a):
    torch, install = settings()
    stop = install(a.out / 'metrics.vram.jsonl')
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'experiments/qhat'))
    sys.path.insert(0, str(ROOT / 'experiments/bakeoff'))
    from qhat import Features
    from bakeoff import qhat_score
    from clic27.eval import metrics as M
    manifest = json.loads((a.out / 'manifest.json').read_text())
    if sha(a.qhat) != manifest['qhat_sha256']:
        raise ValueError('Qhat file changed after encoding')
    model = json.loads(a.qhat.read_text())
    feat = Features('cuda')
    if set(model['features']) - set(feat.names()):
        raise ValueError('Qhat features unavailable; refusing to substitute a different objective')
    existing = M.read_csv(a.out / 'metrics.csv') if (a.out / 'metrics.csv').exists() else []
    done = {(int(r['quality']), r['name']) for r in existing}
    for q in a.qualities:
        fresh = json.loads((a.out / f'q{q}-decode.json').read_text())['images']
        for row in fresh:
            if (q, row['name']) in done:
                continue
            x = M.load(a.data / (row['name'] + '.png'))
            y = M.load(a.out / f'q{q}/recon' / (row['name'] + '.png'))
            values = feat(x, y, only=model['features'])
            values['mse'] = M.mse(x, y)
            values['qhat_v0'] = qhat_score(model, values)
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Non-finite metric')
            record = dict(quality=q, name=row['name'], bytes=row['bytes'], pixels=row['width']*row['height'],
                          bpp=row['bpp'], decode_s=row['decode_s'], **values)
            existing.append(record)
            tmp = a.out / 'metrics.csv.tmp'
            with tmp.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(existing)
            tmp.replace(a.out / 'metrics.csv')
            print(json.dumps(record), flush=True)
    names = [r['name'] for r in manifest['images']]
    complete = {(int(r['quality']), r['name']) for r in existing} == {(q, n) for q in manifest['qualities'] for n in names}
    if complete:
        for q in manifest['qualities']:
            verify_parity(a.out, q, names)
        write_json(a.out / 'status.json', dict(state='complete', fresh_pixel_parity=True,
                                             images=len(names), qualities=manifest['qualities'], metrics=True))
    else:
        write_json(a.out / 'status.json', dict(state='partial', metrics=False, metric_rows=len(existing)))
    stop.set()


def run(a):
    from PIL import Image
    a.out.mkdir(parents=True, exist_ok=False)
    checkpoints = [a.weights / f'mlicpp-mse-q{q}-legacy.pth.tar' for q in a.qualities]
    if any(not p.is_file() for p in checkpoints):
        raise ValueError('Missing requested checkpoints')
    files = sorted(a.data.glob('*.png'))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise ValueError('No validation data')
    records = []
    for path in files:
        with Image.open(path) as image:
            records.append(dict(name=path.stem, width=image.width, height=image.height, sha256=sha(path)))
    write_json(a.out / 'manifest.json', dict(images=records, qualities=a.qualities, upstream_sha=UPSTREAM,
        runner_sha256=sha(__file__), framing_sha256=sha(Path(__file__).with_name('stream.py')),
        environment_sha256=sha(Path(sys.prefix) / 'requirements.freeze.txt'),
        cudnn_v8_api_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0'),
        qhat_sha256=sha(a.qhat), seed=0, architecture='legacy', lambda_mse=LAMBDAS,
        experiment='pretrained inference only; not training or paper replication',
        pixel_quantization='clamp [0,1], multiply 255, round to uint8',
        padding='constant zero at bottom/right to multiple of 64'))
    base = [sys.executable, str(Path(__file__).resolve()), '--upstream', str(a.upstream), '--weights', str(a.weights),
            '--out', str(a.out), '--legacy']
    times = {}
    for q in a.qualities:
        for stage in ('encode', 'decode'):
            cmd = base + ['--stage', stage, '--quality', str(q)]
            if stage == 'encode':
                cmd += ['--data', str(a.data)]
                if a.limit:
                    cmd += ['--limit', str(a.limit)]
            write_json(a.out / 'status.json', dict(state='running', quality=q, stage=stage, command=cmd))
            t0 = time.perf_counter()
            with (a.out / f'q{q}-{stage}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            times[f'q{q}-{stage}'] = time.perf_counter()-t0
            write_json(a.out / 'process_times.json', times)
        verify_parity(a.out, q, [r['name'] for r in records])
    if not a.skip_metrics:
        cmd = base + ['--stage', 'score', '--data', str(a.data), '--qhat', str(a.qhat),
                      '--qualities'] + list(map(str, a.qualities))
        write_json(a.out / 'status.json', dict(state='running', stage='score', command=cmd))
        with (a.out / 'metrics.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(a.out / 'status.json', dict(state='complete', fresh_pixel_parity=True,
                                         images=len(files), qualities=a.qualities, metrics=not a.skip_metrics))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['run', 'encode', 'decode', 'score'], default='run')
    for name in ('upstream', 'weights', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--qhat', type=Path, default=ROOT/'results/2026-10-15_new_codecs/qhat_v0.json')
    p.add_argument('--qualities', type=int, nargs='+', choices=range(1, 5), default=[1, 2, 3, 4])
    p.add_argument('--quality', type=int, choices=range(1, 5))
    p.add_argument('--legacy', action='store_true', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--skip-metrics', action='store_true')
    p.add_argument('--synthesis', type=Path, help='decode only: g_s weights from gc_backbone/train_gs.py')
    a = p.parse_args()
    if a.limit is not None and a.limit <= 0:
        p.error('--limit must be positive')
    if len(set(a.qualities)) != len(a.qualities):
        p.error('--qualities must not contain duplicates')
    try:
        if a.stage in ('encode', 'decode'):
            if a.quality is None:
                p.error('--quality required')
            gpu_stage(a)
        elif a.stage == 'score':
            score(a)
        else:
            run(a)
    except Exception as exc:
        if a.out.is_dir():
            write_json(a.out/'failure.json', dict(stage=a.stage, error=repr(exc)))
            status_path = a.out / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(state='failed', failed_stage=a.stage, error=repr(exc))
            write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
