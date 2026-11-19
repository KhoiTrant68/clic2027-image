"""One-checkpoint decoder pilot: actual streams, three matched training arms, held-out test."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from probe import load, settings, sha, write_json, ROOT
from stream import pack, unpack


def config(path):
    raw = json.loads(path.read_text())
    for key in ('data', 'cache', 'out', 'upstream', 'weights'):
        raw[key] = Path(raw[key])
    return SimpleNamespace(**raw)


def setup(a, label):
    torch, guard = settings()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = guard(a.out / f'{label}.vram.jsonl')
    torch.manual_seed(a.seed)
    write_json(a.out / f'{label}.runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), seed=a.seed,
        cudnn_v8_api_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED'),
        fp32=True, tf32=False, deterministic=True, threads=torch.get_num_threads()))
    return torch, stop


def make_tail(a, torch):
    sys.path.insert(0, str(a.upstream / 'MLIC++'))
    from modules.transform import SynthesisTransformOld
    tail = SynthesisTransformOld(N=192, M=320).synthesis_transform[5:]
    tail.load_state_dict(torch.load(a.cache / 'tail.pt', map_location='cpu', weights_only=True))
    return tail.cuda().eval().requires_grad_(False)


def prepare(a):
    torch, stop = setup(a, 'prepare')
    import numpy as np
    from PIL import Image
    from pilot_model import edge_regions
    manifest = json.loads((a.data / 'manifest.json').read_text())
    names = [r['name'] for r in manifest['images']]
    hashes = [r['sha256'] for r in manifest['images']]
    if len(set(names)) != len(names) or len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate source images; refuse leakage across splits')
    a.cache.mkdir(parents=True, exist_ok=False)
    model, meta = load(a, a.quality)
    model.requires_grad_(False)
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach().cpu().clone())
    hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(capture)
    torch.save(model.g_s.synthesis_transform[5:].cpu().state_dict(), a.cache / 'tail.pt')
    model.cuda()
    records, train_edges = [], []
    started = time.perf_counter()
    for source in manifest['images']:
        path = a.data / 'images' / source['name']
        if sha(path) != source['sha256']:
            raise ValueError('Source hash changed')
        with Image.open(path) as image:
            image = image.convert('RGB')
            count = a.train_crops_per_image if source['split'] == 'train' else a.eval_crops_per_image
            for index in range(count):
                rng = random.Random(f'{a.seed}/{path.stem}/{index}')
                x0, y0 = rng.randrange(image.width-a.crop+1), rng.randrange(image.height-a.crop+1)
                pixels = np.array(image.crop((x0, y0, x0+a.crop, y0+a.crop)))
                x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().cuda()/255
                name = f'{path.stem}-{index}'
                folder = a.cache / source['split']
                folder.mkdir(exist_ok=True)
                model.update_resolutions(16, 16)
                with torch.no_grad():
                    encoded = model.compress(x)
                    blob = pack(a.quality, True, (a.crop, a.crop), tuple(encoded['shape']), encoded['strings'])
                    model.update_resolutions(16, 16)
                    captured.clear()
                    base = model.decompress(encoded['strings'], encoded['shape'])['x_hat'].clamp(0, 1)
                    if len(captured) != 1 or not torch.isfinite(base).all():
                        raise ValueError(f'Invalid cache decode: {name}')
                    feature = captured[0]
                    base_byte = base.mul(255).round().byte().cpu()
                    original = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None]
                    if source['split'] == 'train':
                        train_edges.extend(edge_regions(base, x).flatten().cpu().tolist())
                torch.save(dict(feature=feature, original=original, base=base_byte), folder / f'{name}.pt')
                (folder / f'{name}.bin').write_bytes(blob)
                record = dict(name=name, source=source['name'], split=source['split'], x=x0, y=y0,
                              bytes=len(blob), pixels=a.crop*a.crop, stream_sha256=hashlib.sha256(blob).hexdigest(),
                              feature_sha256=hashlib.sha256(feature.numpy().tobytes()).hexdigest())
                records.append(record)
                write_json(a.out / 'prepare.progress.json', dict(completed=len(records), last=record,
                           seconds=time.perf_counter()-started))
                print(json.dumps(dict(stage='cache', **record)), flush=True)
    hook.remove()
    scale = max(float(np.quantile(train_edges, .95)), 1e-5)
    meta.update(data_manifest_sha256=sha(a.data / 'manifest.json'), records=records, edge_scale=scale,
                initial_tau=min(float(np.quantile(train_edges, a.alpha))/scale, 1.),
                feature_dtype='float32', feature_position='before legacy synthesis_transform[5]',
                checkpoint_warning='fixed legacy q2; actual bpp must be measured, not assumed 0.15')
    write_json(a.cache / 'manifest.json', meta)
    stop.set()


def cached(a, row, torch):
    value = torch.load(a.cache / row['split'] / (row['name']+'.pt'), weights_only=True)
    return value['feature'].cuda(), value['original'].cuda().float()/255, value['base'].cuda().float()/255


def train(a, mode):
    torch, stop = setup(a, mode)
    import lpips
    from pilot_model import Adapter, cvar_objective, edge_regions, empirical_cvar
    from torch.nn import functional as F
    manifest = json.loads((a.cache / 'manifest.json').read_text())
    train_rows = [r for r in manifest['records'] if r['split'] == 'train']
    val_rows = [r for r in manifest['records'] if r['split'] == 'validation']
    tail = make_tail(a, torch)
    torch.manual_seed(a.seed)
    adapter = Adapter(hidden=a.hidden, amplitude=a.amplitude).cuda()
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    tau = torch.nn.Parameter(torch.tensor(manifest['initial_tau'], device='cuda'))
    opt = torch.optim.Adam([{'params': adapter.parameters()}, {'params': [tau], 'lr': a.tau_learning_rate}],
                           lr=a.learning_rate, betas=tuple(a.adam_betas))
    folder = a.out / mode
    folder.mkdir(exist_ok=False)
    write_json(folder / 'config.json', dict(mode=mode, parameters=sum(p.numel() for p in adapter.parameters()),
                seed=a.seed, steps=a.steps, data_order='Python Random(seed) replacement; shared across arms',
                checkpoint_selection='fixed final step; validation logging only',
                architecture='identical in all three arms; CVaR/head supervision differ',
                reference=a.reference, intentional_changes=a.intentional_changes))
    rng = random.Random(a.seed)
    started = time.perf_counter()
    with (folder / 'train.jsonl').open('w', buffering=1) as log:
        for step in range(1, a.steps+1):
            batch = [cached(a, rng.choice(train_rows), torch) for _ in range(a.batch_size)]
            features, target, base = [torch.cat([v[i] for v in batch]) for i in range(3)]
            opt.zero_grad(set_to_none=True)
            corrected, delta, logits = adapter(features)
            output = tail(corrected).clamp(0, 1)
            mse = (output-target).square().mean()
            lp = perceptual(output*2-1, target*2-1).mean()
            errors = edge_regions(output, target, manifest['edge_scale'])
            loss = a.pixel_weight*mse + a.lpips_weight*lp + a.edge_weight*errors.mean()
            tail_loss = cvar_objective(errors, tau, a.alpha)
            head_loss = output.new_tensor(0.)
            if mode in ('cvar', 'risk'):
                loss = loss + a.tail_weight*tail_loss
            if mode == 'risk':
                with torch.no_grad():
                    candidate = tail(features+delta.detach()).clamp(0, 1)
                    proposal_harm = edge_regions(candidate, target, manifest['edge_scale'])
                    base_raw = tail(features).clamp(0, 1)
                    base_harm = edge_regions(base_raw, target, manifest['edge_scale'])
                    label = (proposal_harm-base_harm > a.harm_margin).float().reshape(-1, 1, 2, 2)
                head_loss = F.binary_cross_entropy_with_logits(logits, label)
                loss = loss + a.head_weight*head_loss
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            opt.step()
            with torch.no_grad():
                tau.clamp_(0, 1)
            if step == 1 or step % 10 == 0:
                row = dict(step=step, loss=loss.item(), mse=mse.item(), lpips=lp.item(),
                           edge=errors.mean().item(), cvar_variational=tail_loss.item(), tau=tau.item(),
                           head_loss=head_loss.item(), seconds=time.perf_counter()-started,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(row)+'\n')
                write_json(folder / 'progress.json', row)
                if step == 1 or step % 100 == 0:
                    print(json.dumps(dict(mode=mode, **row)), flush=True)
            if step % a.eval_every == 0:
                adapter.eval()
                values, edges = [], []
                with torch.no_grad():
                    for row in val_rows:
                        f, x, _ = cached(a, row, torch)
                        y = tail(adapter(f)[0]).clamp(0, 1)
                        values.append(float(perceptual(y*2-1, x*2-1).mean()))
                        edges.append(edge_regions(y, x, manifest['edge_scale']).cpu())
                with (folder / 'validation.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(step=step, lpips=sum(values)/len(values),
                        edge_cvar=float(empirical_cvar(torch.cat(edges), a.alpha))))+'\n')
                adapter.train()
                torch.save(dict(adapter=adapter.state_dict(), optimizer=opt.state_dict(), step=step,
                           tau=tau.detach().cpu(), mode=mode), folder / 'last.pt.tmp')
                (folder / 'last.pt.tmp').replace(folder / 'last.pt')
    write_json(folder / 'status.json', dict(state='complete', steps=a.steps, seconds=time.perf_counter()-started))
    stop.set()


def evaluate(a):
    torch, stop = setup(a, 'evaluation')
    import numpy as np
    from PIL import Image
    from pilot_model import Adapter, edge_regions, empirical_cvar
    sys.path.insert(0, str(ROOT / 'src'))
    from clic27.eval import metrics as M
    manifest = json.loads((a.cache / 'manifest.json').read_text())
    tail = make_tail(a, torch)
    metric = M.Perceptual('cuda')
    records = [r for r in manifest['records'] if r['split'] == 'test']
    outputs, summaries = [], []
    for mode in ['base'] + a.modes + ['risk_hide_change']:
        adapter = None
        if mode != 'base':
            adapter = Adapter(hidden=a.hidden, amplitude=a.amplitude).cuda().eval()
            arm = 'risk' if mode == 'risk_hide_change' else mode
            ck = torch.load(a.out / arm / 'last.pt', map_location='cpu', weights_only=True)
            if ck['step'] != a.steps:
                raise ValueError('Not the planned final checkpoint')
            adapter.load_state_dict(ck['adapter'])
        all_edges, method_rows = [], []
        for row in records:
            f, x, base = cached(a, row, torch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                y = tail(f if adapter is None else adapter(f, hide_change=mode=='risk_hide_change')[0]).clamp(0, 1)
                pixels = y.mul(255).round().byte()
                torch.cuda.synchronize()
                suffix_s = time.perf_counter()-started
                y = pixels.float()/255
                edges = edge_regions(y, x, manifest['edge_scale'])[0].cpu()
            # Shared metric API requires uint8 [0,255], not normalized floats.
            orig = x[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            recon = pixels[0].permute(1, 2, 0).cpu().numpy()
            with torch.no_grad():
                values = metric.all(orig, recon)
            folder = a.out / 'recon' / mode
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(recon).save(folder / (row['name']+'.png'))
            entry = dict(method=mode, name=row['name'], source=row['source'], bytes=row['bytes'],
                         pixels=row['pixels'], bpp=row['bytes']*8/row['pixels'], suffix_s=suffix_s,
                         edge_mean=float(edges.mean()), edge_regions=edges.tolist(),
                         pixel_sha256=hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest(), **values)
            all_edges.append(edges)
            method_rows.append(entry)
        outputs.extend(method_rows)
        summary = dict(method=mode, n=len(method_rows), bpp=sum(r['bytes'] for r in method_rows)*8/sum(r['pixels'] for r in method_rows),
            psnr_pooled=M.pooled_psnr([r['mse'] for r in method_rows], [r['pixels'] for r in method_rows]),
            edge_cvar=float(empirical_cvar(torch.cat(all_edges), a.alpha)),
            **{key: sum(r[key] for r in method_rows)/len(method_rows) for key in ['lpips', 'dists', 'msssim', 'edge_mean', 'suffix_s']})
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(a.out / 'test_metrics.json', outputs)
    write_json(a.out / 'summary.json', summaries)
    M.write_csv(summaries, a.out / 'summary.csv')
    stop.set()


def verify(a, mode):
    """Re-decode saved test streams in a new process, with adapter inside g_s."""
    torch, stop = setup(a, f'fresh-{mode}')
    from pilot_model import Adapter
    expected = {r['name']: r['pixel_sha256'] for r in json.loads((a.out / 'test_metrics.json').read_text()) if r['method'] == mode}
    model, meta = load(a, a.quality)
    adapter = Adapter(hidden=a.hidden, amplitude=a.amplitude).cuda().eval()
    adapter.load_state_dict(torch.load(a.out / mode / 'last.pt', map_location='cpu', weights_only=True)['adapter'])
    def apply(module, args):
        return (adapter(args[0])[0],)
    hook = model.g_s.synthesis_transform[5].register_forward_pre_hook(apply)
    result = []
    for name, pixel_hash in expected.items():
        blob = (a.cache / 'test' / (name+'.bin')).read_bytes()
        q, legacy, (h, w), shape, strings = unpack(blob)
        if q != a.quality or not legacy:
            raise ValueError('Wrong stream')
        model.update_resolutions(16, 16)
        with torch.no_grad():
            pixels = model.decompress(strings, shape)['x_hat'][:, :, :h, :w].clamp(0, 1).mul(255).round().byte()
        got = hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest()
        if got != pixel_hash:
            raise ValueError(f'Fresh decoder mismatch: {mode}/{name}')
        result.append(dict(name=name, pixel_sha256=got))
    hook.remove()
    meta.update(mode=mode, matches=len(result), images=result)
    write_json(a.out / f'fresh-{mode}.json', meta)
    stop.set()


def run(a, config_path):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out / 'planned_config.json', json.loads(config_path.read_text()))
    sources = {p.name: sha(p) for p in Path(__file__).parent.glob('pilot*') if p.is_file()}
    sources['fetch_pilot_data.py'] = sha(Path(__file__).with_name('fetch_pilot_data.py'))
    write_json(a.out / 'sources.sha256.json', sources)
    write_json(a.out / 'dependencies.sha256.json', {
        'data_manifest': sha(a.data / 'manifest.json'),
        'metric_module': sha(ROOT / 'src/clic27/eval/metrics.py'),
        'environment_freeze': sha(Path(sys.prefix) / 'requirements.freeze.txt'),
        'base_runner': sha(Path(__file__).with_name('probe.py')),
        'stream': sha(Path(__file__).with_name('stream.py'))})
    stages = [('prepare', None)] + [('train', mode) for mode in a.modes] + [('evaluate', None)] + [('verify', mode) for mode in a.modes]
    for stage, mode in stages:
        write_json(a.out / 'status.json', dict(state='running', stage=stage, mode=mode))
        cmd = [sys.executable, __file__, '--config', str(config_path), '--stage', stage]
        if mode:
            cmd += ['--mode', mode]
        label = stage+'-'+(mode or 'all')
        with (a.out / f'{label}.log').open('w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if code:
            write_json(a.out / 'status.json', dict(state='failed', stage=stage, mode=mode, exit_code=code))
            raise RuntimeError(f'{label} exited {code}; inspect its preserved log')
    if any(sha(Path(__file__).with_name(name)) != digest for name, digest in sources.items()):
        raise ValueError('Pilot sources changed during experiment')
    write_json(a.out / 'status.json', dict(state='complete', fresh_pixel_parity=True, modes=a.modes,
                warning='crop256, fixed legacy q2, single seed; no GAN, OCR/human study or L4 test'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['run', 'prepare', 'train', 'evaluate', 'verify'], required=True)
    p.add_argument('--mode', choices=['mean', 'cvar', 'risk'])
    args = p.parse_args()
    a = config(args.config)
    if args.stage in ('train', 'verify') and args.mode is None:
        p.error('mode is required')
    if args.stage == 'run':
        run(a, args.config)
    elif args.stage == 'train':
        train(a, args.mode)
    elif args.stage == 'verify':
        verify(a, args.mode)
    else:
        globals()[args.stage](a)


if __name__ == '__main__':
    main()
