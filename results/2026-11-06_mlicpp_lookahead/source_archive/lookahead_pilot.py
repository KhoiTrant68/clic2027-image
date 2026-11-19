"""Development counterfactuals: one integer edit, native suffix and decoder.

No critic training. The diagnostic comparator sees final distortion but omits
future CDF-rate changes; it is not an equal-time practical search baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from probe import settings, load, write_json, sha
from stream import pack, unpack


def config(out):
    return SimpleNamespace(**json.loads((out / 'config.json').read_text()))


def setup(out):
    torch, _ = settings()
    a = config(out)
    assert a.device == 'cpu', 'This bounded pilot is CPU-only'
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    # Upstream uses CUDA synchronization only for timers even on CPU tensors.
    # This isolated subprocess never allocates CUDA tensors.
    torch.cuda.synchronize = lambda *args, **kwargs: None
    model, meta = load(SimpleNamespace(upstream=Path(a.upstream),
                                      weights=Path(a.weights), legacy=True, device='cpu'), 2)
    model.requires_grad_(False)
    assert next(model.parameters()).device.type == 'cpu'
    return torch, model, meta


class Intervention:
    def __init__(self, model):
        import models.mlicpp as module
        import utils.ckbd as ckbd
        self.module, self.ckbd, self.model = module, ckbd, model
        self.original = {n: getattr(module, n) for n in ['compress_anchor', 'compress_nonanchor']}
        for name in self.original:
            setattr(module, name, self.wrapper(name))
        self.start(None)

    def start(self, action):
        self.action, self.cursor, self.trace = action, 0, []

    def wrapper(self, name):
        kind = 'nonanchor' if name.endswith('nonanchor') else 'anchor'
        squeeze = getattr(self.ckbd, 'ckbd_' + kind + '_sequeeze')
        unsqueeze = getattr(self.ckbd, 'ckbd_' + kind + '_unsequeeze')

        def call(gc, y, scale, mean, symbols, indexes):
            step = self.cursor
            self.cursor += 1
            y, scale, mean = squeeze(y), squeeze(scale), squeeze(mean)
            idx = gc.build_indexes(scale)
            q = gc.quantize(y, 'symbols', mean)
            residual = (y - mean - q).detach().reshape(-1).numpy().copy()
            if self.action is not None and step == self.action['step']:
                q = q.clone()
                for position, delta in zip(self.action['positions'], self.action['deltas']):
                    assert delta in (-1, 1) and 0 <= position < q.numel()
                    q.reshape(-1)[position] += delta
            si, ii = q.reshape(-1).tolist(), idx.reshape(-1).tolist()
            symbols.extend(si)
            indexes.extend(ii)
            cdf = gc.quantized_cdf
            lengths = gc.cdf_length
            offset = gc.offset
            k = q.reshape(-1).long() - offset[idx.reshape(-1).long()].long()
            ix = idx.reshape(-1).long()
            valid = (k >= 0) & (k < lengths[ix].long() - 2)
            # Do not approximate escape symbols as ordinary CDF entries.
            bits = None
            if bool(valid.all()):
                count = (cdf[ix, k + 1] - cdf[ix, k]).double()
                bits = float((-count.log2() + 16).sum())
            self.trace.append(dict(step=step, symbols=si, indexes=ii, residual=residual,
                                   cdf_bits=bits, escapes=int((~valid).sum())))
            return unsqueeze(q + mean)
        return call

    def close(self):
        for name, fn in self.original.items():
            setattr(self.module, name, fn)


def pixels(t):
    return t[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()


def encode(model, hook, x, action):
    hook.start(action)
    start = time.monotonic()
    encoded = model.compress(x)
    trace = hook.trace
    assert len(trace) == 20
    blob = pack(2, True, tuple(x.shape[-2:]), tuple(encoded['shape']), encoded['strings'])
    image = pixels(model.decompress(encoded['strings'], encoded['shape'])['x_hat'])
    return blob, image, trace, time.monotonic() - start


def candidates(trace, a):
    result = []
    for step in a.steps:
        residual = trace[step]['residual']
        # Stable tie-breaking, source-only rule, never uses final RD to select sites.
        positions = np.argsort(-np.abs(residual), kind='stable')[:a.block_sites].tolist()
        for position in positions[:a.sites_per_step]:
            for delta in (-1, 1):
                result.append(dict(step=step, positions=[position], deltas=[delta]))
        result.append(dict(step=step, positions=positions,
                           deltas=[1 if residual[p] >= 0 else -1 for p in positions]))
    return result


def search(out):
    torch, model, meta = setup(out)
    a = config(out)
    manifest = json.loads((Path(a.reference_run) / 'manifest.json').read_text())['records']
    selected, seen = [], set()
    for r in manifest:
        if r['split'] == 'validation' and r['source'] not in seen and len(selected) < a.sources:
            selected.append(r)
            seen.add(r['source'])
    assert len(selected) == a.sources
    write_json(out / 'inventory.json', dict(records=selected, role='reused development, not holdout',
                                           geometry_test_opened=False))
    write_json(out / 'backbone.json', meta)
    (out / 'streams').mkdir()
    hook = Intervention(model)
    results = []
    with torch.no_grad(), (out / 'progress.jsonl').open('w', buffering=1) as log:
        for record in selected:
            cache = Path(a.reference_run) / 'cache' / (record['name'] + '.pt')
            v = torch.load(cache, map_location='cpu', weights_only=True)
            x = v['original'].float() / 255
            target = v['original'][0].permute(1, 2, 0).numpy().astype(np.float64)
            n = int(x.shape[-1] * x.shape[-2])
            base_blob, base_image, base_trace, base_time = encode(model, hook, x, None)
            # Native untouched encoder must match the instrumentation exactly.
            hook.close()
            orig = model.compress(x)
            assert base_blob == pack(2, True, tuple(x.shape[-2:]), tuple(orig['shape']), orig['strings'])
            assert np.array_equal(base_image, pixels(model.decompress(orig['strings'], orig['shape'])['x_hat']))
            hook = Intervention(model)
            actions = [None] + candidates(base_trace, a)
            for i, action in enumerate(actions):
                if action is None:
                    blob, image, trace, elapsed = base_blob, base_image, base_trace, base_time
                else:
                    blob, image, trace, elapsed = encode(model, hook, x, action)
                    step = action['step']
                    for k in range(step):
                        assert trace[k]['symbols'] == base_trace[k]['symbols']
                        assert trace[k]['indexes'] == base_trace[k]['indexes']
                    assert trace[step]['indexes'] == base_trace[step]['indexes']
                    expected = np.array(base_trace[step]['symbols'])
                    expected[action['positions']] += action['deltas']
                    assert np.array_equal(expected, trace[step]['symbols'])
                    assert unpack(blob)[4][1] == unpack(base_blob)[4][1], 'Hyperstream changed'
                mse = float(((image.astype(np.float64) - target) ** 2).mean())
                bpp = len(blob) * 8 / n
                current_delta, future_delta = 0., 0.
                if action is not None:
                    step = action['step']
                    valid = all(trace[k]['cdf_bits'] is not None and base_trace[k]['cdf_bits'] is not None
                                for k in range(step, 20))
                    if valid:
                        current_delta = (trace[step]['cdf_bits'] - base_trace[step]['cdf_bits']) / n
                        future_delta = sum(trace[k]['cdf_bits'] - base_trace[k]['cdf_bits']
                                           for k in range(step + 1, 20)) / n
                    else:
                        current_delta = future_delta = None
                name = f'{record["name"]}-{i:02d}.bin'
                (out / 'streams' / name).write_bytes(blob)
                row = dict(name=record['name'], candidate=i, action=action, stream=name, bytes=len(blob),
                           bpp=bpp, mse=mse, j=bpp + a.lambda_mse * mse, elapsed_s=elapsed,
                           current_cdf_delta_bpp=current_delta, future_cdf_delta_bpp=future_delta,
                           escapes=sum(t['escapes'] for t in trace),
                           suffix_symbols_changed=0 if action is None else sum(
                               int(np.count_nonzero(np.array(trace[k]['symbols']) != base_trace[k]['symbols']))
                               for k in range(action['step'] + 1, 20)),
                           pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                           stream_sha256=hashlib.sha256(blob).hexdigest())
                results.append(row)
                log.write(json.dumps(row) + '\n')
                write_json(out / 'metrics.json', results)
                print(record['name'], i, 'J', row['j'], 'seconds', elapsed, flush=True)
    hook.close()
    write_json(out / 'smoke.json', dict(identity_stream_and_pixels=True, exact_integer_interventions=True,
                                       identical_prefix_and_hyperstream=True, device='cpu'))


def decode(out):
    torch, model, _ = setup(out)
    rows = json.loads((out / 'metrics.json').read_text())
    with torch.no_grad():
        for row in rows:
            path = out / 'streams' / row['stream']
            assert sha(path) == row['stream_sha256']
            quality, legacy, size, shape, strings = unpack(path.read_bytes())
            assert quality == 2 and legacy
            image = pixels(model.decompress(strings, shape)['x_hat'])
            assert tuple(image.shape[:2]) == size
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['pixel_sha256']
    write_json(out / 'fresh_decode.json', dict(passed=len(rows), expected=len(rows),
                                              encoder_hook_installed=False, source_cache_read=False))


def analyze(out):
    a = config(out)
    rows = json.loads((out / 'metrics.json').read_text())
    images = []
    for name in sorted({r['name'] for r in rows}):
        group = [r for r in rows if r['name'] == name]
        base = group[0]
        assert base['candidate'] == 0
        best = min(group, key=lambda r: r['j'])
        diagnostic = [r for r in group if r['current_cdf_delta_bpp'] is not None]
        local = min(diagnostic, key=lambda r: r['current_cdf_delta_bpp'] + a.lambda_mse * (r['mse'] - base['mse']))
        # Both scores use identical candidates and terminal distortion. Only
        # future rate is removed in the myopic score; no practical speed claim.
        full_cdf = min(diagnostic, key=lambda r: r['current_cdf_delta_bpp'] + r['future_cdf_delta_bpp']
                       + a.lambda_mse * (r['mse'] - base['mse']))
        images.append(dict(name=name, baseline_j=base['j'], best_j=best['j'], best_candidate=best['candidate'],
                           myopic_rate_candidate=local['candidate'], myopic_rate_j=local['j'],
                           full_cdf_candidate=full_cdf['candidate'], full_cdf_j=full_cdf['j'],
                           future_rate_changes_choice=local['candidate'] != full_cdf['candidate'],
                           candidates=len(group), cdf_eligible=len(diagnostic),
                           mean_encode_decode_s=float(np.mean([r['elapsed_s'] for r in group]))))
    base = np.array([r['baseline_j'] for r in images])
    best = np.array([r['best_j'] for r in images])
    local = np.array([r['myopic_rate_j'] for r in images])
    delta = best - local
    rng = np.random.default_rng(a.seed)
    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    relative = float(delta.mean() / local.mean() * 100)
    ci = np.quantile(boot, [.025, .975]).tolist()
    write_json(out / 'analysis.json', dict(images=images,
        best_vs_baseline_percent=float((best.mean() / base.mean() - 1) * 100),
        best_vs_myopic_rate_percent=relative, delta_j_ci95=ci,
        exploratory_gate_passed=relative <= -1 and ci[1] < 0,
        caveats=['4 reused development images', 'same candidates, not equal runtime search',
                 'finite candidate minimum is not a global optimum', 'no critic trained',
                 'bootstrap is exploratory, not a generalization guarantee']))


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, out / 'config.json')
    archive = out / 'source_archive'
    archive.mkdir()
    for name in ['lookahead_pilot.py', 'probe.py', 'stream.py']:
        shutil.copyfile(Path(__file__).parent / name, archive / name)
    start = time.monotonic()
    try:
        for stage in ['search', 'decode']:
            write_json(out / 'status.json', dict(state='running', stage=stage))
            remaining = config(out).wall_time_limit_s - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Pilot wall-time limit')
            subprocess.run([sys.executable, __file__, '--stage', stage, '--out', str(out)],
                           check=True, timeout=remaining)
        analyze(out)
        write_json(out / 'status.json', dict(state='complete', elapsed_s=time.monotonic() - start))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=repr(exc), elapsed_s=time.monotonic() - start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--stage', choices=['run', 'search', 'decode'], default='run')
    args = p.parse_args()
    if args.stage == 'run':
        run(args)
    else:
        globals()[args.stage](args.out)
