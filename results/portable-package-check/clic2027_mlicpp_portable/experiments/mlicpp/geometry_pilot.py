"""Bounded geometry teacher/distillation pilot; real streams and fresh decode."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from probe import ROOT, settings, load, sha, write_json


def config(out):
    return SimpleNamespace(**json.loads((out/'config.json').read_text()))


def setup(out, stage):
    torch, install = settings()
    torch.manual_seed(config(out).seed)
    return torch, install(out/(stage+'.vram.jsonl'))


def backbone(out):
    a = config(out)
    m, meta = load(SimpleNamespace(upstream=Path(a.upstream), weights=Path(a.weights), legacy=True), 2)
    return m.eval().requires_grad_(False), meta


def pixels(t):
    return t[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def teacher(torch, g, y, probes, seed):
    # E[(J^T v)^2] = diag(J^T J). Channel pooling is deliberate.
    rng = torch.Generator(device='cuda').manual_seed(seed)
    y = y.detach().requires_grad_(True)
    f = g(y)
    weight = torch.zeros_like(y)
    for i in range(probes):
        v = torch.randint(0, 2, f.shape, device='cuda', generator=rng).float()*2-1
        grad, = torch.autograd.grad((f*v).sum(), y, retain_graph=i+1 < probes)
        weight += grad.detach().square()/probes
    return weight.mean((2, 3)).clamp_min(1e-10).log().detach()


def prepare(out):
    from geometry_codec import features
    torch, stop = setup(out, 'prepare')
    a = config(out); m, meta = backbone(out)
    prior = Path(a.synthesis_run)
    old = json.loads((prior/'latents/manifest.json').read_text())['records']
    selected = []
    for split, limit in [('train', a.train_limit), ('validation', a.validation_limit)]:
        selected += [dict(r) for r in old if r['split'] == split][:limit]
    cache = out/'cache'; cache.mkdir()
    train_sources = {r['source'] for r in selected if r['split'] == 'train'}
    validation_sources = {r['source'] for r in selected if r['split'] == 'validation'}
    assert not train_sources & validation_sources
    manifest = json.loads((Path(a.test_data)/'manifest.json').read_text())
    excluded = set()
    for path in a.excluded_manifests:
        excluded.update(r['name'] for r in json.loads(Path(path).read_text())['images'])
    if not a.smoke:
        assert not {r['name'] for r in manifest['images']} & excluded
    for r in manifest['images'][:a.test_limit]:
        selected.append(dict(name=Path(r['name']).stem+'-geometry', source=r['name'],
                             source_sha256=r['sha256'], split='test'))
    captured = []
    hook = m.g_s.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach()))
    hcaptured = []
    hh = m.h_s.register_forward_hook(lambda mod, args, h: hcaptured.append(h.detach()))
    for index, r in enumerate(selected):
        if r['split'] != 'test':
            v = torch.load(prior/'latents'/r['split']/(r['name']+'.pt'), weights_only=True)
            original = v['original']
        else:
            path = Path(a.test_data)/'images'/r['source']; assert sha(path) == r['source_sha256']
            with Image.open(path) as image:
                image = image.convert('RGB'); rng = random.Random('9041/'+r['source'])
                left, top = rng.randrange(image.width-255), rng.randrange(image.height-255)
                original = torch.from_numpy(np.array(image.crop((left, top, left+256, top+256))).copy()).permute(2, 0, 1)[None]
                r['crop'] = [left, top]
        x = original.cuda().float()/255
        with torch.no_grad():
            encoded = m.compress(x)
            captured.clear(); hcaptured.clear()
            base = m.decompress(encoded['strings'], encoded['shape'])['x_hat']
            y, h = captured[-1], hcaptured[-1]
        w = teacher(torch, m.g_s, y, a.probes, a.seed+index)
        torch.save(dict(original=original.cpu(), feature=features(h).cpu(), metric=w.cpu()), cache/(r['name']+'.pt'))
        write_json(out/'prepare.progress.json', dict(done=index+1, total=len(selected)))
        if index % 16 == 0: print('teacher', index+1, len(selected), flush=True)
    hook.remove(); hh.remove()
    meta.update(records=selected, test_disjoint_prior=not a.smoke, train_sources=len(train_sources),
                validation_sources=len(validation_sources), decoder='official MSE q2', teacher='RGB VJP diagonal, 4 probes unless smoke')
    write_json(out/'manifest.json', meta); stop.set()


def train(out):
    from geometry_codec import Predictor, centered
    torch, stop = setup(out, 'train'); a = config(out)
    records = json.loads((out/'manifest.json').read_text())['records']
    def tensors(split):
        data = [torch.load(out/'cache'/(r['name']+'.pt'), weights_only=True) for r in records if r['split'] == split]
        return torch.cat([v['feature'] for v in data]).cuda(), centered(torch.cat([v['metric'] for v in data]).cuda())
    x, y = tensors('train'); vx, vy = tensors('validation')
    model = Predictor(x.mean(0), x.std(0, unbiased=False).clamp_min(.01), y.mean(0)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate, betas=(.9, .99))
    start = time.monotonic(); logs = []
    with (out/'train.jsonl').open('w', buffering=1) as log:
        for step in range(1, a.steps+1):
            idx = torch.randint(len(x), (a.batch_size,), device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[idx])-y[idx]).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            assert torch.isfinite(loss)
            if step == 1 or step % 100 == 0 or step == a.steps:
                with torch.no_grad():
                    entry = dict(step=step, train_mse=float(loss), validation_mse=float((model(vx)-vy).square().mean()),
                                 global_validation_mse=float((y.mean(0)-vy).square().mean()), seconds=time.monotonic()-start)
                log.write(json.dumps(entry)+'\n'); logs.append(entry)
                write_json(out/'train.progress.json', entry); print(entry, flush=True)
    # Averaged metric in linear domain for the fixed-volume global optimum.
    raw = torch.cat([torch.load(out/'cache'/(r['name']+'.pt'), weights_only=True)['metric'] for r in records if r['split']=='train'])
    global_metric = raw.exp().mean(0, keepdim=True).log()
    torch.save(dict(model=model.cpu().state_dict(), global_metric=global_metric, step=a.steps), out/'predictor.pt')
    write_json(out/'train.summary.json', dict(parameters=sum(p.numel() for p in model.parameters()), logs=logs,
                checkpoint_sha256=sha(out/'predictor.pt'), objective='distillation of log RGB decoder sensitivity, not RD fine-tuning'))
    stop.set()


def trained(torch, out):
    from geometry_codec import Predictor
    ck = torch.load(out/'predictor.pt', weights_only=True)
    st = ck['model']; p = Predictor(st['mean'], st['std'], st['target'])
    p.load_state_dict(st)
    return p.cuda().eval().requires_grad_(False), ck['global_metric'].cuda()


def evaluate(out):
    from geometry_codec import Quantizer, relative, envelope, read, HEADER
    from stream import pack
    torch, stop = setup(out, 'evaluate'); a = config(out); m, meta = backbone(out)
    p, global_metric = trained(torch, out)
    import lpips
    import piq
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    dists = piq.DISTS(reduction='mean').cuda().eval().requires_grad_(False)
    quant = Quantizer(m, p, global_metric)
    records = [r for r in json.loads((out/'manifest.json').read_text())['records'] if r['split']=='test']
    folder = out/'streams'; folder.mkdir(); result = []
    with torch.no_grad():
        for r in records:
            v = torch.load(out/'cache'/(r['name']+'.pt'), weights_only=True)
            x = v['original'].cuda().float()/255
            # Verify identity wrapper against unmodified upstream on every image.
            quant.close(); baseline = m.compress(x); original = pixels(m.decompress(baseline['strings'], baseline['shape'])['x_hat'])
            raw = pack(2, True, (256, 256), tuple(baseline['shape']), baseline['strings'])
            quant = Quantizer(m, p, global_metric)
            quant.select('uniform', 1.)
            identity = m.compress(x)
            assert pack(2, True, (256,256), tuple(identity['shape']), identity['strings']) == raw
            assert np.array_equal(pixels(m.decompress(identity['strings'], identity['shape'])['x_hat']), original)
            for mode in ['uniform', 'global', 'predicted', 'oracle']:
                oracle = relative(v['metric']).cuda() if mode == 'oracle' else None
                for factor in a.factors:
                    quant.select(mode, factor, oracle)
                    enc = m.compress(x)
                    blob = envelope(enc, mode, factor, None if oracle is None else oracle.cpu().numpy())
                    mode2, factor2, side, shape, strings = read(blob)
                    quant.select(mode2, factor2, side)
                    value = m.decompress(strings, shape)['x_hat']
                    image = pixels(value); target = v['original'][0].permute(1,2,0).numpy()
                    uint = torch.from_numpy(image.copy()).permute(2,0,1)[None].cuda().float()/255
                    mse = float(((image.astype(np.float64)-target)**2).mean())
                    name = f'{r["name"]}-{mode}-{factor:g}.bin'; (folder/name).write_bytes(blob)
                    result.append(dict(name=r['name'], mode=mode, factor=factor, stream=name, bytes=len(blob),
                        bpp=len(blob)*8/65536, payload_only_bpp=(len(blob)-(1280 if mode=='oracle' else 0))*8/65536,
                        mse=mse, psnr=float(10*np.log10(255**2/max(mse,1e-10))), lpips=float(lp(uint*2-1,x*2-1)),
                        dists=float(dists(uint,x)), pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        stream_sha256=hashlib.sha256(blob).hexdigest(), original_codec_bytes=len(raw)))
            write_json(out/'metrics.json', result)
            print('evaluated', r['name'], len(result), flush=True)
    quant.close(); stop.set()


def decode(out):
    from geometry_codec import Quantizer, read
    torch, stop = setup(out, 'decode'); m, _ = backbone(out); p, g = trained(torch, out)
    quant = Quantizer(m, p, g)
    rows = json.loads((out/'metrics.json').read_text()); checked = []
    with torch.no_grad():
        for r in rows:
            blob = (out/'streams'/r['stream']).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == r['stream_sha256']
            mode, factor, side, shape, strings = read(blob)
            quant.select(mode, factor, side)
            image = pixels(m.decompress(strings, shape)['x_hat'])
            assert hashlib.sha256(image.tobytes()).hexdigest() == r['pixel_sha256']
            checked.append(r['stream'])
    write_json(out/'fresh_decode.json', dict(passed=len(checked), expected=len(rows), source_cache_read=False))
    quant.close(); stop.set()


def run(a):
    out = a.out; out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.config, out/'config.json')
    snapshot = out/'source_archive'; snapshot.mkdir()
    for f in ['geometry_pilot.py','geometry_codec.py','geometry_q2.json','probe.py','stream.py']:
        shutil.copyfile(Path(__file__).parent/f, snapshot/f)
    a_cfg = config(out); start = time.monotonic()
    try:
        for stage in ['prepare', 'train', 'evaluate', 'decode']:
            write_json(out/'status.json', dict(state='running', stage=stage, elapsed_s=time.monotonic()-start))
            remaining = a_cfg.wall_time_limit_s-(time.monotonic()-start)
            if remaining <= 0: raise TimeoutError('Pilot wall-time budget exhausted')
            subprocess.run([sys.executable, __file__, '--stage', stage, '--out', str(out)], check=True, timeout=remaining)
        write_json(out/'status.json', dict(state='complete', elapsed_s=time.monotonic()-start))
    except BaseException as e:
        write_json(out/'status.json', dict(state='failed', error=repr(e), elapsed_s=time.monotonic()-start)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['run','prepare','train','evaluate','decode'], default='run')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.stage == 'run': run(args)
    else: globals()[args.stage](args.out)
