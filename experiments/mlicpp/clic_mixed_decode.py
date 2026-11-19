"""Decode a frozen MLIC++ Mixed CLIC archive and optionally time decoder stages."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

from PIL import Image

from geometry_codec import Quantizer
from probe import load
from stream import unpack


@contextmanager
def timed(target, name, totals, label, synchronize):
    function = getattr(target, name)

    def wrapper(*args, **kwargs):
        synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        synchronize()
        totals[label] += time.perf_counter() - start
        return result

    setattr(target, name, wrapper)
    try:
        yield
    finally:
        setattr(target, name, function)


def add_profile(stack, model, totals, synchronize):
    import models.mlicpp as upstream

    for name in ('entropy_parameters_anchor', 'entropy_parameters_nonanchor',
                 'global_intra_context', 'global_inter_context', 'channel_context', 'local_context'):
        for module in getattr(model, name):
            if module is not None:
                stack.enter_context(timed(module, 'forward', totals, 'context', synchronize))
    for name in ('lrp_anchor', 'lrp_nonanchor'):
        for module in getattr(model, name):
            stack.enter_context(timed(module, 'forward', totals, 'lrp', synchronize))
    for target, name, label in (
        (model.g_s, 'forward', 'synthesis'), (model.h_s, 'forward', 'hyper_synthesis'),
        (model.entropy_bottleneck, 'decompress', 'hyper_entropy'),
        (model.gaussian_conditional, 'build_indexes', 'cdf_indexes'),
    ):
        stack.enter_context(timed(target, name, totals, label, synchronize))
    original = upstream.RansDecoder

    class MeasuredRans:
        def __init__(self):
            self.decoder = original()

        def set_stream(self, stream):
            return self.decoder.set_stream(stream)

        def decode_stream(self, *args):
            start = time.perf_counter()
            result = self.decoder.decode_stream(*args)
            totals['rans'] += time.perf_counter() - start
            return result

    upstream.RansDecoder = MeasuredRans
    stack.callback(setattr, upstream, 'RansDecoder', original)


def run(args):
    with zipfile.ZipFile(args.input) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    if manifest['version'] != 1 or manifest['quality'] != 2 or manifest['decoder'] != 2:
        raise ValueError('Expected a Mixed q2 archive')
    entropy_device = manifest.get('entropy_device', 'cuda')
    if entropy_device != args.device:
        raise ValueError(f'Bitstream was encoded for {entropy_device}; decode with that device and runtime')

    import torch

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.device == 'cpu':
        torch.cuda.synchronize = lambda *a, **kw: None
    else:
        limit = args.memory_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit))
    model, _ = load(SimpleNamespace(upstream=args.root/'upstream', weights=args.root/'weights',
                                    legacy=True, device=args.device), 2)
    checkpoint = torch.load(args.root/'mixed.pt', map_location='cpu', weights_only=True)
    model.g_s.load_state_dict(checkpoint['synthesis'])
    model.eval().requires_grad_(False)
    model.update(force=True)
    args.output.mkdir(exist_ok=False)
    rows = []
    totals = defaultdict(float)
    with zipfile.ZipFile(args.input) as archive, torch.inference_mode(), ExitStack() as stack:
        entries = manifest['images']
        names = set()
        if args.profile:
            add_profile(stack, model, totals, torch.cuda.synchronize)
        quant = Quantizer(model, None, None)
        stack.callback(quant.close)
        for record in entries[:args.limit]:
            name = record['name']
            if Path(name).name != name or not name.endswith('.png') or name in names:
                raise ValueError(f'Invalid or duplicate image name: {name}')
            names.add(name)
            payload = archive.read(record['stream'])
            if hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError(f'Stream hash mismatch: {name}')
            q, legacy, (h, w), shape, strings = unpack(payload)
            factor = float(record['factor'])
            if q != 2 or not legacy or not math.isfinite(factor) or not 0 < factor <= 16:
                raise ValueError(f'Invalid stream metadata: {name}')
            quant.select('uniform', factor)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            totals.clear()
            start = time.perf_counter()
            result = model.decompress(strings, shape)['x_hat'][0, :, :h, :w]
            decode_s = time.perf_counter() - start
            pixels = result.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            if pixels.shape != (h, w, 3):
                raise ValueError(f'Decoder output shape mismatch: {name}')
            Image.fromarray(pixels).save(args.output/name)
            row = dict(name=name, width=w, height=h, decode_s=decode_s,
                       pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), stages=dict(totals))
            rows.append(row)
            print(f'{len(rows)}/{len(entries)} {name} {decode_s:.3f} s', flush=True)
    if args.report:
        args.report.write_text(json.dumps(dict(device=args.device,
            synchronized_stages=args.profile and args.device == 'cuda', images=rows), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--input', type=Path, default=Path('bs.zip'))
    p.add_argument('--output', type=Path, default=Path('images'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--limit', type=int)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if args.memory_mib < 1 or (args.limit is not None and args.limit < 1):
        p.error('Memory cap and image limit must be positive')
    run(args)
