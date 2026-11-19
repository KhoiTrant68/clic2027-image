"""Build and validate a one-image CPU stream from the frozen Mixed validation inputs."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import torch

from geometry_codec import Quantizer
from probe import load
from stream import pack, unpack


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # The upstream timings unconditionally call CUDA synchronization.
    torch.cuda.synchronize = lambda *a, **kw: None
    with zipfile.ZipFile(args.gpu_archive) as archive:
        row = json.loads(archive.read('manifest.json'))['images'][0]
        gpu_payload = archive.read(row['stream'])
    quality, legacy, (h, w), _, gpu_strings = unpack(gpu_payload)
    with Image.open(args.references/row['name']) as reference:
        if reference.size != (w, h):
            raise ValueError('Reference dimensions differ from archived stream')
        image = np.array(reference.convert('RGB'))
    image = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
    image = torch.nn.functional.pad(image, (0, (-w)%64, 0, (-h)%64), mode='replicate')
    model, _ = load(SimpleNamespace(upstream=args.root/'upstream', weights=args.root/'weights',
                                    legacy=legacy, device='cpu'), quality)
    mixed = torch.load(args.root/'mixed.pt', map_location='cpu', weights_only=True)
    model.g_s.load_state_dict(mixed['synthesis'])
    model.update(force=True)
    model.eval().requires_grad_(False)
    args.out.mkdir(exist_ok=False)
    with torch.inference_mode(), ExitStack() as stack:
        quant = Quantizer(model, None, None)
        stack.callback(quant.close)
        quant.select('uniform', row['factor'])
        encoded = model.compress(image)
        payload = pack(quality, legacy, (h, w), encoded['shape'], encoded['strings'])
        print(f'Encoded CPU stream, {len(payload)} bytes', flush=True)
        output = model.decompress(encoded['strings'], encoded['shape'])['x_hat'][0, :, :h, :w]
        pixels = output.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    ref = image[0, :, :h, :w].mul(255).round().byte().permute(1, 2, 0).numpy()
    row['sha256'] = digest(payload)
    manifest = dict(version=1, quality=quality, decoder=2, entropy_device='cpu', images=[row])
    with zipfile.ZipFile(args.out/'bs.zip', 'w') as archive:
        archive.writestr(row['stream'], payload)
        archive.writestr('manifest.json', json.dumps(manifest, separators=(',', ':')))
    summary = dict(name=row['name'], factor=row['factor'], pixel_sha256=digest(pixels.tobytes()),
                   cpu_payload_bytes=len(payload), gpu_payload_bytes=len(gpu_payload),
                   main_stream_identical=encoded['strings'][0] == gpu_strings[0],
                   hyper_stream_identical=encoded['strings'][1] == gpu_strings[1],
                   mae_to_reference=float(np.abs(pixels.astype(np.int16)-ref).mean()))
    (args.out/'expected.json').write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'gpu-archive', 'references', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    main(parser.parse_args())
