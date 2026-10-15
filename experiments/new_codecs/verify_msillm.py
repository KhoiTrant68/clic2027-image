"""Selected mix MS-ILLM points: saved streams, fresh decoder, byte and pixel audit.

Uses explicit local runtime/checkpoints; this is not a self-contained package.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from msillm_stream import pack, unpack


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['run', 'encode', 'decode'], default='run')
    for name in ['out', 'hub', 'weights']:
        p.add_argument('--'+name, type=Path, required=True)
    for name in ['bakeoff', 'data', 'reference']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--gpu-memory-guard', action='store_true', required=True)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--limit', type=int, default=5, choices=range(1, 6))
    a = p.parse_args()
    if a.stage == 'run':
        for path in [a.hub/'hubconf.py', a.weights/'msillm_target_0.3bpp.ckpt',
                     a.weights/'msillm_target_0.45bpp.ckpt', a.bakeoff/'allocation.csv']:
            if not path.is_file():
                raise FileNotFoundError(path)
        a.out.mkdir(parents=True, exist_ok=False)
        with (a.bakeoff/'allocation.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['cand']=='mix' and float(r['rate'])==.3 and r['point_cand']=='msillm']
        if len(rows) != 5:
            raise ValueError('Expected the 5 MS-ILLM selections of the integrated mix')
        rows = rows[:a.limit]
        for r in rows:
            for path in [a.data/(r['name']+'.png'), a.reference/r['key']/(r['name']+'.png')]:
                if not path.is_file():
                    raise FileNotFoundError(path)
        (a.out/'selection.json').write_text(json.dumps(rows, indent=2))
        commands, times = [], {}
        for stage in ['encode', 'decode']:
            cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                   '--out', str(a.out), '--hub', str(a.hub), '--weights', str(a.weights), '--gpu-memory-guard']
            if stage == 'encode':
                cmd += ['--data', str(a.data)]
            if a.deterministic:
                cmd += ['--deterministic']
            commands.append(cmd)
            (a.out/'commands.json').write_text(json.dumps(commands, indent=2))
            start = time.perf_counter()
            with (a.out/(stage+'.log')).open('w') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            times[stage+'_process_wall_s'] = time.perf_counter()-start
        import numpy as np
        from PIL import Image
        comparisons = []
        for r in rows:
            name = r['name']+'.png'
            x = np.asarray(Image.open(a.out/'encoded_recon'/name)).astype(np.int16)
            y = np.asarray(Image.open(a.out/'decoded'/name)).astype(np.int16)
            old = np.asarray(Image.open(a.reference/r['key']/name)).astype(np.int16)
            comparisons.append(dict(name=r['name'], fresh_exact=bool(np.array_equal(x,y)),
                                    fresh_max_abs=int(np.abs(x-y).max()), fresh_mae=float(np.abs(x-y).mean()),
                                    fresh_changed_values=int(np.count_nonzero(x-y)), baseline_exact=bool(np.array_equal(y,old)),
                                    baseline_max_abs=int(np.abs(y-old).max()), baseline_mae=float(np.abs(y-old).mean())))
        records = json.loads((a.out/'encode.json').read_text())['images']
        extra = sum(r['bytes']-r['baseline_bytes'] for r in records)
        summary = json.loads((a.bakeoff/'summary.json').read_text())['mix@0.3/all']
        budgets = json.loads((a.bakeoff/'budgets.json').read_text())
        total = summary['bytes']+extra
        result = dict(n=len(rows), deterministic=a.deterministic, **times, baseline_mix_bytes=summary['bytes'], msillm_byte_correction=extra,
                      mix_bytes_after_msillm_correction=total, budget=budgets['budgets_bytes']['0.3'],
                      fits_after_msillm_correction=total<=budgets['budgets_bytes']['0.3'],
                      scope='MS-ILLM branch only; other branches have not all been framed/audited', images=comparisons)
        (a.out/'verification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        if not all(r['fresh_exact'] for r in comparisons):
            raise RuntimeError('Fresh-process parity failed; see verification.json')
        return

    def offline(*args, **kwargs):
        raise RuntimeError('Python TCP disabled for local codec verification')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = offline
    os.environ['HF_HUB_OFFLINE'] = '1'
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from half_vram import install
    stop = install(a.out/(a.stage+'.vram.jsonl'))
    import torch
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(a.hub))
    from neuralcompression._outputs import HyperpriorCompressedOutput
    torch.manual_seed(0)
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dst = a.out/('encoded_recon' if a.stage=='encode' else 'decoded')
    dst.mkdir(exist_ok=False)
    records, loads = [], []
    if a.stage == 'encode':
        (a.out/'streams').mkdir(exist_ok=False)
        rows = json.loads((a.out/'selection.json').read_text())
    else:
        # Decoder learns quality and dimensions only from transmitted bytes.
        rows = [dict(name=f.stem, key='q'+str(unpack(f.read_bytes())[0])) for f in sorted((a.out/'streams').glob('*.bin'))]
    for key in sorted({r['key'] for r in rows}):
        q = int(key[1:])
        checkpoint = a.weights/('msillm_target_'+{4:'0.3',5:'0.45'}[q]+'bpp.ckpt')
        checkpoint_sha = sha(checkpoint)
        t0 = time.perf_counter()
        model = torch.hub.load(str(a.hub), 'msillm_quality_'+str(q), source='local', pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model = model.to('cuda').eval()
        model.update()
        model.update_tensor_devices('compress')
        torch.cuda.synchronize()
        loads.append(dict(key=key, seconds=time.perf_counter()-t0, checkpoint_sha256=checkpoint_sha))
        for r in [r for r in rows if r['key']==key]:
            stream = a.out/'streams'/(r['name']+'.bin')
            with torch.no_grad():
                if a.stage == 'encode':
                    x = np.array(Image.open(a.data/(r['name']+'.png')).convert('RGB'))
                    t = torch.from_numpy(x.copy()).permute(2,0,1)[None].float().div(255).to('cuda')
                    compressed = model.compress(t, force_cpu=False)
                    if len(compressed.latent_strings)!=1 or len(compressed.hyper_latent_strings)!=1:
                        raise ValueError('Expected a single-image batch')
                    stream.write_bytes(pack(q, compressed.image_size, compressed.padded_size,
                                           compressed.latent_strings[0], compressed.hyper_latent_strings[0]))
                saved_q, size, padded, latent, hyper = unpack(stream.read_bytes())
                if saved_q != q:
                    raise ValueError('Wrong model for bitstream')
                state = HyperpriorCompressedOutput([latent], [hyper], size, padded)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = model.decompress(state, force_cpu=False)
                torch.cuda.synchronize()
                decode_s = time.perf_counter()-t0
                t0 = time.perf_counter()
                pixels = y[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                Image.fromarray(pixels).save(dst/(r['name']+'.png'), compress_level=0)
                rec = dict(name=r['name'], key=key, bytes=stream.stat().st_size, payload_bytes=len(latent)+len(hyper),
                           decode_s=decode_s, write_s=time.perf_counter()-t0, sha256=sha(stream))
                if a.stage=='encode':
                    rec['baseline_bytes']=int(r['bytes'])
                records.append(rec)
                print(json.dumps(rec), flush=True)
                del y
        del model
        torch.cuda.empty_cache()
    (a.out/(a.stage+'.json')).write_text(json.dumps(dict(torch=torch.__version__, deterministic=a.deterministic,
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        network='Python TCP disabled',
        loads=loads, images=records, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20), indent=2))
    stop.set()


if __name__ == '__main__':
    main()
