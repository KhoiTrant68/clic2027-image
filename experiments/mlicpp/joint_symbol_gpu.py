"""Joint hard-symbol coordinate search on six independent development sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from probe import settings, load, write_json
from stream import pack, unpack
from lookahead_pilot import Intervention, candidates, encode, pixels


class JointIntervention(Intervention):
    def start(self, actions):
        super().start(None)
        self.actions = actions or []

    def wrapper(self, name):
        original = super().wrapper(name)
        def call(*args):
            matches = [a for a in self.actions if a['step'] == self.cursor]
            self.action = None if not matches else dict(step=self.cursor,
                positions=[p for a in matches for p in a['positions']],
                deltas=[d for a in matches for d in a['deltas']])
            return original(*args)
        return call


def setup(out):
    a = SimpleNamespace(**json.loads((out/'config.json').read_text()))
    torch, _ = settings()
    torch.cuda.set_per_process_memory_fraction(a.allocator_fraction_per_worker)
    torch.manual_seed(a.seed)
    model, meta = load(SimpleNamespace(upstream=Path(a.upstream), weights=Path(a.weights), legacy=True), 2)
    model.requires_grad_(False)
    return a, torch, model, meta


def worker(out, index, fresh=False):
    a, torch, model, meta = setup(out)
    folder = out/f'worker-{index}'
    if fresh:
        result = json.loads((folder/'summary.json').read_text())
        with torch.no_grad():
            for mode in ['baseline', 'joint']:
                blob = (folder/(mode+'.bin')).read_bytes()
                assert hashlib.sha256(blob).hexdigest() == result[mode]['stream_sha256']
                q, legacy, size, shape, strings = unpack(blob)
                assert q == 2 and legacy and size == (256, 256)
                image = pixels(model.decompress(strings, shape)['x_hat'])
                assert hashlib.sha256(image.tobytes()).hexdigest() == result[mode]['pixel_sha256']
        write_json(folder/'fresh_decode.json',dict(passed=2, encoder_hook_installed=False))
        return
    folder.mkdir()
    write_json(folder/'backbone.json',meta)
    record = json.loads((out/'inventory.json').read_text())[index]
    v = torch.load(Path(a.reference_run)/'cache'/(record['name']+'.pt'),map_location='cpu',weights_only=True)
    x = v['original'].float().cuda()/255
    target = v['original'][0].permute(1,2,0).numpy().astype(np.float64)
    hook = JointIntervention(model)
    def evaluate(actions):
        blob, image, trace, seconds = encode(model,hook,x,actions)
        mse = float(((image.astype(np.float64)-target)**2).mean())
        return dict(bpp=len(blob)*8/65536,mse=mse,j=len(blob)*8/65536+a.lambda_mse*mse,
            seconds=seconds,stream_sha256=hashlib.sha256(blob).hexdigest(),
            pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()),blob,trace
    started = time.monotonic()
    with torch.no_grad(), (folder/'progress.jsonl').open('w',buffering=1) as log:
        baseline, base_blob, trace = evaluate([])
        hook.close()
        native = model.compress(x)
        assert base_blob == pack(2,True,(256,256),tuple(native['shape']),native['strings'])
        assert baseline['pixel_sha256'] == hashlib.sha256(pixels(model.decompress(native['strings'],native['shape'])['x_hat']).tobytes()).hexdigest()
        hook = JointIntervention(model)
        (folder/'baseline.bin').write_bytes(base_blob)
        actions, current, blob = [], baseline, base_blob
        calls = 1
        for sweep in range(a.sweeps):
            for step in a.steps:
                # Each group gets equal candidate count; later groups see the
                # accepted prefix and prior edits, unlike independent one-shot trials.
                options = candidates(trace,SimpleNamespace(steps=[step],sites_per_step=a.sites_per_step,block_sites=a.block_sites))
                winner = None
                for action in options:
                    score, trial_blob, trial_trace = evaluate(actions+[action])
                    calls += 1
                    log.write(json.dumps(dict(call=calls,sweep=sweep,step=step,action=action,**score))+'\n')
                    if score['j'] < current['j'] and (winner is None or score['j'] < winner[0]['j']):
                        winner = score, trial_blob, trial_trace, action
                if winner is not None:
                    current, blob, trace, action = winner
                    actions.append(action)
                write_json(folder/'progress.json',dict(sweep=sweep+1,step=step,calls=calls,accepted=len(actions),
                    baseline_j=baseline['j'],current_j=current['j'],elapsed_s=time.monotonic()-started))
                print(index,sweep+1,step,calls,current['j'],flush=True)
        (folder/'joint.bin').write_bytes(blob)
        write_json(folder/'summary.json',dict(source=record,baseline=baseline,joint=current,actions=actions,
            calls=calls,elapsed_s=time.monotonic()-started,identity_matches=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    hook.close()


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,out/'config.json')
    a=SimpleNamespace(**json.loads((out/'config.json').read_text()))
    raw=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(raw.strip()) < 512, 'GPU no longer empty; refusing concurrent launch'
    records=json.loads((Path(a.reference_run)/'manifest.json').read_text())['records']
    selected=[];seen=set()
    for r in records:
        if r['split']=='validation' and r['source'] not in seen and len(selected)<a.workers:
            selected.append(r);seen.add(r['source'])
    assert len(selected)==a.workers
    write_json(out/'inventory.json',selected)
    snapshot=out/'source_archive';snapshot.mkdir()
    for name in ['joint_symbol_gpu.py','lookahead_pilot.py','probe.py','stream.py','run_joint_symbol_gpu.sh']:
        shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    start=time.monotonic(); children=[]
    try:
        with (out/'gpu.jsonl').open('w',buffering=1) as monitor:
            for stage in ['search','decode']:
                write_json(out/'status.json',dict(state='running',stage=stage,workers=a.workers))
                children=[subprocess.Popen([sys.executable,__file__,'--out',str(out),'--stage',stage,'--worker',str(i)]) for i in range(a.workers)]
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):
                        raise RuntimeError('Worker failed; stopping own remaining workers')
                    total,used=map(int,subprocess.check_output(['nvidia-smi','-i','2',
                        '--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
                    monitor.write(json.dumps(dict(elapsed_s=time.monotonic()-start,used_mib=used,total_mib=total))+'\n')
                    if total-used < a.min_free_mib: raise RuntimeError('GPU headroom exhausted')
                    if time.monotonic()-start>a.wall_time_limit_s: raise TimeoutError('Wall time exceeded')
                    time.sleep(2)
                if any(p.returncode != 0 for p in children): raise RuntimeError('Worker failed')
        rows=[json.loads((out/f'worker-{i}'/'summary.json').read_text()) for i in range(a.workers)]
        base=np.mean([r['baseline']['j'] for r in rows]);joint=np.mean([r['joint']['j'] for r in rows])
        write_json(out/'analysis.json',dict(baseline_j=float(base),joint_j=float(joint),
            relative_j_percent=float((joint/base-1)*100),sources=len(rows),
            fresh_decode_passed=2*len(rows),caveat='Development only; no equal-runtime SGA comparison; no training'))
        write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-start))
    except BaseException as exc:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill();p.wait()
        write_json(out/'status.json',dict(state='failed',error=repr(exc),elapsed_s=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--stage',default='run',choices=['run','search','decode']);p.add_argument('--worker',type=int)
    args=p.parse_args()
    if args.stage=='run':run(args)
    else:worker(args.out,args.worker,args.stage=='decode')
