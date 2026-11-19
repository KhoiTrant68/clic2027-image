"""Collect lightweight, completed experiment evidence; leave checkpoints/PNGs/streams outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.run / 'status.json').read_text())
    if status.get('state') != 'complete' or not status.get('metrics'):
        raise ValueError('Only collect a completed metric evaluation')
    if not (a.run / 'comparison.json').is_file():
        raise ValueError('Run compare.py first')
    a.out.mkdir(parents=True, exist_ok=False)
    for path in sorted(a.run.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.log'):
            shutil.copyfile(path, a.out / path.name)
    freeze = Path(sys.prefix) / 'requirements.freeze.txt'
    shutil.copyfile(freeze, a.out / freeze.name)
    sources = list((ROOT / 'experiments/mlicpp').glob('*.py')) + list((ROOT / 'experiments/mlicpp').glob('*.sh'))
    sources += [ROOT / 'src/clic27/eval/metrics.py', ROOT / 'experiments/qhat/qhat.py',
                ROOT / 'experiments/bakeoff/bakeoff.py', ROOT / 'results/2026-10-15_new_codecs/qhat_v0.json']
    manifest = json.loads((a.run / 'manifest.json').read_text())
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if hashes['experiments/mlicpp/probe.py'] != manifest['runner_sha256']:
        raise ValueError('Runner changed during experiment; refuse to bind evidence to different source')
    report = dict(run=str(a.run), upstream_sha=manifest['upstream_sha'], sources=hashes,
                  files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in a.out.iterdir() if path.is_file()})
    (a.out / 'evidence.sha256.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(a.out), files=len(report['files']))))


if __name__ == '__main__':
    main()
