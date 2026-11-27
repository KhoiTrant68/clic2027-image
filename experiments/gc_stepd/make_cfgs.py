"""Rewrite the data paths of configs/D*.json for this machine -> cfg/D*.json. Nothing else changes.
Usage: python make_cfgs.py DATA_ROOT"""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
d = Path(sys.argv[1]).resolve()
for sub in ('patches512_div2k', 'patches512_clic2020', 'valid30'):
    if not (d / sub).is_dir():
        sys.exit(f'missing {d / sub}')
(H / 'cfg').mkdir(exist_ok=True)
for p in sorted((H / 'configs').glob('D*.json')):
    c = json.loads(p.read_text())
    c['data']['train_dirs'] = [str(d / 'patches512_div2k'), str(d / 'patches512_clic2020')]
    c['data']['valid_dir'] = str(d / 'valid30')
    (H / 'cfg' / p.name).write_text(json.dumps(c, indent=1))
    print(p.name, c['model'], c['rate'])
