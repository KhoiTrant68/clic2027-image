"""Audit source crops and collect completed pilot evidence without training data."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
from PIL import Image
import torch
from probe import sha, write_json

out = Path(sys.argv[1])
cfg = json.loads((out/'config.json').read_text())
manifest = json.loads((out/'manifest.json').read_text())
assert json.loads((out/'status.json').read_text())['state'] == 'complete'
sources = {}
for path in cfg['excluded_manifests']+[cfg['test_data']+'/manifest.json']:
    root = Path(path).parent
    for row in json.loads(Path(path).read_text())['images']:
        sources[row['name']] = (root/'images'/row['name'], row['sha256'])
checked = []; by_split = {}
for row in manifest['records']:
    path, expected = sources[row['source']]
    assert sha(path) == expected
    with Image.open(path) as im:
        x, y = row['crop'] if row['split']=='test' else (row['x'], row['y'])
        source = np.array(im.convert('RGB').crop((x,y,x+256,y+256)))
    cached = torch.load(out/'cache'/(row['name']+'.pt'), weights_only=True)['original'][0].permute(1,2,0).numpy()
    assert np.array_equal(cached, source)
    by_split.setdefault(row['split'],set()).add(expected)
    checked.append(dict(name=row['name'],split=row['split'],source_sha256=expected,
                        crop_sha256=hashlib.sha256(source.tobytes()).hexdigest()))
assert not by_split['train'] & by_split['validation']
assert not by_split['train'] & by_split['test']
assert not by_split['validation'] & by_split['test']
write_json(out/'source_audit.json',dict(passed=len(checked),unique_sources={k:len(v) for k,v in by_split.items()},
    train_validation_test_disjoint=True,pretrained_overlap='unknown',records=checked))
files = [p for p in out.rglob('*') if p.is_file() and 'cache' not in p.relative_to(out).parts and p.suffix != '.tgz'
         and p.name != 'artifact.sha256.json']
write_json(out/'artifact.sha256.json',{str(p.relative_to(out)):sha(p) for p in files})
with tarfile.open(out/'evidence.tgz','w:gz') as tar:
    for p in files+[out/'artifact.sha256.json']:
        tar.add(p,arcname=str(p.relative_to(out)))
print('collected',len(files),'files; crop audit',len(checked))
