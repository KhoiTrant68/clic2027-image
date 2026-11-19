"""Check the actual cached targets against source hashes/crops and prior splits."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
import torch


def read(p):return json.loads(p.read_text())


def main(root,smoke):
    torch.set_num_threads(2)
    cfg=read(root/'config.json');dataset=Path(cfg['test_data'])
    source={r['name']:r for r in read(dataset/'manifest.json')['images']}
    selected=[r for r in read(root/'cache/manifest.json')['records'] if r['split']=='test']
    smoke_hashes={r['source_sha256'] for r in read(smoke/'cache/manifest.json')['records'] if r['split']=='test'}
    previous=set()
    for p in [Path('/home/jovyan/clic2027-artifacts/data/div2k/pilot224/manifest.json'),Path('/home/jovyan/clic2027-artifacts/data/div2k/cvpr-holdout64/manifest.json')]:
        previous.update(r['sha256'] for r in read(p)['images'])
    hashes={r['source_sha256'] for r in selected}
    assert len(selected)==len(hashes)==cfg['test_limit']
    assert not hashes&previous and not hashes&smoke_hashes
    for row in selected:
        path=dataset/'images'/row['source'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['source_sha256']==source[row['source']]['sha256']
        assert zlib.crc32(raw)==source[row['source']]['crc32']
        left,top=row['crop']
        with Image.open(path) as image:
            actual=np.array(image.convert('RGB').crop((left,top,left+256,top+256)))
        target=torch.load(root/'cache/test'/(row['name']+'.pt'),weights_only=True)['original'][0].permute(1,2,0).numpy()
        assert np.array_equal(actual,target)
        blob=(root/'cache/test'/(row['name']+'.bin')).read_bytes()
        assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['stream_sha256']
    result=dict(images=len(selected),previous_sources=len(previous),smoke_sources=len(smoke_hashes),
        disjoint_previous=True,disjoint_smoke=True,source_crc_and_sha_verified=True,
        cached_target_matches_source_crop=True,base_streams_verified=True,
        caveat='Disjointness applies to project pilot splits; author pretraining overlap not audited.')
    (root/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    a=p.parse_args();main(a.root,a.smoke)
