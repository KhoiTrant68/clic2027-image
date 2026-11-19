"""Offline evidence audit and paired recognition analysis for the frozen pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(rows, seed):
    delta=np.asarray([int(r['perceptual_correct'])-int(r['base_correct']) for r in rows])
    ix=np.random.default_rng(seed).integers(0,len(rows),(10000,len(rows)))
    return dict(n=len(rows),**{method:sum(r[method+'_correct'] for r in rows) for method in ['source','base','perceptual']},
        harmed=sum(r['base_correct'] and not r['perceptual_correct'] for r in rows),
        recovered=sum(not r['base_correct'] and r['perceptual_correct'] for r in rows),
        accuracy_delta=float(delta.mean()),ci95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist())


def main(root):
    assert read(root/'status.json')['state']=='complete'
    cfg=read(root/'config.json'); gate=read(root/'source_gate.json')
    assert gate['passed']
    rows=read(root/'codec.json')['rows']; ocr=read(root/'independent_ocr.json')
    annotations={r['name']:r for r in read(root/'data.json')['rows'] if r['split']=='test'}
    assert len(rows)==len({r['name'] for r in rows})==cfg['test_images']==256
    assert read(root/'fresh.json')['matches']==2*len(rows)
    expected=[r for r in gate['rows'] if r['split']=='test'][0]['predictions']
    assert [r['source_prediction'] for r in rows]==[cfg['classes'][p] for p in expected]
    for name,digest in read(root/'sources.sha256.json').items():
        assert hashlib.sha256((root/'source_archive'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((root/'source_archive/recognition_ocr.py').read_bytes()).hexdigest()==ocr['source_sha256']
    groups=[]
    for field in ['font','size','contrast','texture']:
        for value in sorted({r[field] for r in annotations.values()}):
            subset=[r for r in rows if annotations[r['name']][field]==value]
            groups.append(dict(field=field,value=value,**compare(subset,cfg['seed'])))
    cnn=compare(rows,cfg['seed']); cnn_eligible=compare([r for r in rows if r['source_correct']],cfg['seed'])
    result=dict(cnn=cnn,cnn_source_correct_subset=cnn_eligible,
        ocr=compare(ocr['rows'],cfg['seed']),
        ocr_source_correct_subset=compare([r for r in ocr['rows'] if r['source_correct']],cfg['seed']),
        cnn_groups=groups, actual_bpp=8*sum(r['bytes'] for r in rows)/(len(rows)*65536),
        source_gate_epoch=gate['epoch'],fresh_pixel_matches=2*len(rows),
        limitations='One seed, one held-out font family, known boxes, synthetic data, image-level exploratory CIs; no latent readout or human evaluation.')
    (root/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cnn_groups'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
