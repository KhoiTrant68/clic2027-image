"""Generate absolute paths for this extracted bundle, without importing torch."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--artifacts',type=Path,required=True)
p.add_argument('--gpu',type=int,default=0)
p.add_argument('--out',type=Path,default=ROOT/'portable/local.json')
a=p.parse_args()
if a.gpu<0:raise ValueError('GPU index must be nonnegative')
base=a.artifacts.resolve()
cfg=json.loads((ROOT/'experiments/mlicpp/end_to_end.json').read_text())
cfg.update(data=str(base/'data/div2k/pilot224'),test_data=str(base/'data/div2k/end-to-end-test32'),
    weights=str(base/'mlicpp/weights'),upstream=str(ROOT/'refs/repos/MLIC'),gpu_index=a.gpu)
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x',encoding='utf-8') as f:json.dump(cfg,f,indent=2)
print(a.out.resolve())
