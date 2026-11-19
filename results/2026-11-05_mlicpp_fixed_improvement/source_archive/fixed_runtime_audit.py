from pathlib import Path
import json,sys,time,hashlib,subprocess
root=Path('/home/jovyan/clic2027-image')
out=Path('/home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105/div2k-q2-fixed-train1000-5000-search40-160-new32')
snapshot=out/'source_archive/runtime';snapshot.mkdir(exist_ok=True)
paths=['src/clic27/eval/metrics.py','experiments/new_codecs/half_vram.py','experiments/mlicpp/projection_probe.py']
manifest={}
for name in paths:
    p=root/name;data=p.read_bytes();target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    manifest[name]=hashlib.sha256(data).hexdigest()
(snapshot/'sha256.json').write_text(json.dumps(manifest,indent=2))
import torch,compressai,numpy
versions={'python':sys.version,'torch':torch.__version__,'compressai':compressai.__version__,'numpy':numpy.__version__}
(out/'runtime.json').write_text(json.dumps(versions,indent=2))
old={r['step']:r for r in map(json.loads,(out/'reference_fixed/train.jsonl').read_text().splitlines())}
new={r['step']:r for r in map(json.loads,(out/'fixed/train.jsonl').read_text().splitlines())}
common=sorted(set(old)&set(new));bad=[s for s in common if old[s]['loss']!=new[s]['loss']]
print(json.dumps({'common_prefix_logged_steps':len(common),'last_step':common[-1],'mismatch_steps':bad,'runtime':versions}))
