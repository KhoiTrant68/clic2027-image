from pathlib import Path
import json,hashlib,tarfile,sys,shutil
import numpy as np
root=Path('/home/jovyan/clic2027-image')
sys.path.insert(0,str(root/'experiments/mlicpp'))
from innovation_stream import unpack
out=Path('/home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105/div2k-q2-fixed-train1000-5000-search40-160-new32')
assert json.loads((out/'status.json').read_text())['state']=='complete'
for name in ['fixed_improvement_figures.py','fixed_runtime_audit.py','run_fixed_improvement.sh','fixed_improvement_q2.json','fixed_improvement_smoke.json']:
    shutil.copyfile(root/'experiments/mlicpp'/name,out/'source_archive'/name)
rows=json.loads((out/'test_metrics.json').read_text());by={(r['method'],r['name']):r for r in rows};audit={}
for method in sorted({r['method'] for r in rows}-{'base'}):
    values=np.concatenate([unpack(p.read_bytes())[2].reshape(-1) for p in sorted((out/'streams'/method).glob('*.bin'))])
    audit[method]=dict(coefficients=len(values),zero_fraction=float(np.mean(values==0)),saturated_fraction=float(np.mean(np.abs(values.astype(int))==127)),mean_absolute_coefficient=float(np.mean(np.abs(values.astype(int)))))
for key in ['reference','long']:
    audit[key+'_search_change']=dict(changed_streams=sum(by[key+'_s40',r['name']]['packet_sha256']!=by[key+'_s160',r['name']]['packet_sha256'] for r in rows if r['method']=='base'),
        changed_outputs=sum(by[key+'_s40',r['name']]['pixel_sha256']!=by[key+'_s160',r['name']]['pixel_sha256'] for r in rows if r['method']=='base'))
(out/'coefficient_audit.json').write_text(json.dumps(audit,indent=2))
files=[out/'fixed/last.pt',out/'reference_fixed/last.pt']
(out/'checkpoints.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n' for p in files))
with tarfile.open(out.parent/'fixed-improvement-evidence.tgz','w:gz') as archive:
    for p in out.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(out)
        if rel.parts[0]=='recon' or (rel.parts[0]=='cache' and rel.name!='manifest.json'):continue
        archive.add(p,arcname=str(rel))
smoke=out.parent/'smoke-fixed-training-search-2x2'
with tarfile.open(out.parent/'fixed-improvement-smoke.tgz','w:gz') as archive:
    for p in smoke.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.log']:archive.add(p,arcname=str(p.relative_to(smoke)))
print((out/'status.json').read_text())
print((out/'summary.json').read_text())
print((out/'coefficient_audit.json').read_text())
