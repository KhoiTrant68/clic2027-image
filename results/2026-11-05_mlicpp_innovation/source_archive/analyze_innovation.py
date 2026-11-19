"""Audit held-out source identities, actual byte constraints, fresh parity and CIs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(root):
    assert read(root/'status.json')['state']=='complete'
    c=read(root/'config.json');rows=read(root/'test_metrics.json')
    methods=['base','conditional','fixed','pixel','latent_tto']
    names=sorted(r['name'] for r in rows if r['method']=='base')
    by={(r['method'],r['name']):r for r in rows}
    assert len(names)==len(set(names))==c['test_limit']
    assert len(by)==len(rows)==len(methods)*len(names)
    assert read(root/'fresh.json')['matches']==len(rows)
    manifest=read(root/'cache/manifest.json')
    assert manifest['test_source_disjoint_previous_288']
    audit=read(root/'source_audit.json')
    assert audit['images']==len(names) and audit['disjoint_smoke'] and audit['cached_target_matches_source_crop']
    for name in names:
        baseline=by['base',name]['bytes']
        assert all(by[m,name]['bytes']==baseline+46 for m in ['conditional','fixed','pixel'])
        assert by['latent_tto',name]['bytes']<=baseline+46
    for name,digest in read(root/'sources.sha256.json').items():
        assert hashlib.sha256((root/'source_archive'/name).read_bytes()).hexdigest()==digest,name
    assert read(root/'conditional/model.json')['parameters']==read(root/'fixed/model.json')['parameters']
    ix=np.random.default_rng(c['seed']).integers(0,len(names),(10000,len(names)))
    comparisons=[]
    for treatment,control in [('conditional','fixed'),('conditional','pixel'),('conditional','latent_tto'),('conditional','base'),('fixed','base'),('latent_tto','base')]:
        for metric in ['lpips','dists','edge','mse']:
            delta=np.asarray([by[treatment,n][metric]-by[control,n][metric] for n in names])
            comparisons.append(dict(treatment=treatment,control=control,metric=metric,delta=float(delta.mean()),
                ci95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()))
    tto=[r for r in rows if r['method']=='latent_tto']
    result=dict(test_sources=len(names),bootstrap_replicates=10000,seed=c['seed'],fresh_pixel_matches=len(rows),
        correction_overhead_bytes=46,primary_parameter_count=read(root/'conditional/model.json')['parameters'],
        source_hashes_verified=True,tto_candidates_attempted_known=sum(r['candidates_attempted'] or 0 for r in tto),
        tto_candidates_feasible_known=sum(r['candidates_feasible'] or 0 for r in tto),
        tto_candidate_counts_unknown=sum(r['candidates_attempted'] is None for r in tto),
        tto_numerical_stops=[dict(name=r['name'],reason=r.get('numerical_stop')) for r in tto if r.get('numerical_stop')],
        tto_guard_rejections=[dict(name=r['name'],**rejection) for r in tto for rejection in r.get('rejected_candidates',[])],
        tto_changed_streams=sum(by['base',r['name']]['packet_sha256']!=r['packet_sha256'] for r in tto),
        tto_changed_outputs=sum(by['base',r['name']]['pixel_sha256']!=r['pixel_sha256'] for r in tto),
        comparisons=comparisons,
        limitations='One seed/rate, new32 DIV2K crops; exploratory pointwise CIs; fixed vs conditional is primary. No prior-confidence or human preference validation; unequal encoder compute across families.')
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
