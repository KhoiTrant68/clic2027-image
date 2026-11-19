"""Greedy actual-byte selection; paired decoders get identical per-image streams."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile
from probe import ROOT,write_json


def package(selected,arm):
    buffer=io.BytesIO();records=[]
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for i,r in enumerate(selected):
            payload=r['stream_path'].read_bytes();filename=f'{i:03}.bin'
            assert hashlib.sha256(payload).hexdigest()==r['stream_sha256']
            z.writestr(filename,payload)
            records.append(dict(name=r['name'],stream=filename,factor=r['factor'],sha256=r['stream_sha256']))
        # Equal-length mode identifiers keep paired package sizes equal.
        z.writestr('manifest.json',json.dumps(dict(version=1,quality=2,decoder=arm,images=records),separators=(',',':')))
    return buffer.getvalue()


def run(a):
    assert json.loads((a.source/'status.json').read_text())['state']=='complete'
    a.out.mkdir(parents=True,exist_ok=False)
    points={};original={}
    for factor in [.5,1.,2.,4.]:
        for arm in ['original','perceptual']:
            folder=a.source/f'{arm}-factor{factor:g}'
            for r in json.loads((folder/'metrics.json').read_text()):
                key=(r['name'],factor)
                if arm=='original':original[key]=r
                else:
                    assert original[key]['stream_sha256']==r['stream_sha256']
                    points.setdefault(r['name'],[]).append(dict(r,factor=factor,stream_path=folder/'streams'/r['stream']))
    names=sorted(points);pixels=sum(points[n][0]['pixels'] for n in names);summary=[]
    for rate in [.075,.15,.3]:
        folder=a.out/f'bpp-{rate:g}';folder.mkdir();budget=int(rate*pixels/8)
        chosen=[min(points[n],key=lambda r:r['bytes']) for n in names]
        size=len(package(chosen,0))
        if size>budget:
            summary.append(dict(target_bpp=rate,feasible=False,min_package_bytes=size,budget_bytes=budget));continue
        while True:
            best=None
            for i,current in enumerate(chosen):
                for candidate in points[names[i]]:
                    gain=current['lpips']-candidate['lpips'];cost=candidate['bytes']-current['bytes']
                    if gain<=0:continue
                    # Exact ZIP metadata varies slightly with factor; verify each accepted move.
                    if size+cost>budget+16:continue
                    utility=gain/max(cost,1)
                    if best is None or utility>best[0]:
                        trial=chosen.copy();trial[i]=candidate;trial_size=len(package(trial,0))
                        if trial_size<=budget:best=(utility,trial,trial_size)
            if best is None:break
            _,chosen,size=best
        for arm,mode in [('original',0),('perceptual',1)]:
            blob=package(chosen,mode);assert len(blob)==size and size<=budget
            (folder/f'{arm}.zip').write_bytes(blob);(folder/arm).mkdir()
            for r in chosen:
                shutil.copyfile(a.source/f'{arm}-factor{r["factor"]:g}'/'recon'/r['name'],folder/arm/r['name'])
        write_json(folder/'selection.json',[{k:r[k] for k in ['name','factor','bytes','pixels','stream_sha256']} for r in chosen])
        cfg=folder/'blind_config.json'
        write_json(cfg,dict(reference=str(a.source/'development32-full/images'),
            codecs={'original':str(folder/'original'),'perceptual':str(folder/'perceptual')}))
        sys.path.insert(0,str(ROOT/'src'));from clic27.blind_score import prepare
        prepare(cfg,folder/'study')
        metrics={arm:{k:sum((r if arm=='perceptual' else original[(r['name'],r['factor'])])[k] for r in chosen)/len(chosen)
            for k in ['psnr','lpips','dists']} for arm in ['original','perceptual']}
        summary.append(dict(target_bpp=rate,feasible=True,package_bytes=size,budget_bytes=budget,
            actual_package_bpp=size*8/pixels,images=len(chosen),metrics=metrics))
        write_json(a.out/'summary.json',summary)
    write_json(a.out/'summary.json',summary)
    write_json(a.out/'status.json',dict(state='complete',studies=sum(r['feasible'] for r in summary),
        caveat='Greedy LPIPS selection on development; paired same-stream control, not optimized mixed-codec comparison; package is experimental, not submission format'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
