"""Wait for a completed CLIC budget study; compare to a frozen historical table."""
import argparse
import json
from pathlib import Path
import time
from probe import write_json

p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
a=p.parse_args();deadline=time.monotonic()+16*3600
while True:
    f=a.study/'status.json'
    state=json.loads(f.read_text()) if f.exists() else {}
    if state.get('state')=='complete':break
    if state.get('state')=='failed':raise RuntimeError('Source study failed; no comparison produced')
    if time.monotonic()>deadline:raise TimeoutError('Comparison waiting budget exhausted')
    time.sleep(30)
ref=json.loads(a.reference.read_text());records=json.loads((a.study/'input/manifest.json').read_text())['images']
assert sorted(r['name'] for r in records)==sorted(ref['image_names'])
rows=[]
for r in json.loads((a.study/'budgets/summary.json').read_text()):
    if not r['feasible']:continue
    old=ref['rates'][str(r['target_bpp'])];new=r['metrics']['perceptual']
    rows.append(dict(target_bpp=r['target_bpp'],new_bpp=r['actual_package_bpp'],historical_bpp=old['bpp'],
        new_lpips=new['lpips'],historical_lpips=old['lpips'],delta_lpips=new['lpips']-old['lpips'],
        new_dists=new['dists'],historical_dists=old['dists'],delta_dists=new['dists']-old['dists']))
write_json(a.study/'historical_mix_comparison.json',dict(rows=rows,reference=ref['source'],
    caveat='Historical metrics only; no fresh mixed-codec decode, no source-pixel hash parity audit. Different allocation objectives and overhead conventions; not exact equal-rate or human comparison.'))
