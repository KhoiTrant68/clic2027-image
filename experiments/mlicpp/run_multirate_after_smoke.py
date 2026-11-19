"""Start the authorized pilot only after all smoke decode checks pass."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from probe import write_json

p=argparse.ArgumentParser()
for name in ['config','smoke','out']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();deadline=time.monotonic()+1800;status=a.out.parent/'launch_status.json'
try:
    while True:
        s=json.loads((a.smoke/'status.json').read_text())
        if s['state']=='complete':
            assert s['smoke'] and s['fresh_matches']==9
            break
        if s['state']=='failed':raise RuntimeError('Smoke failed; main training not started')
        if time.monotonic()>deadline:raise TimeoutError('Smoke wait exceeded 30 minutes')
        write_json(status,dict(state='waiting_for_smoke',smoke=s));time.sleep(10)
    write_json(status,dict(state='running',out=str(a.out)))
    with (a.out.parent/'main.log').open('a',buffering=1) as log:
        subprocess.run([sys.executable,str(Path(__file__).with_name('multirate_synthesis.py')),
            '--config',str(a.config),'--out',str(a.out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(status,dict(state='complete',out=str(a.out)))
except BaseException as exc:
    write_json(status,dict(state='failed',error=repr(exc)));raise
