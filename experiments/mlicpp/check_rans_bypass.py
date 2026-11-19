"""Bounded native-coder regression probe; each child has an eight-second timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(out):
    rows=[]
    for symbol in [0,-138273120]:
        started=time.monotonic()
        try:
            p=subprocess.run([sys.executable,__file__,'--symbol',str(symbol)],capture_output=True,text=True,timeout=8)
            row=dict(symbol=symbol,timed_out=False,exit_code=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr[-1000:])
        except subprocess.TimeoutExpired:
            row=dict(symbol=symbol,timed_out=True,exit_code=None)
        row['seconds']=time.monotonic()-started;rows.append(row)
    out.write_text(json.dumps(dict(rows=rows,note='Native installed binary, synthetic single-symbol input; no GPU or image data.'),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--symbol',type=int);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.symbol is not None:
        import compressai
        from compressai.ans import BufferedRansEncoder
        e=BufferedRansEncoder();e.encode_with_indexes([a.symbol],[0],[[0,32768,65536]],[3],[0])
        print(json.dumps(dict(version=compressai.__version__,bytes=len(e.flush()))),flush=True)
    elif a.out:main(a.out)
    else:p.error('--symbol or --out required')
