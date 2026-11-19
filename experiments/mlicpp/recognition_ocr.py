"""Independent fixed OCR check for source-qualified reader pilot; no tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from PIL import Image

from probe import write_json

OCR=Path('/home/jovyan/clic2027-envs/ocr')


def main(root):
    if (root/'independent_ocr.json').exists():
        raise ValueError('Refuse to replace an OCR result')
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    rows=[r for r in json.loads((root/'data.json').read_text())['rows'] if r['split']=='test']
    env=dict(os.environ,OMP_THREAD_LIMIT='1',TESSDATA_PREFIX=str(OCR/'share/tessdata'))
    results=[]
    with tempfile.TemporaryDirectory(prefix='recognition-ocr-') as folder:
        path=Path(folder)/'glyph.png'
        for i,row in enumerate(rows):
            result=dict(name=row['name'],label=row['label'])
            for method in ['source','base','perceptual']:
                with Image.open(root/'test'/method/(row['name']+'.png')) as image:
                    value=image.convert('RGB').crop(row['box'])
                    value.resize((value.width*4,value.height*4),Image.Resampling.BICUBIC).save(path)
                proc=subprocess.run([str(OCR/'bin/tesseract'),str(path),'stdout','-l','eng','--psm','10','--oem','1','-c','tessedit_char_whitelist=68EF'],env=env,capture_output=True,text=True,check=True,timeout=30)
                recognized=re.sub('[^68EF]','',proc.stdout.upper())
                result[method+'_prediction']=recognized
                result[method+'_correct']=recognized==row['label']
            results.append(result)
            write_json(root/'ocr.progress.json',dict(images=i+1,total=len(rows)))
    write_json(root/'independent_ocr.json',dict(
        protocol='Tesseract eng psm10 oem1, whitelist68EF, generator bbox, 4x bicubic. Fixed before reading codec accuracy. No test adaptation.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        version=subprocess.check_output([str(OCR/'bin/tesseract'),'--version'],text=True).splitlines()[0],
        rows=results,summary={method:dict(n=len(results),correct=sum(r[method+'_correct'] for r in results)) for method in ['source','base','perceptual']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    if a.wait:
        deadline=time.monotonic()+7200
        while True:
            state=json.loads((a.root/'status.json').read_text())['state']
            if state=='complete': break
            if state in ['failed','source_gate_failed'] or time.monotonic()>deadline:
                raise RuntimeError('Upstream failed or wait budget exceeded')
            time.sleep(5)
    main(a.root)
