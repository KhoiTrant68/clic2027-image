"""Fixed first-three test examples, no selection by gain or failure."""
import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont
import torch


def main(root):
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    rows=[r for r in json.loads((root/'cache/manifest.json').read_text())['records'] if r['split']=='test'][:3]
    methods=[('source','Original'),('base','Base decoder'),('conditional','Conditional'),('fixed','Fixed basis'),('pixel','Pixel DCT'),('latent_tto','Latent TTO')]
    measurements={(r['method'],r['name']):r for r in json.loads((root/'test_metrics.json').read_text())}
    font=ImageFont.truetype('/home/jovyan/clic2027-envs/ocr/fonts/DejaVuSans.ttf',15)
    panel=Image.new('RGB',(1592,920),'white');draw=ImageDraw.Draw(panel)
    for j,(_,label) in enumerate(methods):draw.text((8+264*j,8),label,fill='black',font=font)
    for i,row in enumerate(rows):
        value=torch.load(root/'cache/test'/(row['name']+'.pt'),weights_only=True)
        original=Image.fromarray(value['original'][0].permute(1,2,0).numpy())
        for j,(method,_) in enumerate(methods):
            image=original if method=='source' else Image.open(root/'recon'/method/(row['name']+'.png')).convert('RGB')
            panel.paste(image,(8+264*j,38+i*294))
            caption=row['name'] if method=='source' else str(measurements[method,row['name']]['bytes'])+' actual bytes'
            draw.text((8+264*j,298+i*294),caption,fill='black',font=font)
    (root/'figures').mkdir(exist_ok=True)
    panel.save(root/'figures/first3-test.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
