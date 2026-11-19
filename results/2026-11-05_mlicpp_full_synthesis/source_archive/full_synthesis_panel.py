"""Show first three development images in manifest order, no metric selection."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch


def main(root):
    manifest=json.loads((root/'latents/manifest.json').read_text())
    rows=[r for r in manifest['records'] if r['split']=='validation'][:3]
    font=ImageFont.truetype('/home/jovyan/clic2027-envs/ocr/fonts/DejaVuSans.ttf',15)
    methods=[('source','Original'),('base','MLIC++ MSE'),('mean','Small adapter'),('reconstruction','Full synthesis'),('gan','Full + GAN')]
    panel=Image.new('RGB',(1328,920),'white');draw=ImageDraw.Draw(panel)
    for j,(_,label) in enumerate(methods): draw.text((8+264*j,8),label,fill='black',font=font)
    for i,row in enumerate(rows):
        sample=torch.load(root/'latents/validation'/(row['name']+'.pt'),weights_only=True)
        original=Image.fromarray(sample['original'][0].permute(1,2,0).numpy())
        for j,(method,_) in enumerate(methods):
            image=original if method=='source' else Image.open(root/'recon'/method/(row['name']+'.png')).convert('RGB')
            panel.paste(image,(8+264*j,38+i*294))
        draw.text((8,298+i*294),row['name']+' | fixed manifest order; development data',fill='black',font=font)
    (root/'figures').mkdir(exist_ok=True)
    panel.save(root/'figures/first3-validation.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
