"""Source-qualified glyph CNN, followed by actual-stream decoder diagnosis.

Test is opened once, after validation-only checkpoint selection. No model update
after test. A source-gate failure ends the pipeline before codec experiments.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe import ROOT, settings, sha, write_json

FONTS = Path('/home/jovyan/clic2027-envs/ocr/fonts')


def config(out):
    return json.loads((out/'config.json').read_text())


def build_reader(torch):
    n = torch.nn
    return n.Sequential(n.Conv2d(1,32,3,padding=1),n.ReLU(),n.MaxPool2d(2),
        n.Conv2d(32,64,3,padding=1),n.ReLU(),n.MaxPool2d(2),
        n.Conv2d(64,96,3,padding=1),n.ReLU(),n.MaxPool2d(2),n.Flatten(),
        n.Linear(96*6*6,128),n.ReLU(),n.Linear(128,4))


def crop(image, box):
    value = np.asarray(Image.fromarray(image).convert('L').crop(box).resize((48,48),Image.Resampling.BICUBIC),dtype=np.float32)
    # Fixed preprocessing shared by source and reconstructions, no labels.
    value = (value.mean()-value)/max(float(value.std()), 8.)
    return np.clip(value,-4,4)[None]


def render(font_path, size, contrast, texture, label, rng, lower):
    font = ImageFont.truetype(str(font_path),size)
    cell = int(np.ceil(max(font.getlength(c) for c in 'ABDEFG68')))+2
    x0 = int(rng.integers(8, max(9,248-5*cell)))
    y0 = int(rng.integers(144,193) if lower else rng.integers(24,113))
    bg = np.full((256,256,3),224,dtype=np.uint8)
    if texture:
        noise = Image.fromarray(rng.integers(0,25,(16,16),dtype=np.uint8)).resize((256,256),Image.Resampling.BICUBIC)
        bg = np.clip(bg.astype(np.int16)+np.asarray(noise,dtype=np.int16)[...,None]-12,0,255).astype(np.uint8)
    image = Image.fromarray(bg)
    draw = ImageDraw.Draw(image)
    for j,c in enumerate('AB'+label+'DE'):
        draw.text((x0+j*cell,y0),c,font=font,fill=(224-contrast,)*3,anchor='lt')
    box = [x0+2*cell-2,y0-2,x0+3*cell+2,y0+size+2]
    assert min(box)>=0 and max(box)<=256
    return np.asarray(image), box


def generate(out):
    c=config(out); rng=np.random.default_rng(c['seed'])
    fonts={f for split in ['train','validation','test'] for f in c[split+'_fonts']}
    assert all((FONTS/f).is_file() for f in fonts)
    assert not set(c['train_fonts']) & set(c['validation_fonts']+c['test_fonts'])
    rows=[]
    for split in ['train','validation','test']:
        (out/split/'source').mkdir(parents=True)
        grid=list(itertools.product(c[split+'_fonts'],c['sizes'],c['contrasts'],[False,True],c['classes']))
        # Full balanced grid on test; deterministic sampling for train/validation.
        if split=='test':
            assert len(grid)==c['test_images']
        features=[]; labels=[]
        for i in range(c[split+'_images']):
            f,size,contrast,texture,label=grid[i] if split=='test' else grid[int(rng.integers(len(grid)))]
            image,box=render(FONTS/f,size,contrast,texture,label,rng,split=='test')
            name=f'{i:04d}'; path=out/split/'source'/(name+'.png')
            Image.fromarray(image).save(path)
            features.append(crop(image,box)); labels.append(c['classes'].index(label))
            rows.append(dict(split=split,name=name,font=f,size=size,contrast=contrast,texture=texture,label=label,box=box,source_sha256=sha(path)))
        np.savez_compressed(out/split/'source_crops.npz',x=np.stack(features),y=np.asarray(labels,dtype=np.int64))
    write_json(out/'data.json',dict(rows=rows,fonts={f:sha(FONTS/f) for f in sorted(fonts)},
        limitation='Procedural glyphs, privileged boxes; font-family and upper/lower layout shift, not natural scene text.'))


def setup(out,stage):
    torch,install=settings()
    torch.manual_seed(config(out)['seed'])
    return torch,install(out/(stage+'.vram.jsonl'))


def train(out):
    torch,stop=setup(out,'reader'); c=config(out)
    model=build_reader(torch).cuda()
    def data(split):
        with np.load(out/split/'source_crops.npz') as a:
            return torch.from_numpy(a['x']).cuda(),torch.from_numpy(a['y']).cuda()
    x,y=data('train'); vx,vy=data('validation')
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    best=-1.; started=time.monotonic(); history=[]
    for epoch in range(c['epochs']):
        model.train(); order=torch.randperm(len(x),device='cuda')
        for idx in order.split(c['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[idx]),y[idx])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy=float((model(vx).argmax(1)==vy).float().mean())
        history.append(dict(epoch=epoch+1,validation_accuracy=accuracy,loss=float(loss),elapsed_s=time.monotonic()-started))
        if accuracy>best:
            best=accuracy
            torch.save(dict(model=model.state_dict(),epoch=epoch+1,validation_accuracy=accuracy),out/'reader.pt')
        write_json(out/'training.json',history)
        print(json.dumps(history[-1]),flush=True)
    write_json(out/'reader_selection.json',dict(best_validation_accuracy=best,selection='Maximum validation accuracy, earliest tie; no test read.'))
    stop.set()


def gate(out):
    torch,stop=setup(out,'gate'); c=config(out)
    model=build_reader(torch).cuda().eval()
    ck=torch.load(out/'reader.pt',map_location='cuda',weights_only=True); model.load_state_dict(ck['model'])
    rows=[]
    for split in ['validation','test']:
        with np.load(out/split/'source_crops.npz') as a:
            x=torch.from_numpy(a['x']).cuda(); y=a['y']
        with torch.no_grad():
            pred=torch.cat([model(v).argmax(1) for v in x.split(64)]).cpu().numpy()
        rows.append(dict(split=split,n=len(y),correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),predictions=pred.tolist()))
    passed=all(r['accuracy']>=c['source_accuracy_gate'] for r in rows)
    write_json(out/'source_gate.json',dict(passed=passed,threshold=c['source_accuracy_gate'],epoch=ck['epoch'],reader_sha256=sha(out/'reader.pt'),rows=rows))
    stop.set()


def encode(out):
    assert json.loads((out/'source_gate.json').read_text())['passed']
    torch,stop=setup(out,'codec')
    from projection_probe import arguments,adapter_model,uint8
    from probe import load
    from stream import pack
    c=config(out); model,meta=load(arguments(out),c['quality']); adapter=adapter_model(torch)
    reader=build_reader(torch).cuda().eval()
    reader.load_state_dict(torch.load(out/'reader.pt',map_location='cuda',weights_only=True)['model'])
    rows=[r for r in json.loads((out/'data.json').read_text())['rows'] if r['split']=='test']
    for folder in ['streams','base','perceptual']:
        (out/'test'/folder).mkdir()
    result=[]
    for i,row in enumerate(rows):
        path=out/'test/source'/(row['name']+'.png')
        assert sha(path)==row['source_sha256']
        original=np.asarray(Image.open(path).convert('RGB'))
        x=torch.from_numpy(original.copy()).permute(2,0,1)[None].float().cuda()/255
        model.update_resolutions(16,16)
        with torch.no_grad():
            encoded=model.compress(x)
        blob=pack(c['quality'],True,(256,256),tuple(encoded['shape']),encoded['strings'])
        (out/'test/streams'/(row['name']+'.bin')).write_bytes(blob)
        record=dict(name=row['name'],label=row['label'],bytes=len(blob),stream_sha256=hashlib.sha256(blob).hexdigest())
        for method in ['source','base','perceptual']:
            if method=='source':
                image=original
            else:
                hook=None
                if method=='perceptual':
                    hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,a:(adapter(a[0])[0],))
                model.update_resolutions(16,16)
                with torch.no_grad():
                    image=uint8(model.decompress(encoded['strings'],encoded['shape'])['x_hat'])
                if hook is not None: hook.remove()
                Image.fromarray(image).save(out/'test'/method/(row['name']+'.png'))
            with torch.no_grad():
                pred=int(reader(torch.from_numpy(crop(image,row['box'])[None]).cuda()).argmax(1))
            record[method+'_prediction']=c['classes'][pred]
            record[method+'_correct']=record[method+'_prediction']==row['label']
            record[method+'_pixel_sha256']=hashlib.sha256(image.tobytes()).hexdigest()
        result.append(record)
        write_json(out/'codec.progress.json',dict(images=i+1,total=len(rows)))
    write_json(out/'codec.json',dict(metadata=meta,rows=result))
    stop.set()


def verify(out):
    torch,stop=setup(out,'fresh')
    from projection_probe import arguments,adapter_model,uint8
    from probe import load
    from stream import unpack
    c=config(out); model,_=load(arguments(out),c['quality']); adapter=adapter_model(torch)
    rows=json.loads((out/'codec.json').read_text())['rows']; matches=0
    for row in rows:
        blob=(out/'test/streams'/(row['name']+'.bin')).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==row['stream_sha256']
        q,legacy,size,shape,strings=unpack(blob)
        assert q==c['quality'] and legacy and size==(256,256)
        for method in ['base','perceptual']:
            hook=None
            if method=='perceptual':
                hook=model.g_s.synthesis_transform[5].register_forward_pre_hook(lambda m,a:(adapter(a[0])[0],))
            model.update_resolutions(16,16)
            with torch.no_grad(): image=uint8(model.decompress(strings,shape)['x_hat'])
            if hook is not None: hook.remove()
            assert hashlib.sha256(image.tobytes()).hexdigest()==row[method+'_pixel_sha256']
            matches+=1
    write_json(out/'fresh.json',dict(matches=matches,source_images_not_used=True))
    stop.set()


def analyze(out):
    c=config(out); rows=json.loads((out/'codec.json').read_text())['rows']
    summary={m:dict(correct=sum(r[m+'_correct'] for r in rows),n=len(rows),accuracy=np.mean([r[m+'_correct'] for r in rows])) for m in ['source','base','perceptual']}
    # Independent generated images are the resampling unit; font coverage is tiny.
    delta=np.asarray([int(r['perceptual_correct'])-int(r['base_correct']) for r in rows])
    ix=np.random.default_rng(c['seed']).integers(0,len(rows),(10000,len(rows)))
    result=dict(summary=summary,actual_bpp=8*sum(r['bytes'] for r in rows)/(len(rows)*65536),
        harmed=sum(r['base_correct'] and not r['perceptual_correct'] for r in rows),
        recovered=sum(not r['base_correct'] and r['perceptual_correct'] for r in rows),
        accuracy_delta=float(delta.mean()),ci95_image_bootstrap=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),
        warning='Synthetic glyph classification, not human readability; no causal information-loss conclusion; one seed, one new font family.')
    write_json(out/'summary.json',result)


def run(out, config_path):
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_bytes(config_path.read_bytes())
    frozen={name:sha(Path(__file__).with_name(name)) for name in ['recognition_gate.py','projection_probe.py','probe.py','pilot_model.py','stream.py']}
    write_json(out/'sources.sha256.json',frozen)
    started=time.monotonic()
    for stage in ['generate','train','gate','encode','verify','analyze']:
        write_json(out/'status.json',dict(state='running',stage=stage))
        if time.monotonic()-started>config(out)['gpu_hour_ceiling']*3600:
            raise RuntimeError('Pilot wall-time budget exceeded')
        with (out/(stage+'.log')).open('w') as log:
            proc=subprocess.run([sys.executable,__file__,'--out',str(out),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,config(out)['gpu_hour_ceiling']*3600-(time.monotonic()-started)))
        if proc.returncode:
            write_json(out/'status.json',dict(state='failed',stage=stage,exit_code=proc.returncode))
            raise RuntimeError(stage)
        if stage=='gate' and not json.loads((out/'source_gate.json').read_text())['passed']:
            write_json(out/'status.json',dict(state='source_gate_failed',codec_not_run=True))
            return
    assert all(sha(Path(__file__).with_name(k))==v for k,v in frozen.items())
    write_json(out/'status.json',dict(state='complete',elapsed_s=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['run','generate','train','gate','encode','verify','analyze'],required=True)
    p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.stage=='run':
        if a.config is None: p.error('--config required')
        run(a.out,a.config)
    else: globals()[a.stage](a.out)
