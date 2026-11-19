"""Full-image native-byte evaluation and fresh-process decode of one checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from probe import settings,load,write_json,sha,ROOT
from end_to_end import Steps
from geometry_codec import Quantizer
from stream import pack,unpack


def run(a):
    torch,_=settings()
    if a.device=='cpu':torch.cuda.synchronize=lambda *args,**kwargs:None
    else:torch.cuda.set_per_process_memory_fraction(.70)
    c=SimpleNamespace(**json.loads(a.config.read_text()))
    model,meta=load(SimpleNamespace(upstream=Path(c.upstream),weights=Path(c.weights),legacy=True,device=a.device),a.quality)
    q=None
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['state_dict'])
        if ck['log_step'] is not None:
            q=Steps(model,torch,False);q.log_step.data.copy_(ck['log_step']);q.reset()
    if a.quantizer_factor!=1:
        if q is not None:raise ValueError('Do not combine learned steps and scalar sweep')
        if not 0<a.quantizer_factor<=16:raise ValueError('Invalid quantizer factor')
        q=Quantizer(model,None,None);q.select('uniform',a.quantizer_factor)
    model.eval();model.update(force=True)
    if a.synthesis:
        model.g_s.load_state_dict(torch.load(a.synthesis,map_location='cpu',weights_only=True)['synthesis'])
    if a.fresh:
        rows=json.loads((a.out/'metrics.json').read_text())
        with torch.no_grad():
            for r in rows:
                path=a.out/'streams'/r['stream'];assert sha(path)==r['stream_sha256']
                _,_,(h,w),shape,strings=unpack(path.read_bytes())
                if isinstance(q,Steps):q.reset()
                value=model.decompress(strings,shape)['x_hat'][:,:,:h,:w]
                im=value[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
        write_json(a.out/'fresh.json',dict(passed=len(rows)));return
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'streams').mkdir();(a.out/'recon').mkdir()
    write_json(a.out/'decoder.json',dict(backbone=meta,checkpoint=None if a.checkpoint is None else str(a.checkpoint),
        checkpoint_sha256=None if a.checkpoint is None else sha(a.checkpoint),
        learned_steps='shared model parameters; not per-image side information',quantizer_factor=a.quantizer_factor,
        padding='replicate to multiple of 64',device=a.device,
        synthesis_sha256=None if a.synthesis is None else sha(a.synthesis)))
    sys.path.insert(0,str(ROOT/'src'));from clic27.eval.metrics import Perceptual
    metric=Perceptual(a.device)
    manifest=json.loads((a.data/'manifest.json').read_text());rows=[]
    with torch.no_grad():
        for record in manifest['images']:
            path=a.data/'images'/record['name'];assert sha(path)==record['sha256']
            image=np.asarray(Image.open(path).convert('RGB')).copy();h,w=image.shape[:2]
            x=torch.from_numpy(image).permute(2,0,1)[None].float().to(a.device)/255
            x=torch.nn.functional.pad(x,(0,(-w)%64,0,(-h)%64),mode='replicate')
            torch.cuda.synchronize();start=time.monotonic();encoded=model.compress(x)
            torch.cuda.synchronize();enc_time=time.monotonic()-start
            blob=pack(a.quality,True,(h,w),tuple(encoded['shape']),encoded['strings'])
            start=time.monotonic();value=model.decompress(encoded['strings'],encoded['shape'])['x_hat'][:,:,:h,:w]
            torch.cuda.synchronize();dec_time=time.monotonic()-start
            reconstructed=value[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
            stream=Path(record['name']).stem+'.bin';(a.out/'streams'/stream).write_bytes(blob)
            Image.fromarray(reconstructed).save(a.out/'recon'/record['name'])
            rows.append(dict(name=record['name'],pixels=h*w,bytes=len(blob),bpp=len(blob)*8/(h*w),
                stream=stream,stream_sha256=hashlib.sha256(blob).hexdigest(),pixel_sha256=hashlib.sha256(reconstructed.tobytes()).hexdigest(),
                encode_s=enc_time,decode_s=dec_time,**metric.all(image,reconstructed)))
            write_json(a.out/'metrics.json',rows)
    if q:q.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--quality',type=int,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--fresh',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--quantizer-factor',type=float,default=1.)
    p.add_argument('--synthesis',type=Path)
    run(p.parse_args())
