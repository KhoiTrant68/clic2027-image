"""Run one bundled experiment in a consistent device/runtime environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--cpu',action='store_true')
p.add_argument('script',choices=['end_to_end','end_to_end_campaign','end_to_end_eval','cpu_development','cpu_rate_probe','cpu_perceptual_probe'])
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args();cfg=json.loads(a.config.read_text())
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='' if a.cpu else str(cfg['gpu_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
    TORCH_CUDNN_V8_API_DISABLED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
nvrtc=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cuda_nvrtc/lib'
env['LD_LIBRARY_PATH']=str(nvrtc)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
args=a.args[1:] if a.args[:1]==['--'] else a.args
if a.cpu and a.script in ['end_to_end','end_to_end_eval']:args+=['--device','cpu']
if a.cpu and a.script=='end_to_end_campaign':raise ValueError('Full campaign requires GPU')
raise SystemExit(subprocess.call([sys.executable,str(ROOT/'experiments/mlicpp'/f'{a.script}.py'),
    '--config',str(a.config.resolve())]+args,env=env,cwd=ROOT))
