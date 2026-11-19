from pathlib import Path
import shutil, subprocess, os, json
root=Path('/home/jovyan/clic2027-artifacts/mlicpp/innovation-20261105')
out=root/'smoke-guarded-entrypoint'
shutil.copytree(root/'smoke-train-stream-tto-parity',out,ignore=shutil.ignore_patterns('latent_tto'))
os.environ['CUDA_VISIBLE_DEVICES']='0'
for stage in ['latent_search','verify','summarize']:
    subprocess.run(['bash','experiments/mlicpp/run_innovation.sh','--stage',stage,'--out',str(out)],check=True)
print((out/'fresh.json').read_text())
