"""Standalone RD curves, real total bpp including oracle side information."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]); rows=json.loads((out/'analysis.json').read_text())['means']
fig, axes=plt.subplots(1,3,figsize=(13,3.9),constrained_layout=True)
for mode in ['uniform','global','predicted','oracle']:
    data=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['bpp'])
    for ax,metric,title in zip(axes,['pooled_psnr','lpips','dists'],['PSNR (pooled MSE) ↑','LPIPS-Alex ↓','DISTS ↓']):
        ax.plot([r['bpp'] for r in data],[r[metric] for r in data],'-o',label=mode,markersize=4)
        ax.set(xlabel='Actual total bpp',ylabel=title); ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('MLIC++ q2 geometry pilot · 16 held-out 256×256 crops\nFrozen backbone; oracle includes 1,280 bytes/image; not a full-resolution benchmark',fontsize=11)
folder=out/'figures';folder.mkdir(exist_ok=True)
fig.savefig(folder/'rate-distortion.png',dpi=180);fig.savefig(folder/'rate-distortion.pdf')
