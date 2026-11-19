"""Plot the pre-specified paired contrasts, including null/negative results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    analysis=json.loads((root/'analysis.json').read_text())
    pairs=[('long_s160','reference_s40','Combined: train + search'),
           ('long_s40','reference_s40','Train 5000 vs 1000 (search 40)'),
           ('reference_s160','reference_s40','Search 160 vs 40 (train 1000)'),
           ('long_s160','long_s40','Search 160 vs 40 (train 5000)'),
           ('long_s160','reference_s160','Train 5000 vs 1000 (search 160)')]
    lookup={(r['treatment'],r['control'],r['metric']):r for r in analysis['comparisons']}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,metric in zip(axes,['lpips','dists']):
        for i,(t,c,_) in enumerate(pairs):
            row=lookup[t,c,metric];low,high=row['ci95'];value=row['delta']
            color='#136f63' if i==0 else '#415a77'
            ax.plot([low,high],[i,i],color=color,linewidth=2)
            ax.scatter(value,i,color=color,s=45,zorder=3)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_xlabel(f'{metric.upper()} difference (left = improvement)')
        ax.grid(axis='x',alpha=.2);ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p[2] for p in pairs]);ax.invert_yaxis()
    # Shared y-axis limits invert together; set a single explicit order.
    axes[0].set_ylim(len(pairs)-.5,-.5)
    fig.suptitle('Fixed correction: paired quality changes at identical packet size',fontsize=13)
    fig.text(.5,.015,f"{analysis['test_sources']} new source images; pointwise 95% paired bootstrap CIs; one seed / q2 / 256 x 256 crops",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    (root/'figures').mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(root/'figures'/('paired-contrasts.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
