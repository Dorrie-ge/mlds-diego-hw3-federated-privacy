"""Required curves and update-alignment diagnostic; never manufactures results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_style
apply_style()


def plots(data, results):
    results=Path(results);out=results/'figures';out.mkdir(exist_ok=True)
    dist=json.loads((Path(data)/'distribution.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    rows=[('Overall',dist['overall'])]+list(dist['clients'].items())[:5]
    for ax,(name,counts) in zip(axes.flat,rows):
        ax.bar(np.arange(62),counts);ax.set(title=name,ylabel='Samples',xlabel='Class ID (0-61)')
    fig.tight_layout();fig.savefig(out/'class_distributions.png',dpi=180);plt.close(fig)
    for folder in sorted(results.iterdir()):
        if not (folder/'history.json').exists():continue
        history=json.loads((folder/'history.json').read_text());rounds=[r['round'] for r in history]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,metric in zip(axes,['loss','accuracy']):
            for split in ['train','val']:
                ax.plot(rounds,[r[f'{split}_{metric}'] for r in history],label=split)
            ax.set(xlabel='Communication round',ylabel=metric,title=folder.name);ax.legend()
        fig.tight_layout();fig.savefig(out/f'{folder.name}_curves.png',dpi=150);plt.close(fig)
        if (folder/'client_updates.jsonl').exists():
            updates=[json.loads(x) for x in (folder/'client_updates.jsonl').read_text().splitlines()]
            fig,ax=plt.subplots(figsize=(7,4))
            points=ax.scatter([u['round'] for u in updates],[u['alignment'] for u in updates],
                             c=[u['delta_norm'] for u in updates],s=[20+500*u['weight'] for u in updates],alpha=.7)
            ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Communication round',ylabel='Cosine(client update, global update)',title='Client agreement and influence: '+folder.name)
            fig.colorbar(points,ax=ax,label='Client update L2 norm');fig.tight_layout()
            fig.savefig(out/f'{folder.name}_alignment.png',dpi=150);plt.close(fig)
    selection=json.loads((results/'selection.json').read_text())
    tests=json.loads((results/'heldout_test.json').read_text())['results']
    fig,ax=plt.subplots(figsize=(7,4))
    for metric in ['train','test']:
        means=[];stds=[]
        for row in selection['noise_sweep']:
            scores=[]
            for name in row['runs']:
                scores.append(tests[name]['accuracy'] if metric=='test' else json.loads((results/name/'history.json').read_text())[-1]['train_accuracy'])
            means.append(np.mean(scores));stds.append(np.std(scores,ddof=1))
        ax.errorbar([r['scale'] for r in selection['noise_sweep']],means,yerr=stds,marker='o',capsize=3,label=metric+' (mean +/- sample SD, 3 seeds)')
    ax.set(xlabel='Laplace noise scale b on [0,1] pixels',ylabel='Accuracy',title='Noise and classification utility');ax.legend();fig.tight_layout()
    fig.savefig(out/'noise_utility.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--results',default='results/course')
    a=p.parse_args();plots(a.data,a.results)
