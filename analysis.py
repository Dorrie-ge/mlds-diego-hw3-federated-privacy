"""Compute report tables and diagnostic plots from retained experiment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from federated import model, perturb
from plot_style import apply_style
apply_style()


def load(path):
    return json.loads(Path(path).read_text())


def mean_sd(values):
    return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}


def analyze(data, root):
    data,root=Path(data),Path(root);course=root/'course';figs=course/'figures';figs.mkdir(exist_ok=True)
    selection=load(course/'selection.json');test=load(course/'heldout_test.json')
    assert test['frozen_selection_sha256']==hashlib.sha256((course/'selection.json').read_bytes()).hexdigest()
    out={'baseline':[],'noise':[]}
    for group,key in [('baseline_sweep','baseline'),('noise_sweep','noise')]:
        for row in selection[group]:
            finals=[load(course/name/'history.json')[-1] for name in row['runs']]
            out[key].append({k:v for k,v in row.items() if k!='runs'} |
                            {metric:mean_sd([v[metric] for v in finals]) for metric in ['train_accuracy','val_accuracy','val_client_p10']} |
                            {'test_accuracy':mean_sd([test['results'][n]['accuracy'] for n in row['runs']])} if key=='noise' or row==selection['chosen_baseline'] else
                           {k:v for k,v in row.items() if k!='runs'} |
                           {metric:mean_sd([v[metric] for v in finals]) for metric in ['train_accuracy','val_accuracy','val_client_p10']})
    # C/E comparison uses all three repetitions, sample SD bands, never test curves.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in selection['baseline_sweep']:
        histories=[load(course/n/'history.json') for n in row['runs']]
        for ax,metric in zip(axes,['train_accuracy','val_accuracy']):
            values=np.array([[h[metric] for h in hs] for hs in histories]);x=np.arange(values.shape[1]);mean=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot(x,mean,label=f"C={row['fraction']}, E={row['epochs']}");ax.fill_between(x,mean-sd,mean+sd,alpha=.12)
            ax.set(xlabel='Communication round',ylabel=metric.replace('_',' '));ax.legend(fontsize=8)
    fig.suptitle('Client participation and local epochs (3 seeds; bands = sample SD)');fig.tight_layout();fig.savefig(figs/'ce_comparison.png',dpi=180);plt.close(fig)
    # Matched serial/Ray run: same data, selected users, initialization, and seed.
    ray_name=selection['ray_runs'][0];ray_cfg=load(course/ray_name/'config.json')
    serial_name=f"fedavg_C0.04_E{ray_cfg['epochs']}_s{ray_cfg['seed']}"
    a=torch.load(course/ray_name/'final.pt',weights_only=True);b=torch.load(course/serial_name/'final.pt',weights_only=True)
    error=max((a[k]-b[k]).abs().max().item() for k in a)
    ra=load(course/ray_name/'history.json');sa=load(course/serial_name/'history.json')
    metricerr=max(abs(r[k]-s[k]) for r,s in zip(ra,sa) for k in ['train_loss','val_loss','train_accuracy','val_accuracy'])
    updates=[json.loads(x) for x in (course/ray_name/'client_updates.jsonl').read_text().splitlines()]
    overlaps=[]
    for r in range(1,len(ra)):
        us=[u for u in updates if u['round']==r];events=sorted([(u['started'],1) for u in us]+[(u['finished'],-1) for u in us]);active=peak=0
        for _,change in events:active+=change;peak=max(peak,active)
        overlaps.append(peak)
    out['ray']={'name':ray_name,'serial_name':serial_name,'max_parameter_error':error,'max_metric_error':metricerr,
                'peak_overlaps':overlaps,'test_accuracy':test['results'][ray_name]['accuracy'],'final':ra[-1]}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['val_loss','val_accuracy']):
        ax.plot([r['round'] for r in sa],[r[metric] for r in sa],label='Serial 4 clients');ax.plot([r['round'] for r in ra],[r[metric] for r in ra],ls='--',label='Ray 4 actors');ax.set(xlabel='Round',ylabel=metric);ax.legend()
    fig.tight_layout();fig.savefig(figs/'ray_parity.png',dpi=180);plt.close(fig)
    # Negative labels and extended local optimization: learning/control, not format tests.
    clean=load(course/'fedavg_C0.1_E3_s17'/'history.json');null=load(root/'course_controls/shuffled_labels/history.json')
    large=load(root/'course_controls/large_E10/history.json')
    out['controls']={'clean_final':clean[-1],'shuffled_final':null[-1],'large_E_final':large[-1],
                     'clean_E3_round40':clean[40],'zero_lr':load(root/'course_controls/summary.json')['zero_lr_validation_invariant']}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for hs,label in [(clean,'Real labels, E=3'),(null,'Permuted labels, E=3')]:axes[0].plot([r['round'] for r in hs],[r['val_accuracy'] for r in hs],label=label)
    for hs,label in [(clean[:41],'E=3'),(large,'E=10')]:axes[1].plot([r['round'] for r in hs],[r['val_accuracy'] for r in hs],label=label)
    for ax in axes:ax.set(xlabel='Round',ylabel='Clean validation accuracy');ax.legend()
    fig.tight_layout();fig.savefig(figs/'course_controls.png',dpi=180);plt.close(fig)
    # Sample imbalance, participation coverage, and noise boundary behavior.
    dist=load(data/'distribution.json');counts=np.asarray(list(dist['clients'].values()));overall=counts.sum(0)
    probabilities=overall/overall.sum();clientp=counts/counts.sum(1,keepdims=True);tv=.5*abs(clientp-probabilities).sum(1)
    out['data']={'clients':len(counts),'train_plus_val':int(overall.sum()),'class_counts':overall.tolist(),
                 'min_client_samples':int(counts.sum(1).min()),'max_client_samples':int(counts.sum(1).max()),
                 'min_class_samples':int(overall.min()),'max_class_samples':int(overall.max()),
                 'mean_client_total_variation':float(tv.mean()),'max_client_total_variation':float(tv.max())}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sizes=counts.sum(1);axes[0].scatter(sizes,tv);axes[0].set(xlabel='Local dataset samples',ylabel='Total variation from global label mix',title='Writer imbalance and label heterogeneity')
    participation=load(course/serial_name/'participation.json');p10=load(course/f"fedavg_C0.1_E{ray_cfg['epochs']}_s{ray_cfg['seed']}"/'participation.json')
    for ps,label in [(participation,'C=0.04'),(p10,'C=0.10')]:axes[1].hist(list(ps.values()),bins=15,alpha=.5,label=label)
    axes[1].set(xlabel='Rounds selected per writer',ylabel='Writers',title='Sampling opportunity after 120 rounds');axes[1].legend()
    out['participation']={label:{'min':min(ps.values()),'max':max(ps.values()),'unselected':sum(n==0 for n in ps.values())} for label,ps in [('C0.04',participation),('C0.10',p10)]}
    fig.tight_layout();fig.savefig(figs/'heterogeneity_coverage.png',dpi=180);plt.close(fig)
    first=np.load(data/'clients/client_000.npz')['x_train'][:64];noise_rows=[]
    fig,axes=plt.subplots(1,5,figsize=(11,2.5))
    for ax,scale in zip(axes,[0.,.05,.1,.3,1.]):
        noisy=perturb(torch.from_numpy(first),scale,123).numpy();noise_rows.append({'b':scale,'outside_unit_interval_fraction':float(((noisy<0)|(noisy>1)).mean()),'mean_absolute_noise':float(abs(noisy-first).mean())})
        ax.imshow(noisy[0].reshape(28,28),cmap='gray',vmin=0,vmax=1);ax.set_title(f'b={scale}');ax.axis('off')
    fig.suptitle('Display clipped to [0,1]; training values remain unclipped');fig.tight_layout();fig.savefig(figs/'noise_samples.png',dpi=180);plt.close(fig)
    out['noise_boundary']=noise_rows
    (root/'report_metrics.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--results',default='results')
    a=p.parse_args();analyze(a.data,a.results)
