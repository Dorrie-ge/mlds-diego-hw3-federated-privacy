"""Run a predeclared validation-only sweep, then freeze choices before test access."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from federated import Config, aggregate_metrics, metrics, model, run

SEEDS = [17, 29, 43]
SCALES = [0., .05, .1, .3, 1.]


def train_if_needed(data, folder, cfg):
    if not (folder/'COMPLETE').exists():
        run(data,folder,cfg)
    if json.loads((folder/'config.json').read_text()) != asdict(cfg):
        raise ValueError('Saved configuration differs: '+str(folder))
    saved=json.loads((folder/'data_manifest.json').read_text())
    if saved != json.loads((Path(data)/'manifest.json').read_text()):
        raise ValueError('Saved dataset differs')
    return json.loads((folder/'history.json').read_text())[-1]['val_accuracy']


def suite(data, out, rounds):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(data)/'manifest.json').read_text())
    if manifest.get('synthetic') or len(manifest['clients'])!=100:
        raise ValueError('Full homework suite requires the course dataset with 100 users')
    baseline=[]
    for c in [.04,.1]:
        for e in [1,3]:
            names=[];scores=[]
            for seed in SEEDS:
                cfg=Config(seed=seed,rounds=rounds,fraction=c,epochs=e)
                name=f'fedavg_C{c}_E{e}_s{seed}';names.append(name)
                scores.append(train_if_needed(data,out/name,cfg))
            baseline.append({'fraction':c,'epochs':e,'mean_validation':float(np.mean(scores)),'runs':names})
    best=max(baseline,key=lambda r:r['mean_validation'])
    noise=[];ray_runs=[]
    for b in SCALES:
        names=[];scores=[]
        for seed in SEEDS:
            name=f'noise_b{b}_s{seed}';names.append(name)
            cfg=Config(seed=seed,rounds=rounds,epochs=best['epochs'],noise=b)
            scores.append(train_if_needed(data,out/name,cfg))
        noise.append({'scale':b,'mean_validation':float(np.mean(scores)),'runs':names})
    # Predeclared utility rule: largest nonzero b within 2 percentage points
    # of b=0 validation accuracy. A descriptive utility rule, not a DP guarantee.
    eligible=[r for r in noise if r['scale']>0 and r['mean_validation']>=noise[0]['mean_validation']-.02]
    choice=max(eligible,key=lambda r:r['scale'])['scale'] if eligible else None
    # One full Ray run; matched serial controls and separate synthetic parity
    # experiment validate orchestration. Serial/noise sweeps use three seeds.
    for seed in SEEDS[:1]:
        cfg=Config(seed=seed,rounds=rounds,fraction=.04,epochs=best['epochs'],backend='ray')
        name=f'ray_k4_s{seed}';train_if_needed(data,out/name,cfg);ray_runs.append(name)
    selection={'selection_uses':'final validation accuracy only, before any test evaluation',
               'baseline_sweep':baseline,'chosen_baseline':best,'noise_sweep':noise,
               'chosen_noise_scale':choice,'noise_utility_tolerance':.02,'ray_runs':ray_runs,
               'test_runs':best['runs']+sum([n['runs'] for n in noise],[])+ray_runs}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))


def test(data, out):
    out=Path(out);selection_path=out/'selection.json'
    selection=json.loads(selection_path.read_text())
    if (out/'heldout_test.json').exists():
        raise FileExistsError('Test results already exist; preserve the original evaluation')
    results={}
    for name in selection['test_runs']:
        m=model();checkpoint=out/name/'final.pt';m.load_state_dict(torch.load(checkpoint,weights_only=True))
        items=[];confusion=np.zeros((62,62),dtype=np.int64)
        for path in sorted((Path(data)/'test').glob('*.npz')):
            with np.load(path,allow_pickle=False) as d:
                items.append(metrics(m,torch.from_numpy(d['x']),torch.from_numpy(d['y']).long()))
                with torch.no_grad():
                    for start in range(0,len(d['y']),1024):
                        pred=m(torch.from_numpy(d['x'][start:start+1024])).argmax(1).numpy()
                        np.add.at(confusion,(d['y'][start:start+1024],pred),1)
        results[name]=aggregate_metrics(items)|{'n':sum(r['n'] for r in items),
                         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'confusion_matrix':confusion.tolist()}
    (out/'heldout_test.json').write_text(json.dumps({'frozen_selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                                                   'results':results},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','test']);p.add_argument('--data',required=True)
    p.add_argument('--out',default='results/course');p.add_argument('--rounds',type=int,default=100)
    a=p.parse_args()
    if a.action=='train':suite(a.data,a.out,a.rounds)
    else:test(a.data,a.out)
