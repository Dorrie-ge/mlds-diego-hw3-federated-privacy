"""Verify b=0 equals clean FedAvg using retained/retrained paired checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def load(path):
    return json.loads(Path(path).read_text())


def validate(course,output):
    course=Path(course)
    selection=load(course/'selection.json')
    zero=next(row for row in selection['noise_sweep'] if row['scale']==0)
    errors=[];rows=[];seeds=[]
    for noisy_name in zero['runs']:
        noisy=course/noisy_name;cfg=load(noisy/'config.json')
        clean_name=f"fedavg_C{cfg['fraction']}_E{cfg['epochs']}_s{cfg['seed']}"
        clean=course/clean_name
        assert (noisy/'COMPLETE').exists() and (clean/'COMPLETE').exists()
        assert cfg['noise']==0 and cfg==load(clean/'config.json')
        assert load(noisy/'data_manifest.json')==load(clean/'data_manifest.json')
        paths=[clean/'final.pt',noisy/'final.pt']
        if not all(p.exists() for p in paths):
            raise FileNotFoundError('Checkpoint files are excluded from GitHub. Use the local submission ZIP or reproduce experiments.py train with the course data.')
        a,b=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        assert set(a)==set(b)
        for key in a:
            assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype
            assert torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all()
        error=max((a[k]-b[k]).abs().max().item() for k in a)
        assert error==0, f'b=0 checkpoint differs from clean FedAvg: {error}'
        histories=[load(folder/'history.json') for folder in [clean,noisy]]
        assert len(histories[0])==len(histories[1])==cfg['rounds']+1
        keys=['train_loss','val_loss','train_accuracy','val_accuracy',
              'train_clean_loss','train_clean_accuracy','val_client_p10']
        metric_error=max(abs(x[k]-y[k]) for x,y in zip(*histories) for k in keys)
        assert metric_error==0
        seeds.append(cfg['seed']);errors.append(error)
        rows.append({'seed':cfg['seed'],'clean_run':clean_name,'zero_noise_run':noisy_name,
                     'rounds_compared':len(histories[0]),'max_parameter_error':error,
                     'max_history_metric_error':metric_error,
                     'checkpoint_sha256':{p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    result={'b_zero_matches_clean_baseline_exactly':True,'max_parameter_errors_by_seed':errors,
            'seeds':seeds,'test_data_used':False,'checks':rows,
            'history_metrics_ignore':'elapsed wall-clock seconds; compare round 0 through final round'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--course',default='results/course')
    parser.add_argument('--out',default='results/validation/course_crosschecks.json')
    args=parser.parse_args();validate(args.course,args.out)
