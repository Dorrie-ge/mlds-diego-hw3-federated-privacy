"""Counterfactual controls on course training/validation data. Never opens test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from federated import Config, run


def controls(data, out):
    data=Path(data).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Fixed 120-round null control preserves each client's class imbalance but
    # destroys its image-label correspondence. Validation labels remain intact.
    with tempfile.TemporaryDirectory() as td:
        fixture=Path(td);(fixture/'clients').mkdir()
        manifest=json.loads((data/'manifest.json').read_text())
        manifest['intervention']='independent within-client permutation of training labels'
        for i,u in enumerate(manifest['clients']):
            with np.load(data/'clients'/f'{u}.npz') as d:
                arrays={k:d[k] for k in d.files}
            arrays['y_train']=np.random.default_rng(3000+i).permutation(arrays['y_train'])
            np.savez(fixture/'clients'/f'{u}.npz',**arrays)
        (fixture/'manifest.json').write_text(json.dumps(manifest))
        run(fixture,out/'shuffled_labels',Config(rounds=120,epochs=3))
    # Stress the local-update horizon while holding schedule, model seed and C fixed.
    run(data,out/'large_E10',Config(rounds=40,epochs=10))
    run(data,out/'zero_lr',Config(rounds=3,epochs=3,lr=0.))
    h=json.loads((out/'zero_lr'/'history.json').read_text())
    assert len(set(r['val_loss'] for r in h))==1
    assert len(set(r['val_accuracy'] for r in h))==1
    summary={'negative_control':'training labels shuffled within each client; validation unchanged',
             'shuffled_final':json.loads((out/'shuffled_labels'/'history.json').read_text())[-1],
             'large_E_final':json.loads((out/'large_E10'/'history.json').read_text())[-1],
             'zero_lr_validation_invariant':True,'test_data_used':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='results/course_controls')
    a=p.parse_args();controls(a.data,a.out)
