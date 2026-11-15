"""Algorithm experiments with known predictions; synthetic data never count as EMNIST."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from federated import Config, Client, fedavg, model, perturb, run, state_copy


def make_fixture(root):
    (root/'clients').mkdir(parents=True)
    ids=[]
    for i in range(100):
        rng=np.random.default_rng(i)
        y=np.arange(40)%2
        x=np.zeros((40,784),np.float32)
        x[np.arange(40),y]=1.
        # Same easy two-class task, independent per-client permutations.
        idx=rng.permutation(40);key=f'client_{i:03d}';ids.append(key)
        np.savez(root/'clients'/f'{key}.npz',x_train=x[idx[:32]],y_train=y[idx[:32]],
                 x_val=x[idx[32:]],y_val=y[idx[32:]])
    (root/'manifest.json').write_text(json.dumps({'clients':ids,'synthetic':True}))
    (root/'test').mkdir()
    # Any accidental test read during training will fail.
    (root/'test'/'DO_NOT_OPEN.npz').write_text('Not a valid archive; training must not open this.')


def max_error(a,b):
    return max((a[k]-b[k]).abs().max().item() for k in a)


def validate(output, use_ray=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(1)
    result={'data_kind':'synthetic algorithm controls, NOT course EMNIST results'}
    # Known unequal-size weighted mean: (1*2+3*10)/4 = 8, not 6.
    got=fedavg([{'w':torch.tensor([2.])},{'w':torch.tensor([10.])}],[1,3])['w'].item()
    assert got==8.;result['unequal_client_weighting']={'observed':got,'expected':8.,'incorrect_unweighted':6.}
    # One full-batch local SGD step must equal a pooled gradient step.
    x=torch.randn(11,784);y=torch.arange(11)%4
    initial=state_copy(model());states=[]
    for sl in [slice(0,3),slice(3,11)]:
        m=model();m.load_state_dict(initial);o=torch.optim.SGD(m.parameters(),lr=.02)
        o.zero_grad();nn.functional.cross_entropy(m(x[sl]),y[sl]).backward();o.step();states.append(state_copy(m))
    pooled=model();pooled.load_state_dict(initial);o=torch.optim.SGD(pooled.parameters(),lr=.02)
    o.zero_grad();nn.functional.cross_entropy(pooled(x),y).backward();o.step()
    error=max_error(fedavg(states,[3,8]),state_copy(pooled));assert error<1e-6
    result['full_batch_gradient_identity']={'max_abs_error':error,'tolerance':1e-6}
    samples=perturb(torch.zeros(100000,1),.2,5).numpy().ravel()
    mean,var=float(samples.mean()),float(samples.var())
    assert abs(mean)<.005 and abs(var-.08)<.004
    result['laplace_moments']={'sample_size':len(samples),'mean':mean,'variance':var,
                               'expected_mean':0.,'expected_variance':.08}
    zero=perturb(x,0,99);assert torch.equal(zero,x)
    result['b_zero_identity']=True
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);data=root/'fixture';make_fixture(data)
        cfg=Config(rounds=10,epochs=3,lr=.3)
        client=Client(data/'clients'/'client_000.npz',asdict(cfg))
        no_steps=Client(client.path,asdict(Config(epochs=0))).update(initial,1)['state']
        zero_lr=Client(client.path,asdict(Config(lr=0))).update(initial,1)['state']
        assert max_error(no_steps,initial)==0 and max_error(zero_lr,initial)==0
        result['zero_updates']={'E_zero_error':max_error(no_steps,initial),'lr_zero_error':max_error(zero_lr,initial)}
        history=run(data,root/'control',cfg)
        assert history[-1]['val_accuracy']>.95 and history[-1]['val_loss']<history[0]['val_loss']
        result['separable_task']={'initial':history[0],'final':history[-1],
                                  'prediction':'accuracy > .95 and lower validation loss after 10 rounds'}
        result['heldout_test_not_read']=True
        (output/'synthetic_control_history.json').write_text(json.dumps(history,indent=2))
        if use_ray:
            cfg=Config(rounds=1,fraction=.04,epochs=2,seed=19)
            run(data,root/'serial',cfg)
            cfg.backend='ray';run(data,root/'parallel',cfg)
            a=torch.load(root/'serial'/'final.pt',weights_only=True)
            b=torch.load(root/'parallel'/'final.pt',weights_only=True)
            error=max_error(a,b);assert error<1e-6
            result['ray_serial_equivalence']={'max_abs_error':error,'tolerance':1e-6,'selected_clients':4}
            updates=[json.loads(line) for line in (root/'parallel'/'client_updates.jsonl').read_text().splitlines()]
            events=sorted([(u['started'],1) for u in updates]+[(u['finished'],-1) for u in updates])
            active=peak=0
            for _,change in events:active+=change;peak=max(peak,active)
            result['ray_observed_peak_overlapping_updates']=peak
        else:
            result['ray_serial_equivalence']={'status':'not run; use --ray'}
    (output/'algorithm_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/validation');p.add_argument('--ray',action='store_true')
    a=p.parse_args();validate(a.out,a.ray)
