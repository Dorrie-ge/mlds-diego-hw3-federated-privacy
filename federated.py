"""FedAvg primitives. No federated-learning library; only model states leave clients."""
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import platform
import numpy as np
import torch
from torch import nn


def seed_for(seed, *parts):
    token = ':'.join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')


def model():
    # Two affine layers; one hidden layer of 128 ReLU units, 62 output logits.
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 62))


def state_copy(m):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in m.state_dict().items())


def fedavg(states, counts):
    if not states or len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError('Aggregation requires a positive sample count per client')
    if any(list(s) != list(states[0]) for s in states):
        raise ValueError('Mismatched model keys')
    total = sum(counts)
    return OrderedDict((k, sum(s[k].double() * (n / total)
                              for s, n in zip(states, counts)).to(states[0][k].dtype))
                       for k in states[0])


def perturb(x, scale, seed):
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('Noise scale must be finite and nonnegative')
    if scale == 0:
        return x.clone()
    noise = np.random.default_rng(seed).laplace(0, scale, size=tuple(x.shape))
    # Unclipped additive noise, as specified in the assignment.
    return x + torch.as_tensor(noise, dtype=x.dtype)


@dataclass
class Config:
    seed: int = 17
    rounds: int = 100
    fraction: float = .1
    epochs: int = 1
    lr: float = .05
    batch_size: int = 32
    noise: float = 0.
    backend: str = 'serial'
    device: str = 'cpu'
    cpus: int = 4
    gpu_per_actor: float = 0.

    def validate(self, n):
        if not 0 < self.fraction <= .1 or int(n * self.fraction) < 1:
            raise ValueError('C must be in (0, .1] and select at least one client')
        if self.rounds < 1 or self.epochs < 0 or self.batch_size < 1 or self.lr < 0:
            raise ValueError('Invalid optimizer/round settings')
        if self.noise < 0 or not np.isfinite(self.noise):
            raise ValueError('Invalid noise scale')
        if self.backend not in ('serial', 'ray') or not 0 <= self.gpu_per_actor <= 1:
            raise ValueError('Invalid backend/resources')
        if self.backend == 'ray' and self.cpus < 4:
            raise ValueError('Ray experiment requires resources for four 1-CPU actors')


class Client:
    def __init__(self, path, cfg):
        torch.set_num_threads(1)
        self.path, self.cfg = str(path), Config(**cfg)
        self.id = Path(path).stem
        # A client reads only its own partition. Never loads held-out test data.
        with np.load(path, allow_pickle=False) as d:
            self.x = torch.from_numpy(d['x_train'].copy())
            self.y = torch.from_numpy(d['y_train'].copy()).long()
            self.vx = torch.from_numpy(d['x_val'].copy())
            self.vy = torch.from_numpy(d['y_val'].copy()).long()
        self.noisy = perturb(self.x, self.cfg.noise,
                             seed_for(self.cfg.seed, self.id, 'fixed-noise'))

    def ready(self):
        return len(self.y)

    def update(self, state, round_id):
        started = time.time()
        cfg = self.cfg
        torch.manual_seed(seed_for(cfg.seed, self.id, round_id))
        device = cfg.device
        m = model().to(device)
        m.load_state_dict(state)
        opt = torch.optim.SGD(m.parameters(), lr=cfg.lr)
        gen = torch.Generator().manual_seed(seed_for(cfg.seed, self.id, round_id, 'batches'))
        steps = 0
        for _ in range(cfg.epochs):
            for indices in torch.randperm(len(self.y), generator=gen).split(cfg.batch_size):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(self.noisy[indices].to(device)),
                                                   self.y[indices].to(device))
                loss.backward()
                opt.step()
                steps += 1
        return {'state': state_copy(m), 'n': len(self.y), 'client': self.id,
                'steps': steps, 'started': started, 'finished': time.time()}

    def evaluate(self, state, split):
        m = model()
        m.load_state_dict(state)
        if split == 'val':
            x, y = self.vx, self.vy
        elif split == 'train':
            x, y = self.noisy, self.y
        elif split == 'train_clean':
            x, y = self.x, self.y
        else:
            raise ValueError(split)
        return metrics(m, x, y)

    def evaluate_all(self, state):
        m=model();m.load_state_dict(state)
        train=metrics(m,self.noisy,self.y)
        return {'train':train,'val':metrics(m,self.vx,self.vy),
                'train_clean':metrics(m,self.x,self.y) if self.cfg.noise else train}


def metrics(m, x, y, batch=1024):
    m.eval()
    loss, correct = 0., 0
    with torch.no_grad():
        for start in range(0, len(y), batch):
            logits = m(x[start:start+batch])
            labels = y[start:start+batch]
            loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(1) == labels).sum().item()
    return {'loss_sum': loss, 'correct': correct, 'n': len(y)}


def aggregate_metrics(items):
    n = sum(d['n'] for d in items)
    if n == 0:
        raise ValueError('Empty evaluation split')
    return {'loss': sum(d['loss_sum'] for d in items) / n,
            'accuracy': sum(d['correct'] for d in items) / n}


def run(data, output, cfg):
    data, output = Path(data).resolve(), Path(output)
    manifest = json.loads((data/'manifest.json').read_text())
    paths = [data/'clients'/f'{u}.npz' for u in manifest['clients']]
    cfg.validate(len(paths))
    if (output/'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    (output/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    provenance={'started_unix':time.time(),'platform':platform.platform(),
                'torch':torch.__version__,'numpy':np.__version__,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'runtime.json').write_text(json.dumps(provenance,indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    state = state_copy(model())
    rng = np.random.default_rng(seed_for(cfg.seed, 'schedule'))
    history, participation = [], np.zeros(len(paths), dtype=int)
    # Independent client objects retain only their own data. Ray training actors
    # remain ephemeral; these local evaluators are the single-machine simulation.
    evaluators = [Client(path, asdict(cfg)) for path in paths]
    ray = None
    if cfg.backend == 'ray':
        import ray
        if cfg.gpu_per_actor and cfg.device != 'cuda':
            raise ValueError('GPU actors require device=cuda')
        ray.init(num_cpus=cfg.cpus, include_dashboard=False, log_to_driver=False)
        RemoteClient = ray.remote(num_cpus=1, num_gpus=cfg.gpu_per_actor)(Client)
    try:
        for r in range(cfg.rounds+1):
            round_started=time.time()
            if r:
                k = 4 if ray else int(len(paths) * cfg.fraction)
                selected = rng.choice(len(paths), k, replace=False)
                participation[selected] += 1
                if ray:
                    actors = [RemoteClient.remote(str(paths[i]), asdict(cfg)) for i in selected]
                    try:
                        ray.get([a.ready.remote() for a in actors])
                        shared = ray.put(state)
                        updates = ray.get([a.update.remote(shared, r) for a in actors])
                    finally:
                        for a in actors:
                            ray.kill(a)
                else:
                    updates = [evaluators[i].update(state, r) for i in selected]
                old = torch.cat([v.flatten() for v in state.values()])
                new = fedavg([u['state'] for u in updates], [u['n'] for u in updates])
                newvec = torch.cat([v.flatten() for v in new.values()])
                # Non-standard diagnostic: client update alignment and influence.
                with (output/'client_updates.jsonl').open('a') as f:
                    for u in updates:
                        delta = torch.cat([v.flatten() for v in u['state'].values()]) - old
                        global_delta = newvec - old
                        alignment = nn.functional.cosine_similarity(delta[None], global_delta[None]).item()
                        f.write(json.dumps({k:v for k,v in u.items() if k!='state'} |
                                           {'round':r, 'delta_norm':delta.norm().item(),
                                            'alignment':alignment,
                                            'weight':u['n']/sum(v['n'] for v in updates)})+'\n')
                state = new
            row = {'round':r}
            per_client = []
            summaries = [client.evaluate_all(state) for client in evaluators]
            for split in ['train', 'val', 'train_clean']:
                # Evaluation remains local; server receives scalar sums, not images.
                items = [summary[split] for summary in summaries]
                row.update({split+'_'+key:value for key,value in aggregate_metrics(items).items()})
                if split=='val':
                    per_client = [item['correct']/item['n'] for item in items]
            row['val_client_p10'] = float(np.quantile(per_client, .1))
            row['seconds']=time.time()-round_started
            history.append(row)
            (output/'history.json').write_text(json.dumps(history, indent=2))
            print(f"round={r:03d} train={row['train_accuracy']:.4f} val={row['val_accuracy']:.4f}", flush=True)
        torch.save(state, output/'final.pt')
        (output/'participation.json').write_text(json.dumps(dict(zip(manifest['clients'], participation.tolist())), indent=2))
        (output/'COMPLETE').write_text('Training complete; no test data were read.\n')
        provenance['finished_unix']=time.time()
        (output/'runtime.json').write_text(json.dumps(provenance,indent=2))
    finally:
        if ray:
            ray.shutdown()
    return history
