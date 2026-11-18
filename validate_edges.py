"""Executable synthetic boundary controls; no course data or checkpoints needed."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from federated import Client, Config, fedavg, model, perturb, state_copy
from prepare_data import prepare


def validate(output):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    rows = []

    def record(case, method, expected, observed):
        rows.append(dict(case=case, method=method, expected=expected,
                         observed=observed, passed=True))

    def rejects(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError('Expected ValueError')

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        client_file = root / 'client_000.npz'
        x = np.random.default_rng(2026).random((33, 784), dtype=np.float32)
        y = np.full(33, 61, dtype=np.int64)
        np.savez(client_file, x_train=x, y_train=y,
                 x_val=x[:2], y_val=y[:2])
        initial = state_copy(model())
        client = Client(client_file, asdict(Config(epochs=1, batch_size=32)))
        sizes = []
        original = F.cross_entropy

        def observe(logits, labels, *args, **kwargs):
            sizes.append(len(labels))
            return original(logits, labels, *args, **kwargs)

        with patch('torch.nn.functional.cross_entropy', side_effect=observe):
            update = client.update(initial, 1)
        assert sizes == [32, 1] and update['steps'] == 2 and update['n'] == 33
        assert all(torch.isfinite(v).all() for v in update['state'].values())
        assert any(not torch.equal(v, initial[k]) for k, v in update['state'].items())
        evaluation = client.evaluate(update['state'], 'val')
        assert evaluation['n'] == 2 and np.isfinite(evaluation['loss_sum'])
        assert model()(torch.from_numpy(x[:1])).shape == (1, 62)
        record('Missing local classes', 'Train and evaluate a writer containing only label 61',
               'Keep 62 logits; finite, nonzero update and finite validation loss',
               'PASS: 62 logits; finite changed parameters; 2 validation samples')
        record('Final short minibatch', 'Observe actual loss-call batch sizes for n=33, batch=32, E=1',
               'Two SGD steps; batches [32,1]; all 33 samples retained',
               f'PASS: batches {sizes}; steps={update["steps"]}; n={update["n"]}')

        test = root / 'test.json'
        test.write_text(json.dumps({'user_data': {'heldout': {
            'x': np.zeros((2, 784)).tolist(), 'y': [0, 1]}}}))

        def prepared(name, pixels, labels):
            source = root / (name + '.json')
            source.write_text(json.dumps({'user_data': {'writer': {
                'x': pixels.tolist(), 'y': labels}}}))
            dest = root / name
            prepare(source, test, dest, expected=1)
            return dest

        tiny = prepared('two_samples', np.zeros((2, 784)), [0, 61])
        with np.load(tiny / 'clients/client_000.npz') as data:
            assert len(data['y_train']) == len(data['y_val']) == 1
        split = json.loads((tiny / 'split_indices.json').read_text())['client_000']
        assert set(split['train']).isdisjoint(split['val'])
        assert sorted(split['train'] + split['val']) == [0, 1]
        record('Smallest accepted client', 'Prepare n=2 using the production loader',
               'One training and one validation sample; disjoint and exhaustive',
               'PASS: 1/1 split; indices disjoint and exhaustive')
        rejects(lambda: prepared('one_sample', np.zeros((1, 784)), [0]))
        record('Too-small client', 'Prepare n=1 with the production loader',
               'Reject with ValueError', 'PASS: ValueError')
        for i, bad in enumerate([-1, 62, .5]):
            rejects(lambda i=i, bad=bad: prepared('bad_label_' + str(i), np.zeros((2, 784)), [0, bad]))
        record('Invalid labels', 'Prepare labels -1, 62 and 0.5 separately',
               'Reject all three with ValueError', 'PASS: 3/3 rejected')
        for i, bad in enumerate([-.01, 1.01, float('nan'), float('inf')]):
            pixels = np.zeros((2, 784)); pixels[0, 0] = bad
            rejects(lambda i=i, pixels=pixels: prepared('bad_pixels_' + str(i), pixels, [0, 1]))
        record('Invalid clean pixels', 'Prepare pixels -0.01, 1.01, NaN and infinity',
               'Reject all four with ValueError', 'PASS: 4/4 rejected')

    for c in [0, .11, .001]:
        rejects(lambda c=c: Config(fraction=c).validate(100))
    for c in [.01, .1]:
        Config(fraction=c).validate(100)
    record('Participation limits', 'Validate C=0,0.11,0.001 and valid endpoints 0.01,0.1 for 100 clients',
           'Reject three invalid settings; accept both valid endpoints',
           'PASS: 3 rejected; 2 accepted')
    state = {'w': torch.ones(1)}
    for states, counts in [([], []), ([state], []), ([state], [0]), ([state], [-1])]:
        rejects(lambda states=states, counts=counts: fedavg(states, counts))
    record('Invalid aggregation', 'Empty states, unequal lengths, zero and negative counts',
           'Reject all four with ValueError', 'PASS: 4/4 rejected')
    for b in [-.1, float('nan'), float('inf')]:
        rejects(lambda b=b: perturb(torch.zeros(2, 784), b, 1))
    record('Invalid noise scale', 'b=-0.1, NaN and infinity',
           'Reject all three with ValueError', 'PASS: 3/3 rejected')
    clean = torch.tensor([[0., 1.]])
    assert torch.equal(perturb(clean, 0, 1), clean)
    noisy = perturb(torch.zeros(10000, 1), .3, 2026)
    assert (noisy < 0).any() and (noisy > 1).any()
    record('Noise endpoints', 'b=0 bitwise identity; b=0.3 on 10,000 zero pixels',
           'Exact identity at zero; positive scale remains unclipped outside [0,1]',
           'PASS: exact identity; values below 0 and above 1 retained')

    result = {'data_kind': 'synthetic boundary fixtures, NOT course accuracy',
              'executed_on': '2026-11-18',
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'numpy': np.__version__},
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ['federated.py', 'prepare_data.py', 'validate_edges.py']},
              'test_data_used': False, 'all_passed': all(row['passed'] for row in rows),
              'checks': rows}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/validation/edge_validation.json')
    validate(parser.parse_args().out)
