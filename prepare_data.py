"""Prepare explicitly supplied course data. Supports LEAF JSON and TFF HDF5."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from federated import seed_for


def read_users(path, trusted_course_npy=False):
    path = Path(path)
    files = sorted(path.rglob('*.json')) if path.is_dir() else [path]
    users = {}
    for f in files:
        if f.suffix.lower() == '.npy':
            if not trusted_course_npy:
                raise ValueError('Object NPY requires --trusted-course-npy after verifying Canvas provenance')
            for i, rows in enumerate(np.load(f, allow_pickle=True)):
                users[f'user_{i:03d}'] = (rows['images'], rows['labels'])
        elif f.suffix.lower() in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(f) as h:
                for user, group in h['examples'].items():
                    if user in users:
                        raise ValueError('Duplicate client shard: '+user)
                    users[user] = (group['pixels'][:], group['label'][:])
        else:
            d = json.loads(f.read_text())
            for user, rows in d['user_data'].items():
                if user in users:
                    raise ValueError('Duplicate client shard: '+user)
                users[user] = (np.asarray(rows['x']), np.asarray(rows['y']))
    return users


def normalize(x, divisor):
    x = np.asarray(x, dtype=np.float32).reshape(len(x), 784) / divisor
    if not np.isfinite(x).all() or np.min(x)<0 or np.max(x)>1:
        raise ValueError('Pixels outside [0,1]; check explicit pixel-divisor')
    return x


def prepare(train_path, test_path, output, divisor=1., seed=2026, expected=100, trusted_course_npy=False):
    train, test = read_users(train_path,trusted_course_npy), read_users(test_path,trusted_course_npy)
    if len(train)!=expected:
        raise ValueError(f'Expected course subset of {expected} users; got {len(train)}. Do not silently subsample.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    (output/'clients').mkdir(parents=True)
    (output/'test').mkdir()
    distribution, totals, splits = {}, np.zeros(62,dtype=int), {}
    for i,user in enumerate(sorted(train)):
        x,y = train[user];x=normalize(x,divisor);y=np.asarray(y)
        if len(y)<2 or len(x)!=len(y) or not np.isin(y,np.arange(62)).all():
            raise ValueError('Empty/tiny client or invalid labels: '+user)
        y=y.astype(np.int64)
        idx=np.random.default_rng(seed_for(seed,user,'split')).permutation(len(y))
        cut=int(.8*len(y));tr,va=idx[:cut],idx[cut:]
        key=f'client_{i:03d}'
        np.savez_compressed(output/'clients'/f'{key}.npz',x_train=x[tr],y_train=y[tr],x_val=x[va],y_val=y[va])
        hist=np.bincount(y,minlength=62);totals+=hist;distribution[key]=hist.tolist()
        splits[key]={'original_user':user,'train':tr.tolist(),'val':va.tolist()}
    for i,(user,(x,y)) in enumerate(sorted(test.items())):
        x=normalize(x,divisor);y=np.asarray(y)
        if len(x)!=len(y) or not np.isin(y,np.arange(62)).all():
            raise ValueError('Invalid test labels')
        np.savez_compressed(output/'test'/f'heldout_{i:03d}.npz',x=x,y=y.astype(np.int64))
    files={str(f.relative_to(output)):hashlib.sha256(f.read_bytes()).hexdigest()
           for f in output.rglob('*.npz')}
    manifest={'dataset':'course Federated EMNIST (identity must be verified against Canvas)',
              'synthetic':False,'clients':list(distribution),'split_seed':seed,
              'pixel_divisor':divisor,'source_train':str(Path(train_path).resolve()),
              'source_test':str(Path(test_path).resolve()),'files_sha256':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'split_indices.json').write_text(json.dumps(splits))
    (output/'distribution.json').write_text(json.dumps({'overall':totals.tolist(),'clients':distribution},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True)
    p.add_argument('--output',default='data/prepared');p.add_argument('--pixel-divisor',type=float,choices=[1.,255.],required=True)
    p.add_argument('--trusted-course-npy',action='store_true')
    a=p.parse_args();prepare(a.train,a.test,a.output,a.pixel_divisor,trusted_course_npy=a.trusted_course_npy)
