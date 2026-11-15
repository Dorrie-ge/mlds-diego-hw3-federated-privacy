"""Reproduce adapter equivalence and split checks using explicit synthetic data."""
import argparse
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
from prepare_data import prepare, read_users


def validate(output):
    seed=2026
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        users={}
        for i in range(100):
            x=np.zeros((10,784),dtype=np.float32)
            x[:,i%784]=1
            users[f'user_{i:03d}']={'x':x.tolist(),'y':list(range(10))}
        train=root/'train.json';test=root/'test.json'
        train.write_text(json.dumps({'user_data':users}))
        test.write_text(json.dumps({'user_data':{'heldout':users['user_000']}}))
        with h5py.File(root/'train.h5','w') as h:
            examples=h.create_group('examples')
            for name,rows in users.items():
                group=examples.create_group(name)
                group['pixels']=np.asarray(rows['x']).reshape(-1,28,28)
                group['label']=rows['y']
        # Object arrays are created here, so the explicit pickle trust decision
        # applies to known synthetic content, not an arbitrary downloaded file.
        npy=np.empty(100,dtype=object)
        for i,rows in enumerate(users.values()):
            npy[i]={'images':np.asarray(rows['x']).reshape(-1,28,28),
                    'labels':np.asarray(rows['y'])}
        np.save(root/'train.npy',npy)
        refused=False
        try:
            read_users(root/'train.npy')
        except ValueError:
            refused=True
        assert refused, 'Object NPY unexpectedly loaded without explicit trust'
        for source,name,trusted in [(train,'json',False),(root/'train.h5','h5',False),
                                    (root/'train.npy','npy',True),(train,'repeat',False)]:
            prepare(source,test,root/name,seed=seed,trusted_course_npy=trusted)
        for f in (root/'json'/'clients').glob('*.npz'):
            with np.load(f,allow_pickle=False) as expected:
                assert len(expected['y_train'])==8 and len(expected['y_val'])==2
                for name in ['h5','npy','repeat']:
                    with np.load(root/name/'clients'/f.name,allow_pickle=False) as actual:
                        assert set(actual.files)==set(expected.files)
                        for key in expected.files:
                            assert np.array_equal(expected[key],actual[key]),(name,f.name,key)
        splits=json.loads((root/'json'/'split_indices.json').read_text())
        for client,indices in splits.items():
            tr,va=indices['train'],indices['val']
            assert not set(tr)&set(va)
            assert sorted(tr+va)==list(range(10))
            source=users[indices['original_user']]
            with np.load(root/'json'/'clients'/f'{client}.npz') as actual:
                for split,idx in [('train',tr),('val',va)]:
                    assert np.array_equal(actual['x_'+split],np.asarray(source['x'],dtype=np.float32)[idx])
                    assert np.array_equal(actual['y_'+split],np.asarray(source['y'])[idx])
        for name in ['h5','npy','repeat']:
            assert json.loads((root/name/'split_indices.json').read_text())==splits
            assert json.loads((root/name/'distribution.json').read_text())==json.loads((root/'json'/'distribution.json').read_text())
    result={'data_kind':'synthetic adapter controls, not course data','clients_checked':100,
            'json_hdf5_equivalence':True,'per_client_split':'8 train / 2 validation',
            'train_validation_indices_disjoint':True,'indices_exhaustive':True,
            'features_and_labels_follow_saved_indices':True,'json_npy_equivalence':True,
            'same_seed_repeat_equivalence':True,'untrusted_object_npy_rejected':True,
            'split_seed':seed,'course_data_read':False}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/validation/adapter_validation.json')
    validate(parser.parse_args().out)
