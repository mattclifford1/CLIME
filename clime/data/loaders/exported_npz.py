'''
Loader for datasets exported from ~/Repos/toy_datasets as .npz.

That package pins numpy>=2.3.5 and scikit-learn>=1.7.2, which cannot coexist with the
scikit-learn 1.1.3 this repo is pinned to, so its datasets are exported to disk by
experiments/logit_lime/export_toy_datasets.py rather than imported.

Each .npz holds 'X', 'y' and 'feature_names'. Registration is driven by whatever files
are present, so exporting more datasets makes them available without touching the
registry.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import glob
import numpy as np
import clime

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', '..', 'experiments', 'logit_lime', 'extra_datasets')


def get_exported(path):
    '''build a loader for one exported .npz file'''
    def loader(**kwargs):
        raw = np.load(path, allow_pickle=True)
        data = {'X': np.asarray(raw['X'], dtype=np.float64),
                'y': np.asarray(raw['y']).astype(np.int64),
                'feature_names': [str(n) for n in raw['feature_names']]}
        data = clime.data.shuffle_dataset(data)
        return clime.data.proportional_split(data, size=0.7)
    return loader


def available_exported():
    '''{registry name: loader} for every exported file found'''
    out = {}
    if not os.path.isdir(EXPORT_DIR):
        return out
    for path in sorted(glob.glob(os.path.join(EXPORT_DIR, '*.npz'))):
        name = os.path.splitext(os.path.basename(path))[0].replace('_', ' ')
        out[name] = get_exported(path)
    return out
