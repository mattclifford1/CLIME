'''
Size, dimension and imbalance of every dataset in the full grid, as the pipeline sees it
(after the loader's own 70:30 split), written to results/dataset_meta.json for the
robustness slices in analyse_robustness.py.

The quantity that matters most is d / n_test. Both the surrogate's sampler and the
evaluation sampler draw from N(q, cov(test X)), and that covariance has rank at most
n_test - 1. Where d >= n_test the neighbourhood is a flat slice of feature space, which is
where the paper's high-dimensional failures sit.

usage:  python analysis/dataset_meta.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import warnings
import numpy as np
import clime
from sweeps import full_grid, sweep

warnings.filterwarnings('ignore')


def main():
    out = {}
    for name in full_grid.DATASETS:
        train, test = clime.data.AVAILABLE_DATASETS[name](**sweep.DATA_PARAMS)
        Xte = np.asarray(test['X'])
        y = np.r_[np.asarray(train['y']).ravel(), np.asarray(test['y']).ravel()]
        counts = np.bincount(y.astype(int))
        rank = int(np.linalg.matrix_rank(np.cov(Xte.T))) if Xte.shape[1] > 1 else 1
        out[name] = {'family': full_grid.FAMILY_OF[name], 'd': int(Xte.shape[1]),
                     'n_train': int(len(train['X'])), 'n_test': int(len(Xte)),
                     'minority': float(counts.min()/counts.sum()),
                     'cov_rank': rank, 'full_rank': bool(rank == Xte.shape[1])}
        print(f"{name:30s} {out[name]['family']:10s} d={out[name]['d']:4d} "
              f"n_test={out[name]['n_test']:5d} rank={rank:4d} "
              f"minority={out[name]['minority']:.1%}")
    json.dump(out, open(paths.results('dataset_meta.json'), 'w'), indent=1)
    print('written', paths.results('dataset_meta.json'))


if __name__ == '__main__':
    main()
