'''
P7, P8: local versus global class imbalance (FINDINGS.md E4)

Two manipulations crossed over the same six weighting schemes:

  training data     natural, or class 0 undersampled to 20%
  black box         trained normally, or with balanced class weights

The test set keeps its natural class balance in every cell, so the imbalance is a property
of what the black box learned rather than of what it is scored against.

E4 has been open since the CIKM work on the strength of one unreported figure: that class
weights taken from the black box's predictions on its own local sample help, while class
weights taken from the global training-set imbalance do not. This settles it, and P8 tests
whether the mechanism found in the degradation sweep (P6) shows up again under a
manipulation that has nothing to do with corrupting labels.

usage:  OMP_NUM_THREADS=1 uv run python sweeps/sweep_balance.py [processes]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from common import runner, evaluate   # noqa: E402

SWEEP = 'balance'

DATASETS = ['Gaussian', 'Moons', 'Circles',
            'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes',
            'Iris', 'Wine', 'Sonar Rocks vs Mines', 'Ionosphere', 'Wheat Seeds',
            'Abalone Gender', 'Credit Scoring 1', 'Direct Marketing']

# each model appears twice: trained normally, and trained with balanced class weights.
# B2 (FINDINGS.md) means the balanced variants only actually balance since 2026-08-07
PAIRS = [('Logistic', 'Logistic balanced training'),
         ('Random Forest', 'Random Forest balanced training'),
         ('SVM', 'SVM balanced training')]
MODELS = [m for pair in PAIRS for m in pair]

# how imbalanced the black box's training data are
REBALANCING = ['none', 'undersample class 0 to 20%']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def jobs():
    out = []
    for dataset in DATASETS:
        for model in MODELS:
            for rebalancing in REBALANCING:
                key = f'{dataset}|{model}'
                if rebalancing != 'none':
                    key += f'|{rebalancing}'
                out.append((key,
                            {'dataset': dataset, 'model': model,
                             'rebalancing': rebalancing, 'data_params': DATA_PARAMS,
                             'eval_points': 'between_class_means', 'num_points': 20,
                             'with_diagnostics': True}))
    return out


if __name__ == '__main__':
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    runner.run_jobs(SWEEP, jobs(), processes=processes,
                    meta={'schemes': list(evaluate.SCHEMES),
                          'metrics': list(evaluate.METRICS),
                          'eval_data': list(evaluate.EVAL_DATA),
                          'datasets': DATASETS, 'pairs': PAIRS,
                          'rebalancing': REBALANCING,
                          'local_eval_samples': evaluate.LOCAL_EVAL_SAMPLES,
                          'data_params': DATA_PARAMS})
