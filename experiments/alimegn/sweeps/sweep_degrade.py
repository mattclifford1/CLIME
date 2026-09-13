'''
P3, P4, P6: force P(yhat|x) away from P(y|x) and see which weighting wins

aLIMEgn's premise only bites when the black box and the data disagree. Three mechanisms
produce that disagreement (common/degrade.py): label noise on the training set, a model
with too little capacity, and class-imbalanced training data. The ladder of noise rates is
the quantitative axis - `local disagreement` in the diagnostics measures how far apart the
two distributions actually ended up, so the x-axis is measured rather than assumed.

The pair that matters is `bLIMEy (local y)` against `bLIMEy (local yhat)`: identical
points, identical mechanism, class frequencies from the labels in one and from the black
box's predictions in the other.

usage:  OMP_NUM_THREADS=1 uv run python sweeps/sweep_degrade.py [processes] [--seeds 42,1,2]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from common import runner, evaluate, degrade   # noqa: E402

SWEEP = 'degrade'

DATASETS = ['Gaussian', 'Moons', 'Breast Cancer', 'Banknote Authentication',
            'Pima Indian Diabetes', 'Ionosphere']

# the clean references
BASES = ['Logistic', 'MLP', 'Random Forest']
# the noise ladder, on all three bases
NOISY = [f'{base} (label noise {rate:g})'
         for base in BASES for rate in degrade.NOISE_RATES]
# capacity starved rather than mislabelled: a different route to the same gap
UNDERFIT = ['MLP (underfit)', 'Decision Tree (stump)', 'Logistic (over-regularised)',
            'k Nearest Neighbours (k=1)']
MODELS = BASES + NOISY + UNDERFIT

# the third mechanism: a displaced boundary from imbalanced training data
IMBALANCE = ['undersample class 0 to 50%', 'undersample class 0 to 20%']

# seeds 1 and 2 run on this subset only - enough to say whether a near-zero difference is
# inside seed noise, without repeating the whole grid
SEED_SUBSET_DATASETS = ['Gaussian', 'Breast Cancer']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def _job(dataset, model, rebalancing='none', seed=None):
    key = f'{dataset}|{model}'
    if rebalancing != 'none':
        key += f'|{rebalancing}'
    if seed is not None:
        key += f'|seed={seed}'
    kwargs = {'dataset': dataset, 'model': model, 'rebalancing': rebalancing,
              'data_params': DATA_PARAMS, 'eval_points': 'between_class_means',
              'num_points': 20, 'with_diagnostics': True}
    if seed is not None:
        kwargs['seed'] = seed
    return key, kwargs


def jobs(seeds=(None,)):
    out = []
    for dataset in DATASETS:
        for model in MODELS:
            out.append(_job(dataset, model))
        for rebalancing in IMBALANCE:      # imbalance applies to the clean bases
            for model in BASES:
                out.append(_job(dataset, model, rebalancing=rebalancing))
    for seed in [s for s in seeds if s is not None]:
        for dataset in SEED_SUBSET_DATASETS:
            for model in BASES + NOISY:
                out.append(_job(dataset, model, seed=seed))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    seeds = (None,)
    if '--seeds' in args:
        i = args.index('--seeds')
        seeds = tuple(int(s) for s in args[i+1].split(','))
        del args[i:i+2]
    processes = int(args[0]) if args else 12
    runner.run_jobs(SWEEP, jobs(seeds=seeds), processes=processes,
                    meta={'schemes': list(evaluate.SCHEMES),
                          'metrics': list(evaluate.METRICS),
                          'eval_data': list(evaluate.EVAL_DATA),
                          'datasets': DATASETS, 'bases': BASES, 'noisy': NOISY,
                          'underfit': UNDERFIT, 'imbalance': IMBALANCE,
                          'noise_rates': list(degrade.NOISE_RATES),
                          'seed_subset': SEED_SUBSET_DATASETS,
                          'local_eval_samples': evaluate.LOCAL_EVAL_SAMPLES,
                          'data_params': DATA_PARAMS})
