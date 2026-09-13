'''
P1, P2, P5: the evaluation marginal, and what the weighting schemes do about it

Every surrogate is scored twice at each query point - once against the real test set and
once against points drawn around the query point - so the difference between the two
evaluation marginals is measured per query point rather than per configuration.

This is the sweep that tests whether the CIKM'23 effect is a property of the evaluation
distribution (P1), whether class weighting is a correction to it (P2), and whether an
explicit density ratio does the job better (P5).

usage:  OMP_NUM_THREADS=1 uv run python sweeps/sweep_marginal.py [processes]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from common import runner, evaluate   # noqa: E402  (path set above)

SWEEP = 'marginal'

DATASETS = ['Gaussian', 'Moons', 'Circles',
            'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes',
            'Iris', 'Wine', 'Sonar Rocks vs Mines', 'Ionosphere', 'Wheat Seeds',
            'Abalone Gender', 'Credit Scoring 1', 'Direct Marketing']

# one per log-odds geometry from the Logit-LIME taxonomy, so that a result here cannot be
# a property of one model family
MODELS = ['Logistic', 'LDA', 'MLP', 'SVM', 'Random Forest', 'Gradient Boosting']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def jobs():
    out = []
    for dataset in DATASETS:
        for model in MODELS:
            out.append((f'{dataset}|{model}',
                        {'dataset': dataset, 'model': model,
                         'data_params': DATA_PARAMS,
                         'eval_points': 'between_class_means', 'num_points': 20,
                         'with_diagnostics': True}))
    return out


if __name__ == '__main__':
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    runner.run_jobs(SWEEP, jobs(), processes=processes,
                    meta={'schemes': list(evaluate.SCHEMES),
                          'metrics': list(evaluate.METRICS),
                          'eval_data': list(evaluate.EVAL_DATA),
                          'datasets': DATASETS, 'models': MODELS,
                          'local_eval_samples': evaluate.LOCAL_EVAL_SAMPLES,
                          'data_params': DATA_PARAMS})
