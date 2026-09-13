'''
the 2-D version of the same question: where in the space does the mismatch live?

Every figure in the CIKM paper - and every sweep above - walks a single line between the
class means. `evaluation points: grid` has existed since Dec 2023 and has never been used
for a figure (FINDINGS.md E5). On two-dimensional data the PCA grid is the feature space
itself, so a heatmap of the score difference between two weighting schemes is directly
readable: it shows *where* aligning the surrogate's training distribution matters.

400 query points x 6 schemes per configuration, so diagnostics are off here - the line
sweeps measure those, and this sweep exists for the picture.

usage:  OMP_NUM_THREADS=1 uv run python sweeps/sweep_grid.py [processes]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from common import runner, evaluate   # noqa: E402

SWEEP = 'grid'

# two dimensional only: the grid is then the feature space rather than a PCA projection
DATASETS = ['Gaussian', 'Moons']
# a well fit black box, a piecewise constant one, and two degraded ones
MODELS = ['Logistic', 'Random Forest', 'Logistic (label noise 0.3)', 'MLP (underfit)']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def jobs():
    return [(f'{dataset}|{model}',
             {'dataset': dataset, 'model': model, 'data_params': DATA_PARAMS,
              'eval_points': 'grid', 'num_points': 20, 'with_diagnostics': False})
            for dataset in DATASETS for model in MODELS]


if __name__ == '__main__':
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    runner.run_jobs(SWEEP, jobs(), processes=processes,
                    meta={'schemes': list(evaluate.SCHEMES),
                          'metrics': list(evaluate.METRICS),
                          'eval_data': list(evaluate.EVAL_DATA),
                          'datasets': DATASETS, 'models': MODELS,
                          'grid': '20x20 = 400 query points',
                          'local_eval_samples': evaluate.LOCAL_EVAL_SAMPLES,
                          'data_params': DATA_PARAMS})
