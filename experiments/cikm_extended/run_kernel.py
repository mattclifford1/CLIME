'''
The locality kernel width swept around LIME's default of 0.75 sqrt(d), for the plain
random forest and logistic regression. Writes results/raw/kernel/.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import runner, tasks

WIDTHS = (0.25, 0.5, 0.75, 1.5, 3.0)
FAMILIES = ('Random Forest', 'Logistic')
SETTINGS = ('imbalanced', 'native')
SEEDS = range(5)


def jobs(seeds=SEEDS):
    return [((setting, dataset, family, seed, f'width={width}'),
             dict(dataset=dataset, setting=setting, seed=seed, family=family,
                  arms=('plain',), deltas=(), surrogate_deltas=(), kernel_width=width))
            for seed in seeds for setting in SETTINGS for dataset in tasks.DATASETS[setting]
            for family in FAMILIES for width in WIDTHS]


if __name__ == '__main__':
    runner.run_jobs('kernel', jobs(range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS))
