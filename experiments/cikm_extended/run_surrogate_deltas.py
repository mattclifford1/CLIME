'''
Every named deltas method placing the threshold of LIME's surrogate, on the plain random
forest and logistic regression. Writes results/raw/surrogate_deltas/.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import boxes, runner, tasks

FAMILIES = ('Random Forest', 'Logistic')
SETTINGS = ('imbalanced', 'native')
SEEDS = range(2)


def jobs(seeds=SEEDS):
    return [((setting, dataset, family, seed),
             dict(dataset=dataset, setting=setting, seed=seed, family=family,
                  arms=('plain',), deltas=(), surrogate_deltas=boxes.all_deltas(),
                  per_class=5))
            for seed in seeds for setting in SETTINGS for dataset in tasks.DATASETS[setting]
            for family in FAMILIES]


if __name__ == '__main__':
    runner.run_jobs('surrogate_deltas', jobs(range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS))
