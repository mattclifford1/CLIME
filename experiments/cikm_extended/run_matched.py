'''
LIME's surrogate cut to predict class 1 as often as CLIME does, which splits CLIME's gain
into a change of operating point and a change of fit. Plain black boxes. Writes
results/raw/matched/.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import boxes, runner, tasks

SETTINGS = ('imbalanced', 'native', 'balanced')
SEEDS = range(5)


def jobs(seeds=SEEDS):
    return [((setting, dataset, family, seed),
             dict(dataset=dataset, setting=setting, seed=seed, family=family, arms=('plain',),
                  deltas=(), surrogate_deltas=(), matched=True))
            for seed in seeds for setting in SETTINGS for dataset in tasks.DATASETS[setting]
            for family in boxes.FAMILIES]


if __name__ == '__main__':
    runner.run_jobs('matched', jobs(range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS))
