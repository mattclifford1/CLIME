'''
The imbalance ratio swept over 4, 20 and 40 (10 is the imbalanced setting of run_main.py),
on the datasets that keep 30 held-out minority rows at every ratio. Writes results/raw/sweep/.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import boxes, runner, tasks

SETTINGS = ('ratio 4', 'ratio 20', 'ratio 40')
SEEDS = range(5)


def jobs(seeds=SEEDS):
    return [((setting, dataset, family, seed),
             dict(dataset=dataset, setting=setting, seed=seed, family=family))
            for seed in seeds for setting in SETTINGS
            for dataset in tasks.DATASETS[setting] for family in boxes.FAMILIES]


if __name__ == '__main__':
    runner.run_jobs('sweep', jobs(range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS))
