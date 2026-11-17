'''
The CIKM'23 protocol on every dataset at its own class ratio, balanced (110:110) and
imbalanced (200:20), for six model families, three training arms and every threshold
placement. Writes results/raw/main/.

usage: N_JOBS=16 uv run python run_main.py [seeds]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import boxes, runner, tasks

SETTINGS = ('imbalanced', 'native', 'balanced')
SEEDS = range(10)


def jobs(seeds=SEEDS):
    out = []
    for seed in seeds:
        for setting in SETTINGS:
            for dataset in tasks.DATASETS[setting]:
                for family in boxes.FAMILIES:
                    kwargs = dict(dataset=dataset, setting=setting, seed=seed, family=family)
                    if setting == 'balanced':
                        # nothing to correct: the plain model only
                        kwargs.update(arms=('plain',), deltas=())
                    out.append(((setting, dataset, family, seed), kwargs))
    return out


if __name__ == '__main__':
    seeds = range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS
    runner.run_jobs('main', jobs(seeds))
