'''
Six surrogates (LIME, CLIME and their logit and soft-label logistic forms) scored as
explanations: against the black box's log-odds gradient, across a move of its threshold,
and across two draws of the neighbourhood. Writes results/raw/explanations/.

usage: N_JOBS=12 uv run python run_explanations.py [seeds]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
from common import boxes, explanations, runner, tasks

SETTINGS = ('imbalanced', 'native', 'balanced')
SEEDS = range(5)


def jobs(seeds=SEEDS):
    out = []
    for seed in seeds:
        for setting in SETTINGS:
            for dataset in tasks.DATASETS[setting]:
                for family in boxes.FAMILIES:
                    kwargs = dict(dataset=dataset, setting=setting, seed=seed, family=family)
                    if setting == 'balanced':
                        kwargs.update(arms=('plain',), deltas=())
                    out.append(((setting, dataset, family, seed), kwargs))
    return out


if __name__ == '__main__':
    seeds = range(int(sys.argv[1])) if len(sys.argv) > 1 else SEEDS
    runner.run_jobs('explanations', jobs(seeds), job=explanations.run)
