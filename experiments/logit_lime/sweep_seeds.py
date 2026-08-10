'''
Repeat trials, for error bars.

Every result so far is a single train/test split with a single seed. This re-runs a
subset of the main sweep under several seeds so that the headline numbers can carry a
spread rather than a point estimate.

Each seed runs in its own subprocess. That is not fastidiousness: clime.RANDOM_SEED is
read at construction time by the models and the dataset loaders, but run_pipeline is
cached on the options dict alone, so changing the seed inside one process would silently
return the previous seed's cached result.

usage:  python sweep_seeds.py <output_prefix> [n_seeds]
        -> writes <output_prefix>_seed<N>.json for each seed
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
import json
import subprocess

SEEDS = [42, 1, 2, 3, 4]          # 42 is clime's default
DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication',
            'Pima Indian Diabetes', 'Ionosphere']
MODELS = ['Logistic', 'LDA', 'QDA', 'Gaussian Naive Bayes', 'MLP',
          'Random Forest', 'Decision Tree', 'k Nearest Neighbours']

CHILD = '''
import json, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
import sweep
sweep.DATASETS = {datasets!r}
sweep.MODELS = {models!r}
sweep.run({out!r}, seed={seed!r})
'''


def run(prefix, n_seeds=len(SEEDS)):
    here = os.path.dirname(os.path.abspath(__file__))
    for seed in SEEDS[:n_seeds]:
        out = f'{prefix}_seed{seed}.json'
        print(f'=== seed {seed} -> {out} ===', flush=True)
        code = CHILD.format(datasets=DATASETS, models=MODELS, out=out, seed=seed)
        subprocess.run([sys.executable, '-c', code], cwd=here, check=True)
    print('all seeds done')


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else len(SEEDS))
