'''
Explanation agreement on the extended grid.

Same separation as sweep_extended.py: this writes its own file and is exploratory, since
the datasets and black boxes were chosen after the registered test had been run.

usage:  python sweep_explanations_extended.py [results_explanations_extended.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
import shutil
import json
import sweep
import sweep_explanations
from sweep_extended import NEW_MODELS, NEW_GROUPS
from clime.data.loaders.exported_npz import available_exported

REGISTERED = 'results_explanations.json'


def main(out_path='results_explanations_extended.json'):
    if not os.path.exists(out_path) and os.path.exists(REGISTERED):
        shutil.copy(REGISTERED, out_path)
        n = len([k for k in json.load(open(out_path)) if not k.startswith('_')])
        print(f'seeded {out_path} with {n} already computed configurations', flush=True)

    sweep.DATASETS = sweep.DATASETS + sorted(available_exported())
    groups = dict(sweep.MODEL_GROUPS)
    for model in NEW_MODELS:
        groups[NEW_GROUPS[model]] = list(groups.get(NEW_GROUPS[model], [])) + [model]
    sweep.MODELS = [m for ms in groups.values() for m in ms]
    sweep.GROUP_OF = {m: g for g, ms in groups.items() for m in ms}
    # sweep_explanations imported these by value at import time
    sweep_explanations.DATASETS = sweep.DATASETS
    sweep_explanations.MODELS = sweep.MODELS
    sweep_explanations.GROUP_OF = sweep.GROUP_OF

    print(f'{len(sweep.DATASETS)} datasets x {len(sweep.MODELS)} black boxes', flush=True)
    sweep_explanations.run(out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results_explanations_extended.json')
