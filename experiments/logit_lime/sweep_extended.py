'''
Extended sweep: the registered grid plus 15 further datasets and 4 further black boxes.

IMPORTANT - this is deliberately a SEPARATE file from results_taxonomy.json.

results_taxonomy.json is the pre-registered test: 14 datasets x 12 black boxes, fixed in
PREREGISTRATION.md before anything was run. Datasets and models chosen *after* seeing
those results cannot be folded back into it without destroying what the registration was
for. This file is the exploratory extension, and must be reported as such.

The extra datasets come from ~/Repos/toy_datasets (exported to .npz - see
export_toy_datasets.py) and widen the grid from 2-60 features to 2-279, and from roughly
balanced to a 4.9% minority class.

The extra black boxes impose their log-odds geometry by construction instead of inheriting
it from a model family, which is what the failure of registered statement 4 called for
(see clime/models/constructed_log_odds.py). One of them is a natural control the original
grid lacked:

    Nearest Class Mean has *exactly linear* log-odds - the difference of negative squared
    distances to the two centroids is linear in x - while saturating around 80% of the
    local neighbourhood. Linearity and saturation predict opposite outcomes for it, so it
    separates the two accounts within a single model.

usage:  python sweep_extended.py [results_extended.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
import json
import shutil
import sweep
from clime.data.loaders.exported_npz import available_exported

REGISTERED = 'results_taxonomy.json'

NEW_MODELS = ['Nearest Class Mean', 'Polynomial Logistic (deg 2)',
              'RBF Logistic (Nystroem)', 'Bagged Logistic']
# where each new black box would have been placed had it existed at registration time
NEW_GROUPS = {'Nearest Class Mean': 'A linear',
              'Polynomial Logistic (deg 2)': 'B quadratic',
              'RBF Logistic (Nystroem)': 'C smooth',
              'Bagged Logistic': 'unassigned'}


def main(out_path='results_extended.json'):
    # the registered configurations are identical computations, so carry them over rather
    # than spend another five hours reproducing them
    if not os.path.exists(out_path) and os.path.exists(REGISTERED):
        shutil.copy(REGISTERED, out_path)
        n = len([k for k in json.load(open(out_path)) if not k.startswith('_')])
        print(f'seeded {out_path} with {n} registered configurations', flush=True)

    sweep.DATASETS = sweep.DATASETS + sorted(available_exported())
    sweep.MODEL_GROUPS = dict(sweep.MODEL_GROUPS)
    for model in NEW_MODELS:
        group = NEW_GROUPS[model]
        sweep.MODEL_GROUPS[group] = list(sweep.MODEL_GROUPS.get(group, [])) + [model]
    sweep.MODELS = [m for group in sweep.MODEL_GROUPS.values() for m in group]
    sweep.GROUP_OF = {m: g for g, ms in sweep.MODEL_GROUPS.items() for m in ms}

    print(f'{len(sweep.DATASETS)} datasets x {len(sweep.MODELS)} black boxes '
          f'= {len(sweep.DATASETS)*len(sweep.MODELS)} configurations', flush=True)
    sweep.run(out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results_extended.json')
