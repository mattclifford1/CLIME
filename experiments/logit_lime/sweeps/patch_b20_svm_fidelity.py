'''
Recompute the SVM rows that FINDINGS.md B20 invalidated, and nothing else.

B20: thresholded fidelity compared each surrogate with black_box_model.predict(X), which for
an SVC with probability=True is the sign of its decision function rather than the argmax of
the probabilities the surrogate was fitted to. clime/evaluation/faithfulness.py now takes the
black box's class from predict_proba. That is bit-identical for 18 of the 20 registered black
boxes (checked); only 'SVM' and 'SVM balanced training' move, and only the second is outside
this study. So of the Logit-LIME results only the SVM rows of four files change:

    results_fidelity.json         the registered 2x2, 14 SVM configurations
    results_null.json             the null explainer, 14
    results_fidelity_full.json    the full-grid 2x2, 71
    results_null_full.json        the full-grid null explainer, 71

Each file is first copied to results/archive/pre-B20/, the SVM rows are recomputed into a
side file, and they replace the old rows only after a check that every cell whose metric did
not change (Brier, KL) reproduces the old value exactly - the evidence that the recomputation
is the same computation with one metric fixed. rerun_all.sh needs none of this: run from
scratch, the sweeps already use the fixed metric.

usage:  python sweeps/patch_b20_svm_fidelity.py [registered|full ...] [--jobs N]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import argparse
import json
import shutil
import numpy as np

ARCHIVE = paths.RESULTS/'archive'/'pre-B20'
MODEL = 'SVM'
UNCHANGED = ('Brier', 'KL')          # cell-name prefixes whose metric B20 did not touch


def archive(name):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    dst = ARCHIVE/name
    if not dst.exists():
        shutil.copy(paths.results(name), dst)
        print(f'archived {name} -> {dst}', flush=True)


def merge(name, side):
    '''replace the SVM rows of results/<name> with those in results/<side>, after checking'''
    old = json.load(open(paths.results(name)))
    new = json.load(open(paths.results(side)))
    keys = sorted(k for k in new if not k.startswith('_') and k.endswith(f'|{MODEL}'))
    worst, moved = 0.0, []
    for k in keys:
        if 'error' in new[k]:
            raise RuntimeError(f'{k} failed in the recomputation: {new[k]["error"]}')
        if k not in old or 'error' in old[k]:
            continue                     # nothing to compare against; added below
        o, n = old[k]['cells'], new[k]['cells']
        for cell in n:
            a, b = o[cell], n[cell]
            # results_fidelity has {explainer: {'scores': [...]}}, results_null has [...]
            pairs = ([(a[e]['scores'], b[e]['scores']) for e in b] if isinstance(b, dict)
                     else [(a, b)])
            for x, y in pairs:
                d = float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
                if cell.startswith(UNCHANGED):
                    worst = max(worst, d)
                elif d > 0:
                    moved.append((k, cell, d))
    if worst != 0.0:
        raise RuntimeError(f'{name}: an unchanged metric moved by {worst:.3g}, so the '
                           'recomputation is not the same computation - nothing merged')
    for k in keys:
        old[k] = new[k]
    json.dump(old, open(paths.results(name), 'w'))
    # the side file has served its purpose; its shards (if any) stay as the resume state
    os.remove(paths.results(side))
    cells = sorted({c for _, c, _ in moved})
    print(f'{name}: replaced {len(keys)} {MODEL} rows; Brier and KL reproduce exactly; '
          f'{len(moved)} fidelity score lists moved (cells: {cells}), '
          f'largest change {max((d for *_, d in moved), default=0):.4f}', flush=True)


def registered():
    from sweeps import sweep_fidelity, sweep_null
    for name, side, runner in (
            ('results_fidelity.json', 'results_fidelity_b20_svm.json',
             lambda out: sweep_fidelity.run(out, models=[MODEL])),
            ('results_null.json', 'results_null_b20_svm.json',
             lambda out: (setattr(sweep_null, 'MODELS', [MODEL]),
                          sweep_null.run(out, sweep_null.DATASETS)))):
        archive(name)
        runner(side)
        merge(name, side)


def full(jobs):
    from sweeps import parallel, full_grid
    from sweeps.run_full import FULL, ordered
    for name, side, setup, call in (
            ('results_fidelity_full.json', 'results_fidelity_full_b20_svm.json',
             FULL + 'from sweeps import sweep_fidelity as m',
             "m.DATASETS = {datasets!r}\nm.run({out!r}, models=['SVM'])"),
            ('results_null_full.json', 'results_null_full_b20_svm.json',
             FULL + "from sweeps import sweep_null as m\nm.MODELS = ['SVM']\n"
                    'm.GROUP_OF = full_grid.GROUP_OF',
             'm.run({out!r}, {datasets!r})')):
        archive(name)
        parallel.run(side, ordered(full_grid.DATASETS), setup, call, jobs=jobs)
        merge(name, side)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('grids', nargs='*', default=['registered', 'full'])
    p.add_argument('--jobs', type=int, default=30)
    a = p.parse_args()
    if 'registered' in a.grids:
        registered()
    if 'full' in a.grids:
        full(a.jobs)
