'''
Report the extended sweep, keeping the pre-registered result separate from it.

The registered claim is 14 datasets x 12 black boxes, fixed before running
(PREREGISTRATION.md). Everything added afterwards is exploratory and is reported as a
separate column, never merged into the registered numbers.

Three questions the extension is meant to answer:

  1. Does the coarse claim - exactly-linear log-odds versus everything else - hold on
     datasets an order of magnitude wider (up to 279 features) and far more imbalanced
     (down to a 4.9% minority class)?
  2. With log-odds geometry imposed by construction rather than inferred from model
     family, does the graded ordering that failed as registered statement 4 recover?
  3. Nearest Class Mean has exactly linear log-odds AND ~80% saturation. Linearity
     predicts a large benefit, saturation predicts none. Which happens?

usage:  python analyse_extended.py [results_extended.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import json
import numpy as np
from scipy.stats import spearmanr
from analysis.analyse import load, by_group, degenerate, GROUP_ORDER, GROUP_LABEL

REGISTERED_MODELS = ['Logistic', 'LDA', 'QDA', 'Gaussian Naive Bayes', 'MLP', 'SVM',
                     'Decision Tree', 'Random Forest', 'k Nearest Neighbours',
                     'Random Forest (Platt calibrated)',
                     'Random Forest (isotonic calibrated)', 'Gradient Boosting']
CONSTRUCTED = ['Nearest Class Mean', 'Polynomial Logistic (deg 2)',
               'RBF Logistic (Nystroem)', 'Bagged Logistic']


def registered_rows(rows, registered_datasets):
    return [r for r in rows
            if r['dataset'] in registered_datasets and r['model'] in REGISTERED_MODELS]


def table(rows, title):
    if not rows:
        return
    g = by_group(rows)
    print(f'\n{title}   (n = {len(rows)})')
    print(f"{'group':<32s} {'n':>4s} {'median gap':>11s} {'median benefit':>15s} {'better':>8s}")
    for name in GROUP_ORDER:
        if name not in g:
            continue
        v = g[name]
        print(f"{GROUP_LABEL[name]:<32s} {v['n']:>4d} {v['gap']:>+11.3f} "
              f"{v['adv']:>14.2f}x {v['wins']}/{v['n']:>3d}")
    ok = [r for r in rows if r not in degenerate(rows)]
    if len(ok) > 3:
        rg, pg = spearmanr([r['gap'] for r in ok], [r['adv'] for r in ok])
        rs, ps = spearmanr([r['sat'] for r in rows], [r['adv'] for r in rows])
        print(f"  Spearman(gap, benefit) = {rg:+.3f} (p={pg:.2g}, n={len(ok)})   "
              f"Spearman(sat, benefit) = {rs:+.3f} (p={ps:.2g})")


def per_model(rows, models, title):
    print(f'\n{title}')
    print(f"{'black box':<30s} {'n':>4s} {'med gap':>9s} {'med sat':>9s} "
          f"{'med benefit':>13s} {'better':>9s}")
    for m in models:
        sub = [r for r in rows if r['model'] == m]
        if not sub:
            continue
        print(f"{m:<30s} {len(sub):>4d} {np.nanmedian([r['gap'] for r in sub]):>+9.3f} "
              f"{np.median([r['sat'] for r in sub]):>8.1%} "
              f"{np.median([r['adv'] for r in sub]):>12.2f}x "
              f"{sum(r['adv'] > 1 for r in sub)}/{len(sub):>4d}")


def by_dimensionality(rows, dims):
    print('\nBY FEATURE COUNT (group A vs the rest)')
    print(f"{'features':<14s} {'n':>4s} {'A benefit':>12s} {'other benefit':>15s}")
    buckets = [(0, 5), (5, 15), (15, 40), (40, 300)]
    for lo, hi in buckets:
        sub = [r for r in rows if lo < dims.get(r['dataset'], -1) <= hi]
        if not sub:
            continue
        a = [r['adv'] for r in sub if r['group'] == 'A linear']
        o = [r['adv'] for r in sub if r['group'] != 'A linear']
        print(f"{f'{lo+1}-{hi}':<14s} {len(sub):>4d} "
              f"{(np.median(a) if a else float('nan')):>11.2f}x "
              f"{(np.median(o) if o else float('nan')):>14.2f}x")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else paths.results('results_extended.json')
    rows, errors = load(path)
    reg_meta = json.load(open(paths.results('results_taxonomy.json')))
    registered_datasets = sorted({k.split('|')[0] for k in reg_meta
                                  if not k.startswith('_')})

    if errors:
        print(f'{len(errors)} configurations failed:')
        for k, v in list(errors.items())[:12]:
            print(f'   {k:52s} {v[:60]}')

    reg = registered_rows(rows, registered_datasets)
    new = [r for r in rows if r not in reg]
    table(reg, 'PRE-REGISTERED GRID (unchanged, for reference)')
    table(new, 'EXPLORATORY EXTENSION (new datasets and/or new black boxes)')
    table(rows, 'EVERYTHING POOLED (exploratory)')

    per_model(rows, CONSTRUCTED, 'BLACK BOXES WITH GEOMETRY IMPOSED BY CONSTRUCTION')
    per_model(rows, ['Logistic', 'LDA', 'QDA', 'Gaussian Naive Bayes', 'MLP', 'SVM'],
              'their family-based counterparts')

    ncm = [r for r in rows if r['model'] == 'Nearest Class Mean']
    if ncm:
        print('\nTHE LINEARITY / SATURATION CONTROL')
        print('  Nearest Class Mean: log-odds exactly linear, probabilities heavily saturated.')
        print(f"  median saturation {np.median([r['sat'] for r in ncm]):.1%}, "
              f"median benefit {np.median([r['adv'] for r in ncm]):.2f}x, "
              f"better on {sum(r['adv'] > 1 for r in ncm)}/{len(ncm)}")
        print('  saturation account predicts ~1x; linearity account predicts a large gain.')

    dims = {}
    for r in rows:
        dims.setdefault(r['dataset'], r.get('n_features', -1))
    try:
        from clime.data.loaders.exported_npz import available_exported
        import clime
        for name in sorted({r['dataset'] for r in rows}):
            if dims.get(name, -1) < 0:
                tr, _ = clime.data.AVAILABLE_DATASETS[name]()
                dims[name] = int(np.shape(tr['X'])[1])
        by_dimensionality(rows, dims)
    except Exception as e:
        print(f'\n(dimensionality breakdown skipped: {type(e).__name__}: {e})')

    deg = degenerate(rows)
    if deg:
        print(f'\n{len(deg)} degenerate configurations (gap undefined, excluded above):')
        for r in deg[:12]:
            print(f"   {r['dataset']}|{r['model']:24s} sat={r['sat']:.0%} adv={r['adv']:.2f}x")
