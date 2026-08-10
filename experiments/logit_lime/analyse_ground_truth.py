'''
Which surrogate recovers the black box's actual feature importances?

Only answerable for black boxes with exactly linear log-odds, where the model's own
coefficients are the local importances everywhere. See sweep_ground_truth.py.

usage:  python analyse_ground_truth.py [results_ground_truth.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import json
import numpy as np
from scipy.stats import wilcoxon

LABELS = {'standard': 'standard LIME', 'logit': 'Logit-LIME',
          'logreg': 'logistic-regression LIME'}


def load(path):
    d = json.load(open(path))
    return [dict(dataset=k.split('|')[0], model=k.split('|')[1], **v)
            for k, v in d.items() if 'error' not in v]


def summarise(rows, title):
    if not rows:
        return
    print(f'\n{title}   (n = {len(rows)})')
    print(f"{'surrogate':<28s} {'rank rho':>10s} {'top-1':>9s} {'cosine':>9s}")
    for key, label in LABELS.items():
        print(f"{label:<28s} "
              f"{np.nanmean([r[f'rho_{key}'] for r in rows]):>10.3f} "
              f"{np.nanmean([r[f'top1_{key}'] for r in rows]):>9.2f} "
              f"{np.nanmean([r[f'cos_{key}'] for r in rows]):>9.3f}")


def paired_test(rows, stat):
    a = np.array([r[f'{stat}_standard'] for r in rows], dtype=float)
    b = np.array([r[f'{stat}_logit'] for r in rows], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    wins = int(np.sum(b > a))
    ties = int(np.sum(b == a))
    try:
        p = wilcoxon(b, a, zero_method='zsplit').pvalue
    except ValueError:
        p = float('nan')
    print(f"  {stat:6s}  logit better on {wins}/{len(a)} "
          f"(ties {ties})   median difference {np.median(b-a):+.3f}   "
          f"Wilcoxon p = {p:.2g}")


if __name__ == '__main__':
    rows = load(sys.argv[1] if len(sys.argv) > 1 else 'results_ground_truth.json')
    summarise(rows, 'ALL LINEAR BLACK BOXES')
    for model in sorted({r['model'] for r in rows}):
        summarise([r for r in rows if r['model'] == model], model)

    print('\nPAIRED COMPARISON, Logit-LIME against standard LIME')
    for stat in ('rho', 'top1', 'cos'):
        paired_test(rows, stat)

    print('\nBY FEATURE COUNT (top-1 recovery of the true most important feature)')
    print(f"{'features':<12s} {'n':>4s} {'standard':>10s} {'logit':>10s}")
    for lo, hi in [(0, 10), (10, 25), (25, 60), (60, 400)]:
        sub = [r for r in rows if lo < r.get('n_features', -1) <= hi]
        if not sub:
            continue
        print(f"{f'{lo+1}-{hi}':<12s} {len(sub):>4d} "
              f"{np.nanmean([r['top1_standard'] for r in sub]):>10.2f} "
              f"{np.nanmean([r['top1_logit'] for r in sub]):>10.2f}")

    worst = sorted(rows, key=lambda r: r['top1_standard'])[:8]
    print('\nWHERE STANDARD LIME MISSES THE TRUE TOP FEATURE MOST OFTEN')
    for r in worst:
        print(f"  {r['dataset']:24s} {r['model']:20s} d={r.get('n_features','?'):>4} "
              f"standard {r['top1_standard']:.2f}  logit {r['top1_logit']:.2f}")
