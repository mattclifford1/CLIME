'''
Assess the second pre-registration: what does an explanation-optimal surrogate cost?

Four predictions were registered before running (PREREGISTRATION.md). This script tests
each one and prints the outcome, favourable or not.

A note on the numbers. KL is computed as sum_c y_c log(y_c/p_c), which is non-negative in
exact arithmetic but cancels to a small negative value when the surrogate reproduces the
black box to machine precision - which is exactly what the analytic Taylor surrogate does
on group A. Those values are zero to floating point and are floored at 1e-18 for
reporting, rather than being dropped, which would remove the best fits from the average.

usage:  python analysis/analyse_taylor.py [results_taylor.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from scipy.stats import wilcoxon

BRIER, KL = 'Brier score (local)', 'KL divergence (local)'
FLOOR = 1e-18
METHODS = {'standard': 'standard LIME', 'logit': 'Logit-LIME',
           'taylor': 'Taylor (analytic)', 'taylor_fd': 'Taylor (finite diff.)'}
GROUP_ORDER = ['A linear', 'B quadratic', 'C smooth', 'unassigned']


def load(path=None):
    path = paths.results(path or 'results_taylor.json')
    raw = json.load(open(path))
    return [dict(dataset=k.split('|')[0], model=k.split('|')[1], **v)
            for k, v in raw.items()
            if not k.startswith('_') and 'error' not in v and v.get('points')]


def floored(values):
    return np.maximum(np.asarray(values, dtype=float), FLOOR)


def summarise(rows, title):
    print(f'\n{title}   ({len(rows)} configurations)')
    print(f"{'surrogate':<24s} {'cosine':>8s} {'rank rho':>9s} {'top-1':>7s} "
          f"{'Brier':>11s} {'KL':>11s}")
    for key, nice in METHODS.items():
        print(f'{nice:<24s} '
              f'{np.nanmean([r[f"cos_{key}"] for r in rows]):>8.3f} '
              f'{np.nanmean([r[f"rho_{key}"] for r in rows]):>9.3f} '
              f'{np.nanmean([r[f"top1_{key}"] for r in rows]):>7.2f} '
              f'{np.nanmedian(floored([r[f"{BRIER}_{key}"] for r in rows])):>11.2e} '
              f'{np.nanmedian(floored([r[f"{KL}_{key}"] for r in rows])):>11.2e}')


def paired(rows, field, a, b, floor=False, higher_is_better=False):
    '''how often is b better than a, over configurations'''
    x = np.array([r[f'{field}_{a}'] for r in rows], dtype=float)
    y = np.array([r[f'{field}_{b}'] for r in rows], dtype=float)
    if floor:
        x, y = floored(x), floored(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    try:
        p = wilcoxon(y, x, zero_method='zsplit').pvalue
    except ValueError:
        p = float('nan')
    wins = int(np.sum(y > x)) if higher_is_better else int(np.sum(y < x))
    return wins, len(x), p


def ratio(rows, metric, a, b):
    '''median over configurations of a / b, both floored'''
    x = floored([r[f'{metric}_{a}'] for r in rows])
    y = floored([r[f'{metric}_{b}'] for r in rows])
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.median(x[ok]/y[ok]))


if __name__ == '__main__':
    rows = load(sys.argv[1] if len(sys.argv) > 1 else None)
    summarise(rows, 'ALL')
    groups = {g: [r for r in rows if r['group'] == g] for g in GROUP_ORDER}
    for g, sub in groups.items():
        if sub:
            summarise(sub, f'group {g}')

    print('\n' + '='*78)
    print('REGISTERED PREDICTIONS')
    print('='*78)

    print('\n1. Group A: no trade - Taylor should match Logit-LIME on fidelity.')
    a = groups['A linear']
    if a:
        for metric, nice in ((BRIER, 'Brier'), (KL, 'KL')):
            better, n, p = paired(a, metric, 'logit', 'taylor', floor=True)
            print(f'   {nice:<6s} Taylor better on {better}/{n} configurations, '
                  f'median ratio logit/taylor = {ratio(a, metric, "logit", "taylor"):.3g}')
        print(f'   cosine: Logit-LIME {np.mean([r["cos_logit"] for r in a]):.3f}, '
              f'Taylor 1.000 by construction')

    print('\n2. Groups B and C: Taylor should be WORSE on fidelity, better on explanation.')
    for g in ('B quadratic', 'C smooth'):
        sub = groups[g]
        if not sub:
            continue
        print(f'   {g}  (n = {len(sub)})')
        for metric, nice in ((BRIER, 'Brier'), (KL, 'KL')):
            better, n, _ = paired(sub, metric, 'logit', 'taylor', floor=True)
            print(f'     {nice:<6s} Taylor better on {better}/{n}, '
                  f'median Taylor/logit = '
                  f'{ratio(sub, metric, "taylor", "logit"):.3g}  '
                  f'({"WORSE, as registered" if better < n/2 else "BETTER, refutes"})')
        print(f'     cosine  standard {np.nanmean([r["cos_standard"] for r in sub]):.3f}, '
              f'logit {np.nanmean([r["cos_logit"] for r in sub]):.3f}, '
              f'Taylor 1.000 by construction')

    print('\n3. Finite differences should track the analytic gradient except where the '
          'black box saturates.')
    sat_cos, unsat_cos, degen = [], [], []
    for r in rows:
        for p in r['points']:
            (sat_cos if p['saturated'] else unsat_cos).append(p['taylor_fd']['cos'])
            degen.append(p['taylor_fd']['degenerate'])
    sat_cos, unsat_cos = np.array(sat_cos, dtype=float), np.array(unsat_cos, dtype=float)
    print(f'   unsaturated points: cosine {np.nanmean(unsat_cos):.4f} '
          f'(n = {np.sum(np.isfinite(unsat_cos))})')
    print(f'   saturated points:   cosine {np.nanmean(sat_cos):.4f} '
          f'(n = {np.sum(np.isfinite(sat_cos))})')
    print(f'   the estimate collapsed entirely at {int(np.sum(degen))}/{len(degen)} '
          f'points ({np.mean(degen):.1%})')
    steps = [p['taylor_fd']['step'] for r in rows for p in r['points']]
    for h in sorted({s for s in steps if np.isfinite(s)}):
        print(f'     step {h:g} sufficed at '
              f'{sum(1 for s in steps if s == h)/len(steps):.1%} of points')

    print('\n4. Cost.')
    q = [p['n_queries_fd'] for r in rows for p in r['points']]
    print(f'   finite-difference Taylor: {np.mean(q):.0f} black-box queries per '
          f'explanation on average (2d)')
    print(f'   LIME as configured here:  10,000')

    print('\nHOW THE PRACTICAL VERSION COMPARES (finite differences, no white-box access)')
    print(f"{'':<24s} {'cosine':>8s} {'top-1':>7s} {'median KL':>11s}")
    for key in ('standard', 'logit', 'taylor_fd'):
        print(f'{METHODS[key]:<24s} '
              f'{np.nanmean([r[f"cos_{key}"] for r in rows]):>8.3f} '
              f'{np.nanmean([r[f"top1_{key}"] for r in rows]):>7.2f} '
              f'{np.nanmedian(floored([r[f"{KL}_{key}"] for r in rows])):>11.2e}')
    for other in ('standard', 'logit'):
        wins, n, p = paired(rows, 'cos', other, 'taylor_fd', higher_is_better=True)
        print(f'  Taylor(fd) explanation better than {METHODS[other]} on '
              f'{wins}/{n}, Wilcoxon p = {p:.2g}')
