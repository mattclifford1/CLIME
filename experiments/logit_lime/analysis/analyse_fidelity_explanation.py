'''
Does a thresholded fidelity score tell you which explanation to trust?

analyse_gradient_truth.py asks this of Brier and KL - the instruments this study uses -
and finds that they predict explanation correctness in direction but not in magnitude.
The same question has to be asked of the instrument the study does NOT use, or the
decision to replace it is unargued.  This script asks it, by joining the two result files
that already exist:

  results_fidelity.json        per query point, the four (metric, evaluation data) cells
  results_gradient_truth.json  per query point, the cosine of each surrogate's explanation
                               to the analytic gradient of the black box's log-odds

Both sweep the same 20 query points of the same configurations with the same per-point
seeds, so they can be joined index-wise; the join asserts that the local Brier score agrees
between the files before trusting it.  84 configurations overlap (the 6 differentiable
black boxes of the registered grid x 14 datasets), extended to 154 by
results_fidelity_extended.json.

The answer is not the one the draft assumed, and the difference matters.  Fidelity is not
uninformative: pooled over query points it tracks cosine about as well as KL does.  What it
does is stop being able to distinguish surrogates at all - it TIES - and where it does
choose, it prefers the surrogate that is most confident rather than the one that is most
nearly right.  Reported here as four separate things, because they are four separate
claims:

  1. level, direction, magnitude - the three tests of analyse_gradient_truth.py, run for
     the two fidelity cells alongside Brier and KL
  2. ties, and the query points where every surrogate scores exactly 1
  3. which surrogate each instrument crowns, against how good that surrogate's
     explanations actually are
  4. compression - how many orders of magnitude of Brier difference fit inside one
     percentage point of fidelity

usage:  python analysis/analyse_fidelity_explanation.py [--extended] [--saturated]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import argparse
import json
import numpy as np
from scipy.stats import spearmanr

E = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']
SURROGATE = {'bLIMEy (normal)': 'standard', 'bLIMEy (logit)': 'logit',
             'bLIMEy (logistic regression)': 'logreg'}
LABEL = {'standard': 'standard LIME', 'logit': 'Logit-LIME',
         'logreg': 'logistic-regression LIME', 'null': 'null explainer'}
BRIER, KL = 'Brier score (local)', 'KL divergence (local)'

# the instruments, as (name, how to read one point's value, lower is better)
INSTRUMENTS = ['fidelity | local sample', 'fidelity | test data', 'Brier', 'KL']
LOWER_IS_BETTER = {'Brier', 'KL'}

GROUP_ORDER = ['A linear', 'B quadratic', 'C smooth', 'unassigned']
# a KL or Brier of exactly 0 is a perfect fit, not a missing value; clip before logs
FLOOR = 1e-16


def load(extended=False, null=True):
    '''
    one row per (configuration, query point), carrying every instrument and the cosine.

    Points are matched by position: both sweeps walk get_points_between_class_means in
    order and neither skips (checked below), so index i is the same q in both files.
    '''
    grad = json.load(open(paths.results('results_gradient_truth.json')))
    fid = json.load(open(paths.results('results_fidelity.json')))
    if extended:
        path = paths.results('results_fidelity_extended.json')
        if os.path.exists(path):
            fid = {**fid, **{k: v for k, v in json.load(open(path)).items()
                             if not k.startswith('_')}}
        else:
            print(f'note: {path} not present - running on the registered grid only')
    nulls = {}
    if null and os.path.exists(paths.results('results_null.json')):
        nulls = json.load(open(paths.results('results_null.json')))

    rows, mismatch = [], 0.0
    for key, g in grad.items():
        if key.startswith('_') or 'error' in g or not g.get('points'):
            continue
        f = fid.get(key)
        if f is None or 'error' in f:
            continue
        n = json.loads(json.dumps(nulls.get(key, {})))       # may be absent
        dataset, model = key.split('|')
        for i, point in enumerate(g['points']):
            row = {'config': key, 'dataset': dataset, 'model': model,
                   'group': g['group'], 'index': i,
                   'truth_norm': g['truth_norm'][i],
                   'saturated': g['saturation'] > 0}
            for e in E:
                s = SURROGATE[e]
                row[('cos', s)] = point[s]['cos']
                row[('top1', s)] = point[s]['top1']
                row[('Brier', s)] = point[s][BRIER]
                row[('KL', s)] = point[s][KL]
                for cell in ('fidelity | local sample', 'fidelity | test data'):
                    row[(cell, s)] = f['cells'][cell][e]['scores'][i]
                # the two files fitted the same surrogate independently; if they
                # disagree the join is invalid, so measure it rather than assume it
                mismatch = max(mismatch, abs(point[s][BRIER]
                                             - f['cells']['Brier | local sample'][e]['scores'][i]))
            if n and 'cells' in n:
                row[('cos', 'null')] = float('nan')
                row[('top1', 'null')] = float('nan')
                row[('Brier', 'null')] = n['cells']['Brier | local sample'][i]
                # KL was added to that sweep later than the other three cells
                row[('KL', 'null')] = (n['cells']['KL | local sample'][i]
                                       if 'KL | local sample' in n['cells']
                                       else float('nan'))
                for cell in ('fidelity | local sample', 'fidelity | test data'):
                    row[(cell, 'null')] = n['cells'][cell][i]
            rows.append(row)
    return rows, mismatch


def pooled_level(rows, instrument):
    '''rho(instrument, cosine) over every (point, surrogate) pair'''
    x, y = [], []
    for r in rows:
        for s in ('standard', 'logit', 'logreg'):
            v, c = r[(instrument, s)], r[('cos', s)]
            if np.isfinite(v) and np.isfinite(c):
                x.append(np.log10(max(v, FLOOR)) if instrument in LOWER_IS_BETTER else v)
                y.append(c)
    rho = spearmanr(x, y)
    return rho.statistic, rho.pvalue, len(x)


def direction(rows, instrument, a='standard', b='logit', tol=1e-3):
    '''
    of the points where the two surrogates' explanations really differ, how often does
    the instrument name the better one - and how often does it decline to choose?
    '''
    right = ties = total = 0
    for r in rows:
        va, vb = r[(instrument, a)], r[(instrument, b)]
        ca, cb = r[('cos', a)], r[('cos', b)]
        if not all(np.isfinite(v) for v in (va, vb, ca, cb)) or abs(cb - ca) <= tol:
            continue
        total += 1
        if va == vb:
            ties += 1
            continue
        better = (vb < va) if instrument in LOWER_IS_BETTER else (vb > va)
        right += int(better == (cb > ca))
    decided = total - ties
    return right, decided, ties, total


def magnitude(rows, instrument, a='standard', b='logit'):
    '''does a bigger reading gap go with a bigger explanation gap?'''
    x, y = [], []
    for r in rows:
        va, vb = r[(instrument, a)], r[(instrument, b)]
        ca, cb = r[('cos', a)], r[('cos', b)]
        if not all(np.isfinite(v) for v in (va, vb, ca, cb)):
            continue
        if instrument in LOWER_IS_BETTER:
            x.append(np.log10(max(va, FLOOR)) - np.log10(max(vb, FLOOR)))
        else:
            x.append(vb - va)
        y.append(cb - ca)
    rho = spearmanr(x, y)
    return rho.statistic, rho.pvalue, len(x)


def blind_points(rows, instrument, surrogates=('standard', 'logit', 'logreg')):
    '''
    query points where the instrument gives EVERY surrogate its best possible reading.

    For an agreement in [0,1] that is a score of exactly 1.  The spread of cosine across
    the same three surrogates at those points is what the instrument declined to see.
    '''
    perfect, spreads = 0, []
    for r in rows:
        v = [r[(instrument, s)] for s in surrogates]
        if not all(np.isfinite(x) for x in v) or not all(x == 1.0 for x in v):
            continue
        perfect += 1
        c = [r[('cos', s)] for s in surrogates]
        c = [x for x in c if np.isfinite(x)]
        if len(c) > 1:
            spreads.append(max(c) - min(c))
    return perfect, len(rows), np.array(spreads)


def crowns(rows, instrument, surrogates=('standard', 'logit', 'logreg')):
    '''which surrogate the instrument ranks first, counted over configurations'''
    by_config = {}
    for r in rows:
        by_config.setdefault(r['config'], []).append(r)
    counts = {s: 0 for s in surrogates}
    for cfg, rs in by_config.items():
        means = {}
        for s in surrogates:
            v = [r[(instrument, s)] for r in rs if np.isfinite(r[(instrument, s)])]
            means[s] = np.mean(v) if v else np.nan
        if not all(np.isfinite(v) for v in means.values()):
            continue
        best = (min if instrument in LOWER_IS_BETTER else max)(means, key=means.get)
        counts[best] += 1
    return counts, len(by_config)


def explanation_quality(rows, surrogates=('standard', 'logit', 'logreg')):
    out = {}
    for s in surrogates:
        c = np.array([r[('cos', s)] for r in rows], dtype=float)
        t = np.array([r[('top1', s)] for r in rows], dtype=float)
        out[s] = (np.nanmean(c), np.nanmean(t), int(np.sum(~np.isfinite(c))))
    return out


def compression(rows):
    '''
    how much Brier difference fits inside how little fidelity difference.

    Per query point, the ratio of the two surrogates' Brier scores against the difference
    in their fidelity, so the reader can see the exchange rate directly.
    '''
    ratio, diff = [], []
    for r in rows:
        a, b = r[('Brier', 'standard')], r[('Brier', 'logit')]
        fa, fb = r[('fidelity | test data', 'standard')], r[('fidelity | test data', 'logit')]
        if not all(np.isfinite(v) for v in (a, b, fa, fb)) or min(a, b) <= 0:
            continue
        ratio.append(a/b)
        diff.append(fb - fa)
    return np.array(ratio), np.array(diff)


def base_rates(rows, fid_tol=0.01, ratios=(2.0, 10.0), cos_margins=(0.1, 0.2, 0.4)):
    '''
    How often is the worked example's shape - fidelity cannot separate the two surrogates,
    a proper scoring rule can, and their explanations differ - actually the case?

    The figure shows one point.  Whether that point is a curiosity or the common case is
    this table's job, and it is reported at several strictnesses because any single choice
    of threshold would be arbitrary.
    '''
    tied, total = 0, 0
    counts = {(r, c): 0 for r in ratios for c in cos_margins}
    for row in rows:
        f = [(row[(cell, 'standard')], row[(cell, 'logit')])
             for cell in ('fidelity | local sample', 'fidelity | test data')]
        b, bl = row[('Brier', 'standard')], row[('Brier', 'logit')]
        d_cos = row[('cos', 'logit')] - row[('cos', 'standard')]
        if not all(np.isfinite(v) for pair in f for v in pair) or not np.isfinite(d_cos):
            continue
        if not np.isfinite(b) or not np.isfinite(bl) or bl <= 0:
            continue
        total += 1
        if not all(abs(a - c) <= fid_tol for a, c in f):
            continue
        tied += 1
        ratio = b/bl
        for r in ratios:
            for c in cos_margins:
                counts[(r, c)] += int(ratio >= r and abs(d_cos) > c)
    return tied, total, counts


def report(rows, title):
    print('\n' + '='*78)
    print(f'{title}   ({len({r["config"] for r in rows})} configurations, '
          f'{len(rows)} query points)')
    print('='*78)

    print('\n1. DOES THE INSTRUMENT TRACK EXPLANATION CORRECTNESS?')
    print(f"{'instrument':<24s} {'level rho':>11s} {'right when it chooses':>23s} "
          f"{'ties':>14s} {'magnitude rho':>15s}")
    for ins in INSTRUMENTS:
        rho, p, n = pooled_level(rows, ins)
        right, decided, ties, total = direction(rows, ins)
        mrho, mp, _ = magnitude(rows, ins)
        sign = '(lower better)' if ins in LOWER_IS_BETTER else ''
        print(f'{ins:<24s} {rho:>+11.3f} {right:>10d}/{decided:<6d} '
              f'{right/max(decided,1):>5.1%} {ties/max(total,1):>13.1%} '
              f'{mrho:>+15.3f}   {sign}')
    print('  level rho: pooled over surrogates and points, against cosine to the truth.')
    print('  ties: points where the two surrogates get an identical reading, as a share')
    print('        of the points where their explanations differ at all.')

    print('\n2. POINTS THE INSTRUMENT CANNOT SEE')
    for ins in ('fidelity | local sample', 'fidelity | test data'):
        perfect, n, spreads = blind_points(rows, ins)
        if perfect:
            print(f'  {ins:<24s} every surrogate scores exactly 1.000 at '
                  f'{perfect}/{n} points ({perfect/n:.1%});')
            print(f'  {"":<24s}   cosine spread there: median {np.median(spreads):.3f}, '
                  f'> 0.2 at {np.mean(spreads > 0.2):.1%} of them')
        else:
            print(f'  {ins:<24s} no points where every surrogate scores exactly 1.000')

    print('\n3. WHICH SURROGATE EACH INSTRUMENT CROWNS')
    q = explanation_quality(rows)
    print(f"{'instrument':<24s} " + ' '.join(f'{LABEL[s]:>26s}'
                                             for s in ('standard', 'logit', 'logreg')))
    for ins in INSTRUMENTS:
        counts, n = crowns(rows, ins)
        print(f'{ins:<24s} ' + ' '.join(
            f'{counts[s]:>18d}/{n:<7d}' for s in ('standard', 'logit', 'logreg')))
    print(f"{'mean cosine to truth':<24s} " + ' '.join(
        f'{q[s][0]:>26.3f}' for s in ('standard', 'logit', 'logreg')))
    print(f"{'mean top-1 agreement':<24s} " + ' '.join(
        f'{q[s][1]:>26.2f}' for s in ('standard', 'logit', 'logreg')))
    nan = {s: q[s][2] for s in q if q[s][2]}
    if nan:
        print(f'  (no scorable direction - a zero coefficient vector - at: '
              + ', '.join(f'{LABEL[s]} {v}' for s, v in nan.items()) + ' points)')

    print('\n4. COMPRESSION')
    ratio, diff = compression(rows)
    for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, np.inf)):
        m = (ratio >= lo) & (ratio < hi)
        if m.sum():
            hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
            print(f'  Brier ratio {lo:>5g}-{hi_s:<5s} ({m.sum():>5d} points): '
                  f'median fidelity difference {np.median(diff[m]):+.4f}')
    print('  Logit-LIME is better on Brier at '
          f'{np.mean(ratio > 1):.1%} of points and better on test-data fidelity at '
          f'{np.mean(diff > 0):.1%}, tied at {np.mean(diff == 0):.1%}')

    print('\n5. HOW OFTEN IS THE WORKED EXAMPLE\'S SHAPE THE CASE?')
    tied, total, counts = base_rates(rows)
    print(f'  both fidelity cells agree to within 0.01 at {tied}/{total} points '
          f'({tied/total:.1%}).  Of all {total} points, the share where that happens AND')
    for (r, c), n in sorted(counts.items()):
        print(f'    Brier differs by >= {r:>4.0f}x and cosine by > {c:.1f}: '
              f'{n:>5d}  ({n/total:.2%})')


def per_group(rows):
    print('\n' + '='*78)
    print('BY LOG-ODDS GROUP: how often does each instrument name the better explanation?')
    print('='*78)
    print(f"{'group':<16s} {'n points':>9s} " +
          ' '.join(f'{i:>24s}' for i in INSTRUMENTS))
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        cells = []
        for ins in INSTRUMENTS:
            right, decided, ties, total = direction(sub, ins)
            cells.append(f'{right/max(decided,1):>11.1%} ({ties/max(total,1):>4.0%} tied)')
        print(f'{g:<16s} {len(sub):>9d} ' + ' '.join(f'{c:>24s}' for c in cells))
    print('  read as: share of decided comparisons called correctly (share tied).')


def null_column(rows):
    have = [r for r in rows if ('fidelity | local sample', 'null') in r]
    if not have:
        print('\n(no results_null.json - run sweeps/sweep_null.py for the base rate)')
        return
    print('\n' + '='*78)
    print(f'THE NULL EXPLAINER, on the {len({r["config"] for r in have})} configurations '
          f'that also have a gradient truth')
    print('='*78)
    for cell in ('fidelity | local sample', 'fidelity | test data'):
        v = np.array([r[(cell, 'null')] for r in have])
        beats = [r for r in have
                 if r[(cell, 'null')] >= r[(cell, 'standard')]]
        print(f'  {cell:<24s} mean {v.mean():.4f}   '
              f'>= standard LIME at {len(beats)}/{len(have)} points '
              f'({len(beats)/len(have):.1%})')
    b_null = np.array([r[('Brier', 'null')] for r in have])
    worst = [r for r in have
             if r[('Brier', 'null')] > max(r[('Brier', s)]
                                           for s in ('standard', 'logit', 'logreg'))]
    print(f'  {"Brier | local sample":<24s} mean {b_null.mean():.4f}   '
          f'worst of the four at {len(worst)}/{len(have)} points '
          f'({len(worst)/len(have):.1%})')


def main(extended=False, saturated_only=None):
    rows, mismatch = load(extended=extended)
    print(f'join check: largest disagreement in local Brier between the two result '
          f'files = {mismatch:.3g}')
    if mismatch > 1e-12:
        print('  WARNING: the two sweeps did not fit the same surrogates. The join is '
              'invalid; re-run both before reading anything below.')
    if saturated_only is not None:
        rows = [r for r in rows if r['saturated'] == saturated_only]
        print(f'filtered to saturated={saturated_only}: {len(rows)} points')

    report(rows, 'ALL CONFIGURATIONS WITH BOTH A GRADIENT TRUTH AND A FIDELITY SWEEP')
    per_group(rows)
    null_column(rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--extended', action='store_true',
                   help='include results_fidelity_extended.json (the blind 70)')
    p.add_argument('--saturated', dest='saturated', action='store_true', default=None)
    p.add_argument('--unsaturated', dest='saturated', action='store_false')
    a = p.parse_args()
    main(extended=a.extended, saturated_only=a.saturated)
