'''
How much did upgrading the stack move the numbers?

Compares the archived scikit-learn 1.1.3 / numpy 1.24 results against the current
scikit-learn 1.9 / numpy 2.4 ones. The point is not that they differ - they must, the
libraries changed - but whether any *conclusion* differs. Specifically:

  - do the group medians and win counts still say the same thing?
  - does the Delta correlation hold?
  - does any individual configuration flip across break-even, and if so, is it one that
    was near 1.0 anyway (noise) or one the argument rests on?

usage:  python compare_stacks.py [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import os
import numpy as np
from scipy.stats import spearmanr
from analysis.analyse import load, by_group, GROUP_ORDER, GROUP_LABEL

ARCHIVE = str(paths.ARCHIVE/'sklearn1.1.3')


def paired(new_path):
    old_path = os.path.join(ARCHIVE, os.path.basename(new_path))
    if not os.path.exists(old_path):
        raise SystemExit(f'no archived counterpart at {old_path}')
    new, _ = load(new_path)
    old, _ = load(old_path)
    old_by_key = {(r['dataset'], r['model']): r for r in old}
    pairs = [(old_by_key[(r['dataset'], r['model'])], r)
             for r in new if (r['dataset'], r['model']) in old_by_key]
    return pairs, len(old), len(new)


def main(path):
    pairs, n_old, n_new = paired(path)
    print(f'{os.path.basename(path)}: {n_old} old, {n_new} new, {len(pairs)} in common\n')

    o_adv = np.array([p[0]['adv'] for p in pairs])
    n_adv = np.array([p[1]['adv'] for p in pairs])
    o_gap = np.array([p[0]['gap'] for p in pairs])
    n_gap = np.array([p[1]['gap'] for p in pairs])

    ok = (o_adv > 0) & (n_adv > 0) & np.isfinite(o_adv) & np.isfinite(n_adv)
    log_ratio = np.abs(np.log10(n_adv[ok]/o_adv[ok]))
    print('ADVANTAGE  |log10(new/old)|')
    print(f'  median {np.median(log_ratio):.3f}   90th pct {np.percentile(log_ratio, 90):.3f}'
          f'   max {log_ratio.max():.3f}')
    print(f'  within 2x of the old value: {np.mean(log_ratio < np.log10(2)):.1%}')

    g_ok = np.isfinite(o_gap) & np.isfinite(n_gap)
    print(f'\nGAP  max |new-old| = {np.max(np.abs(n_gap[g_ok]-o_gap[g_ok])):.3f}'
          f'   median |new-old| = {np.median(np.abs(n_gap[g_ok]-o_gap[g_ok])):.4f}')

    # the thing that would actually change a conclusion
    flips = [(p, o, n) for p, o, n in zip(pairs, o_adv, n_adv)
             if np.isfinite(o) and np.isfinite(n) and (o > 1) != (n > 1)]
    print(f'\nCONFIGURATIONS THAT CROSSED BREAK-EVEN: {len(flips)} of {len(pairs)}')
    decisive = [(p, o, n) for p, o, n in flips
                if max(o, 1/max(o, 1e-30)) > 1.5 or max(n, 1/max(n, 1e-30)) > 1.5]
    for p, o, n in flips[:15]:
        marker = '  <-- not marginal' if (p, o, n) in decisive else ''
        print(f'   {p[1]["dataset"]:24s} {p[1]["model"]:24s} '
              f'[{p[1]["group"][0]}] {o:8.2f}x -> {n:8.2f}x{marker}')
    if decisive:
        print(f'   {len(decisive)} of these moved by more than 1.5x in one direction')

    print('\nGROUP MEDIANS  (old -> new)')
    go, gn = by_group([p[0] for p in pairs]), by_group([p[1] for p in pairs])
    print(f"{'group':<32s} {'benefit':>26s} {'better':>16s}")
    for name in GROUP_ORDER:
        if name not in gn:
            continue
        print(f"{GROUP_LABEL[name]:<32s} {go[name]['adv']:>11.2f}x -> {gn[name]['adv']:>10.2f}x"
              f"   {go[name]['wins']:>3d}/{go[name]['n']:<3d} -> {gn[name]['wins']:>3d}/{gn[name]['n']:<3d}")

    for label, gaps, advs in [('old', o_gap, o_adv), ('new', n_gap, n_adv)]:
        m = np.isfinite(gaps) & np.isfinite(advs)
        r, p = spearmanr(gaps[m], advs[m])
        print(f'\nSpearman(gap, benefit) {label}: rho = {r:+.3f}  p = {p:.2g}  n = {m.sum()}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json'))
