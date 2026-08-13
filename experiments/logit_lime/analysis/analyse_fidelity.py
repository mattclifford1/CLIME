'''
Analyse the 2x2 of evaluation choices (sweep_fidelity.py).

Two questions:

  1. Does the effect survive the evaluation protocol of clifford2023reconciling - fidelity
     measured over the real test set - or is it an artefact of measuring a proper scoring
     rule over locally sampled points?
  2. Section 2.2 of the paper argues in prose that fidelity is the wrong instrument here,
     because thresholding at 0.5 discards exactly the calibration Logit-LIME changes. Is
     that borne out?

The diagnostic gap is joined in from the main sweep, so we can also ask whether Delta
still predicts the benefit when the benefit is measured their way.

usage:  python analyse_fidelity.py [results_fidelity.json] [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import json
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from analysis.analyse import E, GROUP_ORDER, GROUP_LABEL, degenerate

CELLS = ['Brier | local sample', 'Brier | test data',
         'fidelity | local sample', 'fidelity | test data']


def load(fid_path, tax_path):
    fid = json.load(open(fid_path or paths.results('results_fidelity.json')))
    tax = json.load(open(tax_path or paths.results('results_taxonomy.json')))
    rows = []
    for key, v in fid.items():
        if key.startswith('_') or 'error' in v:
            continue
        dataset, model = key.split('|')
        t = tax.get(key, {})
        r = dict(dataset=dataset, model=model, group=v['group'],
                 gap=t.get('diagnostic', {}).get('gap', np.nan),
                 sat=t.get('diagnostic', {}).get('saturation', np.nan))
        for cell in CELLS:
            c = v['cells'][cell]
            r[cell] = {e: c[e]['mean'] for e in E}
        rows.append(r)
    errors = {k: v['error'] for k, v in fid.items()
              if not k.startswith('_') and 'error' in v}
    return rows, errors


def gain(row, cell):
    '''
    How much better Logit-LIME is, in the natural units of that cell.

    Brier is an error, so the benefit is the ratio std/logit (>1 means Logit-LIME wins).
    Fidelity is an agreement in [0,1], so the benefit is the difference logit - std
    (>0 means Logit-LIME wins). Keeping each in its own units is the point of the
    exercise - the question is what an author reading their own table would conclude.
    '''
    std, log = row[cell][E[0]], row[cell][E[1]]
    if cell.startswith('Brier'):
        return std/max(log, 1e-30)
    return log - std


def wins(rows, cell):
    thresh = 1.0 if cell.startswith('Brier') else 0.0
    return sum(gain(r, cell) > thresh for r in rows)


def main(fid_path=None, tax_path=None):
    rows, errors = load(fid_path, tax_path)
    n = len(rows)
    print(f'\n{n} configurations, {len(errors)} failed')

    print('\n' + '='*78)
    print('THE 2x2 OF EVALUATION CHOICES')
    print('='*78)
    print(f"{'cell':<26s} {'median benefit':>16s} {'Logit-LIME better':>19s} "
          f"{'spearman with gap':>19s}")
    ok = [r for r in rows if r not in degenerate(rows)]
    for cell in CELLS:
        g = [gain(r, cell) for r in rows]
        unit = 'x' if cell.startswith('Brier') else ' (abs)'
        rho, p = spearmanr([r['gap'] for r in ok], [gain(r, cell) for r in ok])
        print(f'{cell:<26s} {np.median(g):>15.3f}{unit:<6s} {wins(rows, cell):>7d}/{n:<4d} '
              f'{rho:>+13.3f} (p={p:.1g})')

    print('\n' + '='*78)
    print('WHAT FIDELITY SEES WHERE BRIER SEES ORDERS OF MAGNITUDE')
    print('='*78)
    print('The cases the paper is about: where the Brier benefit is largest, what would')
    print('an evaluation by fidelity have reported instead?')
    biggest = sorted(rows, key=lambda r: -gain(r, 'Brier | local sample'))[:10]
    print(f"\n{'dataset':<24s} {'model':<24s} {'Brier':>12s} "
          f"{'fid std':>9s} {'fid logit':>10s} {'diff':>8s}")
    for r in biggest:
        f = r['fidelity | test data']
        print(f"{r['dataset']:<24s} {r['model']:<24s} "
              f"{gain(r, 'Brier | local sample'):>11.3g}x "
              f"{f[E[0]]:>9.4f} {f[E[1]]:>10.4f} {f[E[1]]-f[E[0]]:>+8.4f}")

    for cell in ['fidelity | test data', 'fidelity | local sample']:
        d = [r[cell][E[1]] - r[cell][E[0]] for r in rows]
        stat, p = wilcoxon(d)
        print(f'\n{cell}: median difference {np.median(d):+.4f}, '
              f'mean {np.mean(d):+.4f}, range [{min(d):+.4f}, {max(d):+.4f}]')
        print(f'   Wilcoxon signed rank on the difference: p = {p:.2g}')
        print(f'   Logit-LIME better in {sum(x > 0 for x in d)}/{len(d)}, '
              f'worse in {sum(x < 0 for x in d)}, tied in {sum(x == 0 for x in d)}')

    print('\n' + '='*78)
    print('DOES THE CONCLUSION CHANGE PER GROUP?')
    print('='*78)
    print(f"{'group':<32s} {'n':>3s}  " + '  '.join(f'{c.split(chr(124))[0].strip()[:5]}'
                                                    f'/{c.split(chr(124))[1].strip()[:5]:<5s}'
                                                    for c in CELLS))
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        cells = '  '.join(f'{np.median([gain(r, c) for r in sub]):>11.3f}' for c in CELLS)
        print(f'{GROUP_LABEL[g]:<32s} {len(sub):>3d}  {cells}')

    print('\n' + '='*78)
    print('DISAGREEMENT: does fidelity ever rank the two surrogates the other way?')
    print('='*78)
    flip = [r for r in rows
            if (gain(r, 'Brier | local sample') > 1) != (gain(r, 'fidelity | test data') > 0)]
    print(f'{len(flip)}/{n} configurations where Brier and fidelity disagree on which '
          f'surrogate is better')
    for r in sorted(flip, key=lambda r: -gain(r, 'Brier | local sample'))[:12]:
        print(f"   {r['dataset']:<24s} {r['model']:<26s} "
              f"Brier {gain(r, 'Brier | local sample'):>10.3g}x   "
              f"fidelity {gain(r, 'fidelity | test data'):>+8.4f}")

    return rows


if __name__ == '__main__':
    main(*sys.argv[1:])
