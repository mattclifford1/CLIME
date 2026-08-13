'''
Table: the 2x2 of evaluation choices (sweep_fidelity.py -> tables/fidelity-2x2.tex).

One row per log-odds group, one column per (metric, evaluation data) cell, so the reader
can see whether the conclusion depends on the two ways our evaluation departs from
clifford2023reconciling.

usage:  python analysis/table_fidelity.py [results_fidelity.json] [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import numpy as np
from analysis.analyse import GROUP_ORDER, GROUP_LABEL
from analysis.analyse_fidelity import load, gain, CELLS


def fmt_ratio(x):
    if x >= 1000:
        e = int(np.floor(np.log10(x)))
        return f'${x/10**e:.1f}\\!\\cdot\\!10^{{{e}}}$'
    return f'${x:.2f}$'


def main(fid_path=None, tax_path=None):
    rows, _ = load(fid_path, tax_path)
    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \caption{The two axes on which our evaluation departs from '
        r'\citet{clifford2023reconciling}: what is measured, and what it is measured over. '
        r'Their protocol is the rightmost column, ours the leftmost. Brier entries are the '
        r"standard surrogate's error divided by Logit-LIME's ($>1$ favours Logit-LIME); "
        r'fidelity entries are the difference in agreement, Logit-LIME minus standard '
        r'($>0$ favours Logit-LIME). Group medians.}',
        r'  \label{tab:fidelity}',
        r'  \small',
        r'  \begin{tabular}{lrrrrr}',
        r'    \toprule',
        r'    & & \multicolumn{2}{c}{Brier ratio} & \multicolumn{2}{c}{fidelity difference} \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}',
        r'    black box group & $n$ & local sample & test data & local sample & test data \\',
        r'    \midrule',
    ]
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        cells = []
        for c in CELLS:
            m = np.median([gain(r, c) for r in sub])
            cells.append(fmt_ratio(m) if c.startswith('Brier') else f'${m:+.4f}$')
        lines.append(f'    {GROUP_LABEL[g]} & {len(sub)} & ' + ' & '.join(cells) + r' \\')
    allc = []
    for c in CELLS:
        m = np.median([gain(r, c) for r in rows])
        allc.append(fmt_ratio(m) if c.startswith('Brier') else f'${m:+.4f}$')
    lines += [
        r'    \midrule',
        f'    all & {len(rows)} & ' + ' & '.join(allc) + r' \\',
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    open(paths.table('fidelity-2x2.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nwritten tables/fidelity-2x2.tex')


if __name__ == '__main__':
    main(*sys.argv[1:])
