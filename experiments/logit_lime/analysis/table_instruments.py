'''
Table: what each instrument can see (sweep_instruments.py -> tables/instruments.tex).

The figure shows the shapes; this is the arithmetic behind them.  Each entry is the
dynamic range of one instrument over one axis of the synthetic surrogate family - the
difference between its best and its worst reading as the explanation is taken from exactly
right to badly wrong.  A range of zero is an instrument that returns the same number for
the truth and for every error on that axis.

usage:  python analysis/table_instruments.py [results_instruments.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np

LABEL = {'fidelity | local sample': 'fidelity, local sample',
         'fidelity | test data': 'fidelity, test data',
         'fidelity at f(q)': 'fidelity at $f(q)$',
         'Spearman': 'Spearman of $p$',
         'Brier': 'Brier',
         'KL': 'KL'}
CONFIG_LABEL = {'Gaussian|Logistic': 'Gaussian, $d=2$',
                'Breast Cancer|Logistic': 'Breast Cancer, $d=30$'}


def rng(curve):
    v = np.asarray(curve, dtype=float)
    return float(np.nanmax(v) - np.nanmin(v))


def fmt(x):
    '''an exact zero is the result, so it is not dressed up as 0.000'''
    if x == 0.0:
        return r'$\mathbf{0}$'
    if x < 0.001:
        return f'${x:.0e}$'.replace('e-0', r'\!\cdot\!10^{-') + '}$'.replace('$$', '$')
    return f'${x:.3f}$'


def main(path=None):
    d = json.load(open(paths.results(path or 'results_instruments.json')))
    meta = d['_meta']
    configs = [k for k in d if not k.startswith('_')]
    instruments = [name for name, _, _ in
                   [tuple(i) for i in meta['instruments']]]

    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \footnotesize',
        r'  \caption{What each instrument can see. Entries are the dynamic range of the '
        r'instrument - its best reading minus its worst - as one property of the surrogate '
        r'is moved away from the truth and everything else is held exactly right '
        r'(Figure~\ref{fig:instruments}). \emph{direction} rotates the explanation through '
        r'a full turn, at the query point nearest the decision boundary and at the most '
        r'confident one; \emph{slope} multiplies the slope by $0.1$ to $10$, keeping '
        r'$g(q)=f(q)$, which slides the surrogate\'s class boundary; \emph{confidence} '
        r'multiplies the whole log-odds by the same factors, which leaves the boundary '
        r'where it is. The last two columns take the largest range over the three query '
        r'points, so each instrument is shown at its most responsive. A bold zero is exact '
        r'in floating point, not rounded: every member of the family gets an identical '
        r'reading. Spearman\'s small non-zero entries in the last two columns are a tie '
        r'artefact rather than a response - it reads exactly $1$ until the surrogate\'s '
        r'probabilities saturate to $0$ or $1$ in floating point, which only happens at the '
        r'top of the range.}',
        r'  \label{tab:instruments}',
        r'  \begin{tabular}{lrrrr}',
        r'    \toprule',
        r'    & \multicolumn{2}{c}{direction} & & \\',
        r'    \cmidrule(lr){2-3}',
        r'    instrument & boundary & confident & slope & confidence \\',
    ]
    summary = {}
    for cfg in configs:
        points = d[cfg]['points']
        boundary, confident = points[0], points[-1]
        lines += [r'    \midrule',
                  rf'    \multicolumn{{5}}{{l}}{{\emph{{{CONFIG_LABEL.get(cfg, cfg)}}}, '
                  rf'logistic black box}} \\']
        for name in instruments:
            vals = [rng(boundary['curves'][name]['theta']),
                    rng(confident['curves'][name]['theta']),
                    max(rng(p['curves'][name]['scale']) for p in points),
                    max(rng(p['curves'][name]['sharpen']) for p in points)]
            summary[(cfg, name)] = vals
            lines.append(f'    \\quad {LABEL.get(name, name)} & '
                         + ' & '.join(fmt(v) for v in vals) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']

    open(paths.table('instruments.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nwritten tables/instruments.tex')

    print(f"\n{'':44s} {'dir@bnd':>9s} {'dir@conf':>9s} {'slope':>9s} {'confidence':>11s}")
    for (cfg, name), v in summary.items():
        print(f'{cfg:>24s}  {name:<18s} ' + ' '.join(f'{x:>9.4g}' for x in v[:3])
              + f' {v[3]:>11.4g}')


if __name__ == '__main__':
    main(*sys.argv[1:])
