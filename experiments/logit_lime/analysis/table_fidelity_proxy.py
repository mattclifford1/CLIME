'''
Table: what each instrument does with the same surrogates
(analyse_fidelity_explanation.py -> tables/fidelity-proxy.tex).

One row per instrument, over the query points where a gradient ground truth exists.  The
columns are the three questions that can be asked of an instrument used as a proxy for
explanation correctness - does it track it, does it choose correctly when it chooses, and
how often does it decline to choose - followed by which surrogate it ends up crowning.  The
final rows give those same surrogates' actual explanation accuracy, which is what makes the
crown column readable: the instrument's favourite is not the surrogate with the best
explanations.

usage:  python analysis/table_fidelity_proxy.py [--extended]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import argparse
import numpy as np
from analysis.analyse_fidelity_explanation import (
    load, pooled_level, direction, crowns, explanation_quality, blind_points,
    INSTRUMENTS, LOWER_IS_BETTER)

ROW_LABEL = {'fidelity | local sample': 'fidelity, local sample',
             'fidelity | test data': 'fidelity, test data',
             'Brier': 'Brier', 'KL': 'KL'}
SURROGATES = ('standard', 'logit', 'logreg')
HEADS = {'standard': 'standard', 'logit': 'Logit-LIME', 'logreg': 'hard label'}


def main(extended=False):
    rows, mismatch = load(extended=extended)
    assert mismatch <= 1e-12, f'the join is invalid: Brier disagrees by {mismatch:.3g}'
    n_cfg = len({r['config'] for r in rows})

    body = []
    for ins in INSTRUMENTS:
        rho, _, _ = pooled_level(rows, ins)
        right, decided, ties, total = direction(rows, ins)
        counts, n = crowns(rows, ins)
        perfect, n_pts, _ = blind_points(rows, ins)
        blind = (f'${perfect/n_pts:.1%}$'.replace('%', r'\%')
                 if ins not in LOWER_IS_BETTER else '--')
        body.append(
            f'    {ROW_LABEL[ins]} & ${rho:+.2f}$ & '
            f'${right/max(decided,1):.0%}$'.replace('%', r'\%') + ' & '
            + f'${ties/max(total,1):.0%}$'.replace('%', r'\%') + f' & {blind} & '
            + ' & '.join(f'${counts[s]}$' for s in SURROGATES) + r' \\')

    q = explanation_quality(rows)
    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \footnotesize',
        r'  \caption{Each instrument applied to the same surrogates, at the '
        rf'{len(rows):,} query points of the {n_cfg} configurations that have an analytic '
        + (r'explanation ground truth ($70$ of them run blind against the third '
           r'pre-registration). ' if extended else r'explanation ground truth. ')
        + r'\emph{tracks} is the rank correlation of the reading '
        r'with cosine-to-truth, pooled over surrogates and points (negative is the '
        r'expected sign where a lower score is better), and it shows that fidelity is '
        r'\emph{not} uninformative. \emph{correct} and \emph{tied} split the comparisons '
        r'between standard LIME and Logit-LIME at points where their explanations differ: '
        r'how often the instrument names the one with the better explanation, and how '
        r'often it returns exactly the same number for both and so names neither. '
        r'\emph{blind} is the share of query points at which all three surrogates score a '
        r'perfect $1$. \emph{crowns} counts the configurations in which each surrogate has '
        r'the best mean reading. The last two rows are the surrogates themselves, and are '
        r'what the crown column has to be read against.}',
        r'  \label{tab:fidelityproxy}',
        r'  \begin{tabular}{lrrrrrrr}',
        r'    \toprule',
        rf'    & & & & & \multicolumn{{3}}{{c}}{{crowns (of {n_cfg})}} \\',
        r'    \cmidrule(lr){6-8}',
        r'    instrument & tracks & correct & tied & blind & '
        + ' & '.join(HEADS[s] for s in SURROGATES) + r' \\',
        r'    \midrule',
        *body,
        r'    \midrule',
        r'    \multicolumn{5}{l}{\emph{mean cosine of the explanation to the truth}} & '
        + ' & '.join(f'${q[s][0]:.3f}$' for s in SURROGATES) + r' \\',
        r'    \multicolumn{5}{l}{\emph{mean top-1 feature agreement}} & '
        + ' & '.join(f'${q[s][1]:.2f}$' for s in SURROGATES) + r' \\',
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    out = paths.table('fidelity-proxy.tex')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwritten {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--extended', action='store_true')
    a = p.parse_args()
    main(extended=a.extended)
