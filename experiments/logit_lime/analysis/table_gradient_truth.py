'''
Generate tables/gradient-truth.tex - explanation accuracy against the analytic gradient,
by registered group.

The coefficient ground truth covers three black boxes; this covers eleven, so the table
is arranged by group to show where the extension actually reaches: A was already
answerable, B and C were not.

usage:  python analysis/table_gradient_truth.py [results_gradient_truth.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import numpy as np
from scipy.stats import wilcoxon
from analysis.analyse_gradient_truth import load, GROUP_ORDER

GROUP_HEAD = {'A linear': r'\textbf{A} --- exactly linear',
              'B quadratic': r'\textbf{B} --- quadratic',
              'C smooth': r'\textbf{C} --- smooth, non-polynomial',
              'unassigned': r'unassigned'}
NICE = {'Logistic': 'Logistic regression', 'LDA': 'LDA',
        'Nearest Class Mean': 'Nearest class mean', 'QDA': 'QDA',
        'Gaussian Naive Bayes': 'Gaussian naive Bayes', 'Bayes Optimal': 'Bayes optimal',
        'Polynomial Logistic (deg 2)': 'Polynomial logistic', 'MLP': 'MLP', 'SVM': 'SVM',
        'RBF Logistic (Nystroem)': 'RBF logistic', 'Bagged Logistic': 'Bagged logistic'}


def stats(rows):
    out = []
    for field in ('cos', 'rho', 'top1'):
        a = np.array([r[f'{field}_standard'] for r in rows], dtype=float)
        b = np.array([r[f'{field}_logit'] for r in rows], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        out.append((np.mean(a[ok]), np.mean(b[ok])))
    a = np.array([r['cos_standard'] for r in rows], dtype=float)
    b = np.array([r['cos_logit'] for r in rows], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    wins = int(np.sum(b[ok] > a[ok]))
    return out, wins, int(np.sum(ok))


def line(label, rows, indent=False):
    (cos, rho, top1), wins, n = stats(rows)
    name = f'\\quad {label}' if indent else label
    bold = (lambda v: f'$\\mathbf{{{v:.3f}}}$') if not indent else (lambda v: f'${v:.3f}$')
    return (f'{name} & {n} & ${cos[0]:.3f}$ & {bold(cos[1])} & '
            f'${rho[0]:.3f}$ & ${rho[1]:.3f}$ & ${top1[0]:.2f}$ & ${top1[1]:.2f}$ & '
            f'{wins}/{n} \\\\')


def main(path=None):
    rows = load(path)
    n_datasets = len({r['dataset'] for r in rows})

    lines = [
        r'\begin{table}[tbp]', r'  \centering', r'  \footnotesize',
        r'  \setlength{\tabcolsep}{4pt}',
        r'  \caption{Explanation accuracy against the analytic gradient of the black '
        r"box's log-odds, which is the true local importance vector wherever it exists. "
        fr'Each black box is aggregated over {n_datasets} datasets and $20$ query points: '
        r'mean cosine similarity, mean Spearman correlation of $|\beta|$, and the rate at '
        r'which the surrogate names the truly most important feature. The last column '
        r'counts datasets on which Logit-LIME has the higher cosine. Groups D and E are '
        r'absent because a piecewise constant black box has no local gradient, so no '
        r'ground truth exists for them at all.}',
        r'  \label{tab:gradient}',
        r'  \begin{tabular}{lrrrrrrrr}', r'    \toprule',
        r'    & & \multicolumn{2}{c}{cosine} & \multicolumn{2}{c}{rank $\rho$} & '
        r'\multicolumn{2}{c}{top-1} & \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
        r'    black box & $n$ & std. & logit & std. & logit & std. & logit & better \\',
        r'    \midrule',
    ]
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        lines.append(f'    {GROUP_HEAD[g]} & \\multicolumn{{8}}{{l}}{{}} \\\\')
        for model in sorted({r['model'] for r in sub}):
            lines.append('    ' + line(NICE.get(model, model),
                                       [r for r in sub if r['model'] == model],
                                       indent=True))
        lines.append('    \\midrule')
    lines.append('    ' + line(r'\textbf{all}', rows))
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']

    open(paths.table('gradient-truth.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # the paired test quoted in the prose
    a = np.array([r['cos_standard'] for r in rows], dtype=float)
    b = np.array([r['cos_logit'] for r in rows], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    print(f'\nlogit better on cosine in {int(np.sum(b[ok] > a[ok]))}/{int(np.sum(ok))}, '
          f'Wilcoxon p = {wilcoxon(b[ok], a[ok], zero_method="zsplit").pvalue:.2g}')
    print('\nwritten tables/gradient-truth.tex')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
