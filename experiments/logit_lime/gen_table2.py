'''
Generate table2.tex - the pre-registered test, aggregated per black box over all datasets.

Rows are grouped by the a priori log-odds geometry registered in PREREGISTRATION.md, and
the five model families that had never been run are marked, since those are what make the
test a prediction rather than a description.

usage:  python gen_table2.py [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import numpy as np
from analyse import load, GROUP_ORDER

NICE = {'Logistic': 'Logistic regression', 'LDA': 'LDA', 'QDA': 'QDA',
        'Gaussian Naive Bayes': 'Gaussian naive Bayes', 'MLP': 'MLP', 'SVM': 'SVM',
        'Decision Tree': 'Decision tree', 'Random Forest': 'Random forest',
        'k Nearest Neighbours': '$k$-nearest neighbours',
        'Random Forest (Platt calibrated)': 'Random forest + Platt',
        'Random Forest (isotonic calibrated)': 'Random forest + isotonic',
        'Gradient Boosting': 'Gradient boosting'}
# never run before the pre-registration, so their outcome could not have informed it
NEW = {'LDA', 'QDA', 'Gaussian Naive Bayes', 'Decision Tree', 'k Nearest Neighbours'}
GROUP_HEAD = {'A linear': r'\textbf{A} --- exactly linear log-odds',
              'B quadratic': r'\textbf{B} --- quadratic log-odds',
              'C smooth': r'\textbf{C} --- smooth, non-polynomial',
              'D piecewise constant': r'\textbf{D} --- piecewise constant',
              'E calibrated forest': r'\textbf{E} --- calibrated forest',
              'unassigned': r'unassigned'}


def fmt_adv(v):
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return f'${v/10**e:.1f}\\!\\cdot\\!10^{{{e}}}$'
    return f'${v:,.2f}$'


rows, _ = load(sys.argv[1] if len(sys.argv) > 1 else 'results_taxonomy.json')
n_datasets = len({r['dataset'] for r in rows})

lines = [
    r'\begin{table}[t]', r'\centering', r'\footnotesize',
    r'\setlength{\tabcolsep}{5pt}',
    r'\caption{The pre-registered test. Each row aggregates one black box over the '
    fr'{n_datasets} datasets: median log-odds linearity gap $\Delta$, median saturation, '
    r'median Logit-LIME advantage (ratio of local Brier scores, ${>}1$ favours '
    r'Logit-LIME), and the number of datasets on which Logit-LIME is better. Groups and '
    r'their predicted ordering were registered before any of these runs '
    r'(Section~\ref{sec:prereg}). Rows marked $\dagger$ are model families that had never '
    r'been run, so their results could not have informed the grouping.}',
    r'\label{tab:groups}',
    r'\begin{tabular}{lrrrr}', r'\toprule',
    r'Black box & $\Delta$ & sat. & advantage & better \\',
    r'\midrule']

for gi, g in enumerate(GROUP_ORDER):
    sub = [r for r in rows if r['group'] == g]
    if not sub:
        continue
    if gi:
        lines.append(r'\addlinespace')
    lines.append(fr'\multicolumn{{5}}{{l}}{{{GROUP_HEAD[g]}}} \\')
    for model in sorted({r['model'] for r in sub}):
        m = [r for r in sub if r['model'] == model]
        dag = r'$^\dagger$' if model in NEW else ''
        lines.append(
            fr"\quad {NICE.get(model, model)}{dag} & "
            fr"${np.nanmedian([r['gap'] for r in m]):+.2f}$ & "
            fr"${np.median([r['sat'] for r in m])*100:.0f}\%$ & "
            fr"{fmt_adv(np.median([r['adv'] for r in m]))} & "
            fr"{sum(r['adv'] > 1 for r in m)}/{len(m)} \\")
    if len({r['model'] for r in sub}) < 2:
        continue                      # a one-model group median just repeats the row
    lines.append(r'\cmidrule(l){2-5}')
    lines.append(fr"\quad \emph{{group median}} & "
                 fr"${np.nanmedian([r['gap'] for r in sub]):+.2f}$ & "
                 fr"${np.median([r['sat'] for r in sub])*100:.0f}\%$ & "
                 fr"{fmt_adv(np.median([r['adv'] for r in sub]))} & "
                 fr"{sum(r['adv'] > 1 for r in sub)}/{len(sub)} \\")

lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('table2.tex', 'w').write('\n'.join(lines) + '\n')
print(f'table2.tex written  ({len(rows)} rows, {n_datasets} datasets)')
