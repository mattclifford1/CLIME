'''
Table: how far each surrogate's explanation moves when the black box's threshold does
(-> tables/threshold.tex, results/analysis_threshold.json).

usage:  python analysis/table_threshold.py [results_threshold.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from analysis.analyse_gradient_truth import GROUP_ORDER
from analysis.table_gradient_truth import GROUP_HEAD

MOVES = {'0.1 or 0.9': ('0.1', '0.9'), '0.01 or 0.99': ('0.01', '0.99')}
SURROGATES = ('standard', 'logit')


def load(path=None):
    raw = json.load(open(paths.results(path or 'results_threshold.json')))
    return [dict(dataset=k.split('|')[0], model=k.split('|')[1], **v) for k, v in raw.items()
            if 'error' not in v and v.get('points')]


def summarise(rows, thresholds):
    '''per surrogate: how far the size, the direction and the top feature of the explanation move'''
    out = {'configs': len(rows)}
    for label in SURROGATES:
        moves = [p['moves'][t][label] for r in rows for p in r['points'] for t in thresholds]
        ratio = np.array([m['norm_ratio'] for m in moves], dtype=float)
        ok = np.isfinite(ratio) & (ratio > 0)
        change = np.abs(np.log(ratio[ok]))
        out[label] = {'fold': float(np.exp(np.median(change))),
                      'doubled': float(np.mean(change >= np.log(2))),
                      'within_5_percent': float(np.mean(change <= np.log(1.05))),
                      'undefined': float(1 - ok.mean()),
                      'cos': float(np.nanmedian([m['cos'] for m in moves])),
                      'top_changed': float(1 - np.mean([m['same_top'] for m in moves])),
                      'points': int(ok.sum())}
    return out


def main(path=None):
    rows = load(path)
    groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in rows)]
    summary = {name: {**{g: summarise([r for r in rows if r['group'] == g], thresholds)
                         for g in groups}, 'all': summarise(rows, thresholds)}
               for name, thresholds in MOVES.items()}
    json.dump(summary, open(paths.results('analysis_threshold.json'), 'w'), indent=1)

    n_datasets = len({r['dataset'] for r in rows})
    lines = [
        r'\begin{table}[H]',
        r'  \centering',
        r'  \footnotesize',
        r'  \setlength{\tabcolsep}{4pt}',
        r"  \caption{Each explanation before and after the black box's threshold is moved from "
        r'$0.5$, over ' f'{n_datasets}' r' datasets, the black boxes with an analytic gradient '
        r'and $20$ query points, both directions pooled. Size: median factor by which the norm '
        r'of the coefficients changes, up or down. Top: share of query points whose largest '
        r'coefficient changes feature. Size leaves out the moves with an all-zero explanation '
        r'before or after (under $1\%$). The gradient of the log-odds is the same before and '
        r'after.}',
        r'  \label{tab:threshold}',
        r'  \begin{tabular}{lrrrrrrrrr}',
        r'    \toprule',
        r'    & & \multicolumn{4}{c}{threshold to $0.1$ or $0.9$} & '
        r'\multicolumn{4}{c}{threshold to $0.01$ or $0.99$} \\',
        r'    \cmidrule(lr){3-6}\cmidrule(lr){7-10}',
        r'    & & \multicolumn{2}{c}{size} & \multicolumn{2}{c}{top} & '
        r'\multicolumn{2}{c}{size} & \multicolumn{2}{c}{top} \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
        r'    black boxes & configs & std. & logit & std. & logit & std. & logit & std. & logit \\',
        r'    \midrule']
    for g in [*groups, 'all']:
        cells = []
        for name in MOVES:
            s = summary[name][g]
            cells += [f'${s[k]["fold"]:.2f}\\times$' for k in SURROGATES]
            cells += [f'${100*s[k]["top_changed"]:.0f}\\%$' for k in SURROGATES]
        if g == 'all':
            lines.append(r'    \midrule')
        head = {'unassigned': 'bagged logistic regression'}.get(g, GROUP_HEAD.get(g, 'all'))
        lines.append(f'    {head} & {summary[next(iter(MOVES))][g]["configs"]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    open(paths.table('threshold.tex'), 'w').write('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:2])
