'''
the six weighting schemes, side by side, as a LaTeX table

One row per scheme, and for each: what it does to the collapse along the line, what it
costs on the surrogate's own marginal, and how it fares once the black box is degraded.

usage:  uv run python analysis/table_schemes.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np

import common_analysis as ca
from common import paths
from analyse_degrade import mechanism

NORMAL = 'bLIMEy (normal)'
SCHEMES = (NORMAL,
           'bLIMEy (cost sensitive sampled)',
           'bLIMEy (cost sensitive class)',
           'bLIMEy (local y)',
           'bLIMEy (local yhat)',
           'bLIMEy (density ratio)')
LABELS = {NORMAL: 'standard LIME',
          'bLIMEy (cost sensitive sampled)': r'class weights, $\hat{y}$ on the sample',
          'bLIMEy (cost sensitive class)': r'class weights, $y$ globally',
          'bLIMEy (local y)': r'class weights, $y$ locally',
          'bLIMEy (local yhat)': r'class weights, $\hat{y}$ locally',
          'bLIMEy (density ratio)': 'density ratio'}
FIDELITY = 'fidelity (local)'
KL = 'KL divergence (local)'


def row(marginal, degrade, scheme):
    worst = np.nanmedian([ca.worst_point(e, scheme, FIDELITY, 'test data')
                          for e in marginal.values()])
    variation = np.nanmedian([ca.variation(e, scheme, FIDELITY, 'test data')
                              for e in marginal.values()])
    cells = {'worst': worst, 'variation': variation}
    for name, eval_data in (('kl test', 'test data'), ('kl local', 'sample locally')):
        gains = [ca.paired_gain(e, scheme, NORMAL, KL, eval_data)
                 for e in marginal.values()]
        median, better, n = ca.median_and_count(gains)
        cells[name] = (median, better, n)
    degraded = [e for e in degrade.values() if mechanism(e) != 'clean']
    gains = [ca.paired_gain(e, scheme, NORMAL, KL, 'test data') for e in degraded]
    cells['degraded'] = ca.median_and_count(gains)
    return cells


def main():
    marginal, _, _ = ca.load('results_marginal.json')
    degrade, _, _ = ca.load('results_degrade.json')

    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \footnotesize',
             r'  \setlength{\tabcolsep}{5pt}',
             r'  \caption{The six schemes over the $84$ marginal configurations and the '
             r'$210$ degraded ones. \emph{Worst point} is the median lowest local fidelity '
             r'along the line and \emph{variation} the median max${}-{}$min along it; KL '
             r'columns are median $\log_{10}$ ratios against standard LIME, positive '
             r'favouring the scheme. Note that every scheme repairing the test-set score '
             r'makes the own-marginal score worse.}',
             r'  \label{tab:schemes}',
             r'  \begin{tabular}{lrrrrr}',
             r'    \toprule',
             r'    & \multicolumn{2}{c}{fidelity on test data} '
             r'& \multicolumn{2}{c}{$\log_{10}$ KL gain} & degraded \\',
             r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}',
             r'    weighting scheme & worst point & variation & test data '
             r'& own marginal & test data \\',
             r'    \midrule']
    print(f"{'scheme':38s} {'worst':>7s} {'variation':>10s} {'KL test':>20s} "
          f"{'KL local':>20s} {'degraded':>20s}")
    for scheme in SCHEMES:
        cells = row(marginal, degrade, scheme)
        print(f"{LABELS[scheme][:36]:38s} {cells['worst']:7.3f} "
              f"{cells['variation']:10.3f} "
              f"{ca.fmt(*cells['kl test']):>20s} {ca.fmt(*cells['kl local']):>20s} "
              f"{ca.fmt(*cells['degraded']):>20s}")
        if scheme == NORMAL:
            body = (f"    {LABELS[scheme]} & ${cells['worst']:.3f}$ & "
                    f"${cells['variation']:.3f}$ & --- & --- & --- \\\\")
        else:
            def cell(key):
                median, better, n = cells[key]
                return f'${median:+.3f}$ ({better}/{n})'
            body = (f"    {LABELS[scheme]} & ${cells['worst']:.3f}$ & "
                    f"${cells['variation']:.3f}$ & {cell('kl test')} & "
                    f"{cell('kl local')} & {cell('degraded')} \\\\")
        lines.append(body)
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']

    out = paths.table('schemes.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('\nwritten', out)


if __name__ == '__main__':
    main()
