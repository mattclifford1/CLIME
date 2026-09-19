'''
Δ against R²_logit as predictors of Logit-LIME's advantage, and tables/diagnostic.tex.

Three sets of configurations, never pooled:

    registered   results_taxonomy.json, 14 x 12, degenerate rule 'paper' (the file predates
                 the guarded R²). What the paper reported
    extended     results_extended.json, 29 x 16, rule 'paper'. Seen before the fifth
                 registration chose R²_logit, so this is where the choice was made
    new          results_full.json restricted to the 42 datasets added in the fifth
                 registration, rule 'guarded'. Blind: the out-of-sample test

Intervals are the dataset-level cluster bootstrap in diagnostic_stats.

usage:  python analysis/analyse_diagnostic.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from analysis import diagnostic_stats as ds
from sweeps import full_grid


def sets():
    out = {}
    out['registered'] = ds.usable(ds.load(paths.results('results_taxonomy.json')), 'paper')
    out['extended'] = ds.usable(ds.load(paths.results('results_extended.json')), 'paper')
    full = paths.results('results_full.json')
    if os.path.exists(full):
        rows = ds.load(full)
        new = [r for r in rows if r['dataset'] in full_grid.NEW]
        seen = [r for r in rows if r['dataset'] not in full_grid.NEW]
        out['new'] = ds.usable(new, 'guarded')
        out['new (paper rule)'] = ds.usable(new, 'paper')
        out['seen, recomputed (guarded)'] = ds.usable(seen, 'guarded')
    return out


def fmt_ci(v, ci):
    return f'{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]'


def report(name, rows, excluded, B=2000):
    s = ds.summary(rows)
    print(f'\n== {name}: n = {s["n"]} configurations on {s["n_datasets"]} datasets '
          f'({excluded} excluded)')
    ci = {}
    for k in ds.PREDICTORS + ['sat']:
        ci[f'rho_{k}'] = ds.cluster_bootstrap(rows, ds.rho_stat(k), B)
        print(f'   rho({ds.LABEL[k]:22s}, adv) = {fmt_ci(s[f"rho_{k}"], ci[f"rho_{k}"])}   '
              f'AUC(adv>2) = {s[f"auc_{k}"]:.3f}   AUC(adv>10) = {s[f"auc10_{k}"]:.3f}')
    ci['diff'] = ds.cluster_bootstrap(rows, ds.diff_stat('r2_logit', 'gap'), B)
    d = s['rho_r2_logit'] - s['rho_gap']
    print(f'   rho(R²_logit) − rho(Δ) = {fmt_ci(d, ci["diff"])}')
    ci['auc_r2_logit'] = ds.cluster_bootstrap(rows, ds.auc_stat('r2_logit'), B)
    ci['auc_gap'] = ds.cluster_bootstrap(rows, ds.auc_stat('gap'), B)
    print(f'   partial rho(Δ, adv | R²_logit)   = {s["partial_gap|r2_logit"]:+.3f}')
    print(f'   partial rho(R²_p, adv | R²_logit) = {s["partial_r2_prob|r2_logit"]:+.3f}')
    for rule in ('rule_r2_logit>0.95', 'rule_gap>0.35'):
        p, r, n = s[rule]
        print(f'   {rule[5:]:16s} flags {n:4d}: precision {p:.2f}, recall {r:.2f} '
              f'(advantage > 2x)')
    s['ci'] = ci
    s['diff'] = d
    return s


def write_table(results):
    cols = [c for c in ('registered', 'extended', 'new') if c in results]
    head = {'registered': 'registered', 'extended': 'extended (seen)',
            'new': 'new (blind)'}
    def cell(s, k):
        lo, hi = s['ci'][k]
        return f'${s[k]:+.2f}$ {{\\scriptsize $[{lo:+.2f}, {hi:+.2f}]$}}'
    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \footnotesize',
             r'  \caption{Which quantity predicts Logit-LIME''s advantage (standard LIME''s '
             r'local Brier score over Logit-LIME''s). $\rho$ is Spearman''s rank correlation '
             r'with the advantage, bracketed by a $95\%$ interval from resampling '
             r'\emph{datasets}, since the black boxes fitted to one dataset are not '
             r'independent. The registered and extended grids were seen before $\Rlogit$ was '
             r'chosen; the new grid, $42$ datasets added afterwards, was not. Precision and '
             r'recall are for flagging an advantage above $2\times$.}',
             r'  \label{tab:diagnostic}',
             r'  \begin{tabular}{@{}l' + 'r'*len(cols) + '@{}}', r'    \toprule',
             '    & ' + ' & '.join(head[c] for c in cols) + r' \\',
             '    configurations (datasets) & ' + ' & '.join(
                 f"${results[c]['n']}$ (${results[c]['n_datasets']}$)" for c in cols) + r' \\',
             r'    \midrule']
    for k, lab in [('gap', r'$\rho$, $\Delta = \Rlogit - \Rprob$'),
                   ('r2_logit', r'$\rho$, $\Rlogit$'), ('r2_prob', r'$\rho$, $\Rprob$')]:
        lines.append(f'    {lab} & ' + ' & '.join(cell(results[c], f'rho_{k}') for c in cols)
                     + r' \\')
    lines.append(r'    partial $\rho$, $\Rprob$ given $\Rlogit$ & ' + ' & '.join(
        f"${results[c]['partial_r2_prob|r2_logit']:+.2f}$" for c in cols) + r' \\')
    lines.append(r'    \midrule')
    for k, lab in [('gap', r'AUC, $\Delta$'), ('r2_logit', r'AUC, $\Rlogit$')]:
        lines.append(f'    {lab} & ' + ' & '.join(f"${results[c][f'auc_{k}']:.2f}$"
                                                   for c in cols) + r' \\')
    for rule, lab in [('rule_gap>0.35', r'$\Delta > 0.35$: precision / recall'),
                      ('rule_r2_logit>0.95', r'$\Rlogit > 0.95$: precision / recall')]:
        lines.append(f'    {lab} & ' + ' & '.join(
            f"${results[c][rule][0]:.2f}$ / ${results[c][rule][1]:.2f}$" for c in cols) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    open(paths.table('diagnostic.tex'), 'w').write('\n'.join(lines))
    print('\nwritten', paths.table('diagnostic.tex'))


def main():
    results = {}
    for name, (rows, excluded) in sets().items():
        results[name] = report(name, rows, excluded)
    write_table(results)
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
              open(paths.results('analysis_diagnostic.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
