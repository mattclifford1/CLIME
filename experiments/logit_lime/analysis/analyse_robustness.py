'''
Does the diagnostic hold up across the grid? One row per slice of the full grid:

    dataset family, dimension, class imbalance, sampling-covariance rank   (results_full.json)
    seed                                        (results_full.json + results_full_seed*.json)
    query-point placement                       (+ results_querypoints_full.json)
    kernel width                                (results_kernel_full.json)

For each: the number of configurations and datasets, Spearman ρ of R²_logit and of Δ with
Logit-LIME's advantage (each with a dataset-level cluster-bootstrap interval), and the
fraction of group A configurations where Logit-LIME is better. All use the guarded rule of
diagnostic_stats. Writes results/analysis_robustness.json (read by
figures/fig_robustness.py) and tables/robustness.tex.

usage:  python analysis/analyse_robustness.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from analysis import diagnostic_stats as ds
from sweeps import full_grid

GROUP_A = ['Logistic', 'LDA', 'Nearest Class Mean']
B = 1000


def kernel_rows(path):
    '''rows per (scale, dataset, model) from the kernel sweep, in load()'s shape'''
    d = json.load(open(path))
    by = {}
    for k, v in d.items():
        if k.startswith('_') or 'error' in v:
            continue
        scale, dataset, model = k.split('|')
        dg = v['diagnostic']
        by.setdefault(float(scale), []).append(dict(
            dataset=dataset, model=model,
            adv=v[ds.STD]['mean']/max(v[ds.LOGIT]['mean'], 1e-30),
            gap=dg['gap'], r2_logit=dg['r2_logit'], r2_prob=dg['r2_prob'],
            sat=dg['saturation'], r2_logit_g=dg.get('r2_logit_guarded', np.nan),
            r2_prob_g=dg.get('r2_prob_guarded', np.nan),
            n_def=min(dg.get('n_defined_logit', 0), dg.get('n_defined_prob', 0))))
    return by


def slice_stats(label, axis, rows):
    ok, excl = ds.usable(rows, 'guarded')
    a = [r for r in rows if r['model'] in GROUP_A]
    out = {'label': label, 'axis': axis, 'n': len(ok), 'excluded': excl,
           'n_datasets': len({r['dataset'] for r in ok}),
           'group_a_better': float(np.mean([r['adv'] > 1 for r in a])) if a else np.nan}
    for k in ('r2_logit', 'gap'):
        out[f'rho_{k}'] = ds.rho_stat(k)(ok) if len(ok) > 2 else np.nan
        out[f'ci_{k}'] = ds.cluster_bootstrap(ok, ds.rho_stat(k), B) \
            if out['n_datasets'] > 2 else (np.nan, np.nan)
    print(f"{axis:14s} {label:22s} n={out['n']:5d} ({out['n_datasets']:2d} ds, {excl:3d} excl)  "
          f"rho R2logit {out['rho_r2_logit']:+.2f} [{out['ci_r2_logit'][0]:+.2f},"
          f"{out['ci_r2_logit'][1]:+.2f}]  rho gap {out['rho_gap']:+.2f}  "
          f"A better {out['group_a_better']:.0%}", flush=True)
    return out


def main():
    meta = json.load(open(paths.results('dataset_meta.json')))
    full = ds.load(paths.results('results_full.json'), meta)
    slices = [slice_stats('all', 'grid', full)]

    for fam in full_grid.FAMILIES:
        slices.append(slice_stats(fam, 'family',
                                  [r for r in full if full_grid.FAMILY_OF[r['dataset']] == fam]))
    for lo, hi, lab in [(0, 5, 'd ≤ 5'), (6, 20, '6–20'), (21, 60, '21–60'),
                        (61, 10**6, '> 60')]:
        slices.append(slice_stats(lab, 'dimension',
                                  [r for r in full if lo <= r['meta_d'] <= hi]))
    for lo, hi, lab in [(0, .15, 'minority < 15%'), (.15, .35, '15–35%'), (.35, 1, '≥ 35%')]:
        slices.append(slice_stats(lab, 'imbalance',
                                  [r for r in full if lo <= r['meta_minority'] < hi]))
    for flag, lab in [(True, 'full rank'), (False, 'rank deficient')]:
        slices.append(slice_stats(lab, 'covariance',
                                  [r for r in full if r['meta_full_rank'] == flag]))

    seeds = {42: full}
    for sd in (1, 2, 3, 4):
        f = paths.results(f'results_full_seed{sd}.json')
        if os.path.exists(f):
            seeds[sd] = ds.load(f, meta)
    for sd, rows in seeds.items():
        slices.append(slice_stats(f'seed {sd}', 'seed', rows))

    slices.append(slice_stats('between class means', 'query points', full))
    f = paths.results('results_querypoints_full.json')
    if os.path.exists(f):
        slices.append(slice_stats('random test points', 'query points', ds.load(f, meta)))

    f = paths.results('results_kernel_full.json')
    if os.path.exists(f):
        for scale, rows in sorted(kernel_rows(f).items()):
            slices.append(slice_stats(f'k scale {scale:g}', 'kernel width', rows))

    json.dump(slices, open(paths.results('analysis_robustness.json'), 'w'), indent=1,
              default=float)
    write_table(slices)


def write_table(slices):
    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \scriptsize',
             r'  \setlength{\tabcolsep}{4pt}',
             r'  \caption{The diagnostic across the full grid ($71$ datasets $\times$ $16$ '
             r'black boxes). $\rho$ is Spearman''s correlation with Logit-LIME''s advantage, '
             r'with a $95\%$ interval from resampling datasets. ``A better'''' is the fraction '
             r'of group A configurations where Logit-LIME wins. Configurations whose black box '
             r'is constant to rounding over most neighbourhoods are excluded (``excl.'''').}',
             r'  \label{tab:robustness}',
             r'  \begin{tabular}{@{}llrrccr@{}}', r'    \toprule',
             r'    & slice & configs & excl. & $\rho$, $\Rlogit$ & $\rho$, $\Delta$ & A better \\',
             r'    \midrule']
    last = None
    for s in slices:
        axis = s['axis'] if s['axis'] != last else ''
        if s['axis'] != last and last is not None:
            lines.append(r'    \addlinespace')
        last = s['axis']
        lo, hi = s['ci_r2_logit']
        lab = s['label'].replace('%', r'\%').replace('≤', r'$\le$').replace('≥', r'$\ge$')
        lines.append(f"    {axis} & {lab} & ${s['n']}$ & ${s['excluded']}$ & "
                     f"${s['rho_r2_logit']:+.2f}$ {{\\scriptsize $[{lo:+.2f}, {hi:+.2f}]$}} & "
                     f"${s['rho_gap']:+.2f}$ & ${100*s['group_a_better']:.0f}\\%$ \\\\")
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    open(paths.table('robustness.tex'), 'w').write('\n'.join(lines))
    print('written', paths.table('robustness.tex'))


if __name__ == '__main__':
    main()
