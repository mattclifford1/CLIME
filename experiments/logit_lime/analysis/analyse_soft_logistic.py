'''
The sixth registration, scored: Logit-LIME's model class fitted by cross-entropy against
the black box's probabilities (PREREGISTRATION.md), and tables/soft-logistic.tex.

The soft-label surrogate was run alone (sweeps/sweep_soft_logistic.py). Everything it is
compared with is already on disk for the same query points and evaluation samples, and is
joined here:

    standard LIME, Logit-LIME, hard-label: local Brier and KL   results_full.json
    their cosine to grad logit f(q)                            results_gradient_truth_full.json
    their fidelity cells (CIKM'23 protocol)                    results_fidelity_full.json
    alpha = 0.01 refits of standard LIME and Logit-LIME        results_ridge_alpha.json

The join is checked, not assumed: the soft sweep refitted standard LIME at each
configuration's first query point, and that score must equal the stored one exactly.

Two grids, never pooled:
    registered   14 datasets x the 12 registered black boxes (x the 11 differentiable ones
                 for S5). 31 of the 168 were seen before the registered run - 28 in the
                 review probe, 3 Moons configurations in a smoke test of the sweep - so the
                 test is the other 137 and the 31 are reported alongside
    full         every other configuration of the 71 x 16 grid (x the 11 differentiable
                 black boxes for S5): the 57 further datasets, and the registered datasets
                 x the 4 black boxes added after registration, less the probe-seen Nearest
                 Class Mean configurations

R²_logit (S6) uses the guarded rule of the fifth registration on both grids
(diagnostic_stats, at least MIN_DEFINED of 20 query points defined).

usage:  python analysis/analyse_soft_logistic.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from scipy.stats import spearmanr
from analysis import diagnostic_stats as ds
from sweeps import sweep, full_grid
from common import gradients

SOFT_FILE = 'results_soft_logistic_full.json'
STD, LOGIT, HARD = 'bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)'
BRIER, KL = 'Brier score (local)', 'KL divergence (local)'
REGISTERED_MODELS = list(sweep.MODELS)          # sweep.py's own lists: never configure()d here
GROUP_ORDER = ['A linear', 'B quadratic', 'C smooth', 'D piecewise constant',
               'E calibrated forest', 'unassigned']
GROUP_LABEL = {'A linear': 'A linear log-odds', 'B quadratic': 'B quadratic',
               'C smooth': 'C smooth', 'D piecewise constant': 'D piecewise constant',
               'E calibrated forest': 'E calibrated forest',
               'unassigned': 'unassigned'}

PROBE_DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication',
                  'Pima Indian Diabetes']
PROBE_MODELS = ['Logistic', 'MLP', 'SVM', 'Random Forest', 'Random Forest (Platt calibrated)',
                'Decision Tree', 'k Nearest Neighbours', 'Nearest Class Mean']
# seen before the registered run: the review probe, and a smoke test of the sweep that
# printed the new surrogate's own scores (not the comparison) for three more
SEEN = ({(d, m) for d in PROBE_DATASETS for m in PROBE_MODELS}
        | {('Moons', m) for m in ('Logistic', 'Random Forest', 'SVM')})


def load_json(name):
    return json.load(open(paths.results(name)))


def finite(*v):
    return all(np.isfinite(x) for x in v)


def frac(flags):
    flags = list(flags)
    return float(np.mean(flags)) if flags else float('nan')


def rows():
    '''one row per configuration with a soft-label result, everything joined to it'''
    soft = {k: v for k, v in load_json(SOFT_FILE).items()
            if not k.startswith('_') and 'error' not in v}
    full = load_json('results_full.json')
    grad = load_json('results_gradient_truth_full.json')
    fid = load_json('results_fidelity_full.json')
    meta = load_json('dataset_meta.json')
    diag_rows = {(r['dataset'], r['model']): r
                 for r in ds.load(paths.results('results_full.json'), meta)}

    out, join_err = [], 0.0
    for key, s in soft.items():
        dataset, model = key.split('|')
        r = dict(dataset=dataset, model=model, group=s['group'],
                 family=full_grid.FAMILY_OF[dataset], seen=(dataset, model) in SEEN,
                 soft_brier=s['mean Brier | local sample'], soft_kl=s['mean KL | local sample'],
                 soft_brier_weak=s['mean Brier | local sample | C=50'],
                 soft_kl_weak=s['mean KL | local sample | C=50'],
                 soft_cos=s.get('mean cos', float('nan')),
                 soft_fid_test=s['mean fidelity | test data'],
                 soft_fid_local=s['mean fidelity | local sample'],
                 unconverged=sum(not p['converged'] for p in s['points']),
                 constant=sum(p['constant'] for p in s['points']),
                 soft_brier_points=[p['Brier | local sample'] for p in s['points']])
        f = full.get(key)
        if f is not None and 'error' not in f:
            b, k = f['metrics'][BRIER], f['metrics'][KL]
            for tag, e in (('std', STD), ('logit', LOGIT), ('hard', HARD)):
                r[f'{tag}_brier'] = b[e]['mean']
                r[f'{tag}_kl'] = k[e]['mean']
            join_err = max(join_err, abs(b[STD]['scores'][0] - s['check_standard_brier_q0']))
            r['sat'] = f['diagnostic']['saturation']
            d = diag_rows.get((dataset, model))
            if d is not None:
                r.update(r2_logit_g=d.get('r2_logit_g', np.nan), n_def=d.get('n_def', 0),
                         r2_logit_paper=d['r2_logit'], gap_paper=d['gap'])
        g = grad.get(key)
        if g is not None and 'error' not in g and g.get('points'):
            for tag, lab in (('std', 'standard'), ('logit', 'logit'), ('hard', 'logreg')):
                r[f'{tag}_cos'] = g[f'cos_{lab}']
        fd = fid.get(key)
        if fd is not None and 'error' not in fd:
            for tag, e in (('std', STD), ('logit', LOGIT), ('hard', HARD)):
                r[f'{tag}_fid_test'] = fd['cells']['fidelity | test data'][e]['mean']
                r[f'{tag}_fid_local'] = fd['cells']['fidelity | local sample'][e]['mean']
            # Bayes Optimal is outside results_full.json; its Brier comes from here
            if 'std_brier' not in r:
                c = fd['cells']['Brier | local sample']
                for tag, e in (('std', STD), ('logit', LOGIT), ('hard', HARD)):
                    r[f'{tag}_brier'] = c[e]['mean']
        out.append(r)
    return out, join_err


def grid_split(all_rows):
    reg = [r for r in all_rows if r['family'] == 'registered']
    return {'registered': [r for r in reg if r['model'] in REGISTERED_MODELS],
            'registered, differentiable': [r for r in reg if gradients.has_gradient(r['model'])],
            'full': [r for r in all_rows if r['model'] in full_grid.MODELS
                     and not (r['family'] == 'registered' and r['model'] in REGISTERED_MODELS)],
            'full, differentiable': [r for r in all_rows if r['family'] != 'registered'
                                     and gradients.has_gradient(r['model'])]}


def predictions(main, diff, rule):
    '''S1-S7 on one set of rows; `rule` decides where R²_logit is defined (S6)'''
    m = [r for r in main if finite(r.get('std_brier', np.nan), r['soft_brier'])]
    s1a = frac(r['soft_brier'] < r['std_brier'] for r in m)
    d_adv = [r['std_brier']/max(r['soft_brier'], 1e-30) for r in m
             if r['group'] == 'D piecewise constant']
    s1b = float(np.median(d_adv)) if d_adv else float('nan')
    d_rows = [r for r in m if r['group'] == 'D piecewise constant']
    d_wins = sum(r['soft_brier'] < r['std_brier'] for r in d_rows)
    d_logit_adv = (float(np.median([r['std_brier']/max(r['logit_brier'], 1e-30) for r in d_rows]))
                   if d_rows else float('nan'))
    d_logit_wins = sum(r['logit_brier'] < r['std_brier'] for r in d_rows)
    d_kl_wins = sum(r['soft_kl'] < r['logit_kl'] for r in d_rows)
    d_kl_ratio = (float(np.median([r['logit_kl']/max(r['soft_kl'], 1e-30) for r in d_rows]))
                  if d_rows else float('nan'))
    s2 = frac(r['soft_kl'] < r['logit_kl'] for r in m)
    logit_wins = [r for r in m if r['logit_brier'] < r['soft_brier']]
    s3 = frac(r['group'] == 'A linear' for r in logit_wins)
    s4 = frac(r['soft_kl'] < r['hard_kl'] for r in m)
    dd = [r for r in diff if finite(r['soft_cos'], r.get('std_cos', np.nan),
                                    r.get('logit_cos', np.nan))]
    cos = {t: float(np.mean([r[f'{t}_cos'] for r in dd])) for t in ('std', 'logit', 'soft')}
    s5b = frac(r['soft_cos'] > r['std_cos'] for r in dd)
    nobo = [r for r in dd if r['model'] != 'Bayes Optimal']
    s5_nobo = {'n': len(nobo),
               'cos': {t: float(np.mean([r[f'{t}_cos'] for r in nobo])) for t in ('std', 'logit', 'soft')},
               'better_than_std': frac(r['soft_cos'] > r['std_cos'] for r in nobo)}
    # not registered: how much of S5's failure is a near-tie, and the comparison with Logit-LIME
    near_tie = frac(abs(r['soft_cos'] - r['std_cos']) < 1e-3 for r in dd)
    beats_logit = frac(r['soft_cos'] > r['logit_cos'] for r in dd)
    logit_beats_std = frac(r['logit_cos'] > r['std_cos'] for r in dd)
    # S1's losses that are numerically nothing: standard LIME and the soft-label surrogate
    # both below 1e-9 (a tie at 0 included). The hard-label surrogate need not be
    s1_losses = [r for r in m if r['soft_brier'] >= r['std_brier']]
    s1_null = [r for r in s1_losses if max(r['std_brier'], r['soft_brier']) < 1e-9]
    s1_band = [r['std_brier']/max(r['soft_brier'], 1e-30) for r in s1_losses if r not in s1_null]
    if rule == 'paper':
        ok = [r for r in m if np.isfinite(r.get('gap_paper', np.nan))
              and abs(r['gap_paper']) <= 1]
        r2 = [r['r2_logit_paper'] for r in ok]
    else:
        ok = [r for r in m if r.get('n_def', 0) >= ds.MIN_DEFINED]
        r2 = [r['r2_logit_g'] for r in ok]
    s6 = float(spearmanr(r2, [r['std_brier']/max(r['soft_brier'], 1e-30) for r in ok])[0])
    s7 = float(spearmanr([r['sat'] for r in m],
                         [r['logit_kl']/max(r['soft_kl'], 1e-30) for r in m])[0])
    return {
        'n': len(m), 'n_diff': len(dd), 'n_logit_wins': len(logit_wins), 'n_s6': len(ok),
        'S1': (f'better Brier than standard in {s1a:.1%} (>= 90%); group D median '
               f'advantage {s1b:.3g} (> 1)', s1a >= 0.9 and s1b > 1),
        'S2': (f'better KL than Logit-LIME in {s2:.1%} (>= 70%)', s2 >= 0.7),
        'S3': (f'{s3:.1%} of the {len(logit_wins)} Logit-LIME wins are group A (>= 60%)',
               s3 >= 0.6),
        'S4': (f'better KL than hard-label in {s4:.1%} (>= 90%)', s4 >= 0.9),
        'S5': (f"mean cosine soft {cos['soft']:.3f} vs Logit-LIME {cos['logit']:.3f} "
               f"(>= -0.02) and standard {cos['std']:.3f}; better than standard in "
               f'{s5b:.1%} (>= 85%)', cos['soft'] >= cos['logit'] - 0.02 and s5b >= 0.85),
        'S6': (f'rho(R2_logit, advantage over standard) {s6:+.3f} (>= 0.5)', s6 >= 0.5),
        'S7': (f'rho(saturation, Logit-LIME KL / soft KL) {s7:+.3f} (>= +0.3)', s7 >= 0.3),
        'values': dict(s1a=s1a, s1b=s1b, s2=s2, s3=s3, s4=s4, cos=cos, s5b=s5b, s6=s6, s7=s7,
                       near_tie=near_tie, beats_logit=beats_logit,
                       group_d=dict(n=len(d_rows), soft_adv=s1b, soft_wins=d_wins,
                                    logit_adv=d_logit_adv, logit_wins=d_logit_wins,
                                    kl_wins=d_kl_wins, kl_ratio=d_kl_ratio),
                       s5_without_bayes_optimal=s5_nobo,
                       logit_beats_std=logit_beats_std, s1_losses=len(s1_losses),
                       s1_numerically_null=len(s1_null),
                       s1_loss_band=(min(s1_band), max(s1_band)) if s1_band else None,
                       s1_loss_datasets=sorted({r['dataset'] for r in s1_losses
                                                if r not in s1_null})),
    }


def report(name, p):
    print(f"\n--- {name}: {p['n']} configurations ({p['n_diff']} differentiable) ---")
    for s in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'):
        text, held = p[s]
        print(f"  {s}  {'HELD  ' if held else 'FAILED'}  {text}")
    v = p['values']
    print(f"      (not registered) S1 losses {v['s1_losses']}, of which numerically null "
          f"{v['s1_numerically_null']}; the rest at {v['s1_loss_band']} on {v['s1_loss_datasets']}")
    gd = v['group_d']
    print(f"      (not registered) group D: soft {gd['soft_adv']:.3g}x better in {gd['soft_wins']}/{gd['n']}; "
          f"Logit-LIME {gd['logit_adv']:.3g}x better in {gd['logit_wins']}/{gd['n']}; soft lower KL than "
          f"Logit-LIME in {gd['kl_wins']}/{gd['n']} by a median {gd['kl_ratio']:.3g}x")
    nb = v['s5_without_bayes_optimal']
    print(f"      (not registered) S5 without Bayes Optimal (n={nb['n']}): cosine soft {nb['cos']['soft']:.3f} "
          f"logit {nb['cos']['logit']:.3f} std {nb['cos']['std']:.3f}; better than standard in "
          f"{nb['better_than_std']:.1%}")
    print(f"      (not registered) cosine within 1e-3 of standard's in {v['near_tie']:.1%}; "
          f"soft beats Logit-LIME's explanation in {v['beats_logit']:.1%}; "
          f"Logit-LIME beats standard in {v['logit_beats_std']:.1%}")


def losses(main, diff, label):
    '''where the soft-label surrogate loses, by group (not registered; reads S1, S4, S5)'''
    print(f'\n  where it loses, {label}')
    for g in GROUP_ORDER:
        sub = [r for r in main if r['group'] == g]
        sd = [r for r in diff if r['group'] == g and finite(r['soft_cos'], r.get('std_cos', np.nan))]
        if not sub:
            continue
        print(f"    {g:22s} worse Brier than standard {sum(r['soft_brier'] >= r['std_brier'] for r in sub):3d}/{len(sub):<3d}"
              f" worse KL than hard {sum(r['soft_kl'] >= r['hard_kl'] for r in sub):3d}/{len(sub):<3d}"
              + (f" explanation not better than standard {sum(r['soft_cos'] <= r['std_cos'] for r in sd):3d}/{len(sd):<3d}"
                 f" (median cos diff {np.median([r['soft_cos'] - r['std_cos'] for r in sd]):+.4f})" if sd else ''))


def by_group(main, diff):
    out = {}
    for g in GROUP_ORDER:
        sub = [r for r in main if r['group'] == g]
        if not sub:
            continue
        sd = [r for r in diff if r['group'] == g and finite(r['soft_cos'], r.get('std_cos',
                                                                                 np.nan))]
        out[g] = {
            'n': len(sub),
            'adv_logit': float(np.median([r['std_brier']/max(r['logit_brier'], 1e-30)
                                          for r in sub])),
            'adv_soft': float(np.median([r['std_brier']/max(r['soft_brier'], 1e-30)
                                         for r in sub])),
            'wins_logit': sum(r['logit_brier'] < r['std_brier'] for r in sub),
            'wins_soft': sum(r['soft_brier'] < r['std_brier'] for r in sub),
            'kl_soft_vs_logit': float(np.median([r['logit_kl']/max(r['soft_kl'], 1e-30)
                                                 for r in sub])),
            'kl_wins_vs_logit': sum(r['soft_kl'] < r['logit_kl'] for r in sub),
            'kl_wins_vs_hard': sum(r['soft_kl'] < r['hard_kl'] for r in sub),
            'n_diff': len(sd),
            'cos': {t: (float(np.mean([r[f'{t}_cos'] for r in sd])) if sd else float('nan'))
                    for t in ('std', 'logit', 'soft')},
            'sat': float(np.median([r['sat'] for r in sub])),
        }
    return out


def fmt_ratio(x):
    if not np.isfinite(x):
        return '--'
    if x >= 1000:
        m, e = f'{x:.1e}'.split('e')
        return f'${m}\\!\\cdot\\!10^{{{int(e)}}}$'
    return f'${x:.3g}$' if x < 100 else f'${x:.0f}$'


def write_table(groups, groups_full, n_reg, n_full):
    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \scriptsize',
        r'  \setlength{\tabcolsep}{3.5pt}',
        r"  \caption{Logit-LIME's model class fitted two ways. \emph{Advantage} is standard "
        r"LIME's local Brier score divided by the surrogate's (median over configurations; "
        r'$>1$ favours the surrogate), with the count of configurations it improves on '
        r'standard LIME. \emph{KL vs logit} is Logit-LIME\textquoteright s local KL divided by '
        r"the soft-label surrogate's (median), with the count where the soft-label surrogate "
        r'is better; \emph{vs hard} counts where it beats the hard-label surrogate on KL. '
        r'\emph{Cosine} is the mean cosine of the explanation to $\nabla\logit f(q)$ '
        r'(standard / Logit-LIME / soft-label) over the $n_\nabla$ configurations of the '
        r"group's differentiable black boxes. These are S5's population, not the row's: on the "
        r'registered datasets they include the five differentiable black boxes added after '
        r'registration (Nearest Class Mean, the polynomial, RBF and bagged logistic models and '
        r"the Gaussian class-conditional model), so the unassigned row's cosine is the bagged "
        r"logistic's alone. Registered grid: " + f'${n_reg}$' + r' configurations, $31$ of them '
        r'seen before the sixth registration was scored ($28$ in a probe, $3$ in a smoke test). '
        r'Full grid: the other ' + f'${n_full}$' + r' configurations of the '
        r'$71\times16$ grid that have results and were not seen in the probe.}',
        r'  \label{tab:softlabel}',
        r'  \begin{tabular}{@{}lr rr rr rrr rc@{}}',
        r'    \toprule',
        r'    & & \multicolumn{2}{c}{advantage (better)} & '
        r'\multicolumn{2}{c}{KL vs logit} & & & & & \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}',
        r'    group & $n$ & Logit-LIME & soft-label & ratio & better & vs hard & '
        r'sat. & & $n_\nabla$ & cosine \\',
    ]
    for title, gs in (('registered grid', groups), ('full grid, the other configurations',
                                                     groups_full)):
        lines += [r'    \midrule', rf'    \multicolumn{{11}}{{@{{}}l}}{{\emph{{{title}}}}} \\']
        for g in GROUP_ORDER:
            if g not in gs:
                continue
            v = gs[g]
            c = v['cos']
            cos = ('/'.join(f'{c[t]:.2f}' for t in ('std', 'logit', 'soft'))
                   if np.isfinite(c['soft']) else '--')
            lines.append(
                f"    {GROUP_LABEL[g]} & {v['n']} & {fmt_ratio(v['adv_logit'])} "
                f"({v['wins_logit']}) & {fmt_ratio(v['adv_soft'])} ({v['wins_soft']}) & "
                f"{fmt_ratio(v['kl_soft_vs_logit'])} & {v['kl_wins_vs_logit']} & "
                f"{v['kl_wins_vs_hard']} & {100*v['sat']:.0f}\\% & & "
                f"{v['n_diff'] if v['n_diff'] else '--'} & {cos} \\\\")
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    open(paths.table('soft-logistic.tex'), 'w').write('\n'.join(lines))
    print('\nwritten', paths.table('soft-logistic.tex'))


def extras(main, label):
    '''unregistered: the weak penalty, fidelity, crowns, convergence'''
    print(f'\n  extras on {label} (not registered)')
    w = [r for r in main if finite(r['soft_brier_weak'], r['soft_kl_weak'])]
    print(f"    C = 50 vs C = 0.5: median Brier ratio (0.5 / 50) "
          f"{np.median([r['soft_brier']/max(r['soft_brier_weak'], 1e-30) for r in w]):.3g}; "
          f"C = 50 better than standard (alpha = 1) in "
          f"{frac(r['soft_brier_weak'] < r['std_brier'] for r in w):.1%}")
    f = [r for r in main if finite(r.get('std_fid_test', np.nan), r['soft_fid_test'])]
    if f:
        print(f"    fidelity | test data: soft better than standard in "
              f"{frac(r['soft_fid_test'] > r['std_fid_test'] for r in f):.1%}, tied "
              f"{frac(r['soft_fid_test'] == r['std_fid_test'] for r in f):.1%}; median "
              f"difference {np.median([r['soft_fid_test'] - r['std_fid_test'] for r in f]):+.4f}")
        for ins, key in (('fidelity | test data', 'fid_test'), ('KL', 'kl')):
            lower = ins == 'KL'
            crowns = {t: 0 for t in ('std', 'logit', 'hard', 'soft')}
            for r in f:
                vals = {t: r.get(f'{t}_{key}', np.nan) for t in crowns}
                if not all(np.isfinite(v) for v in vals.values()):
                    continue
                best = (min if lower else max)(vals, key=vals.get)
                crowns[best] += 1
            print(f'    crowns under {ins}: ' + ', '.join(f'{t} {n}' for t, n in crowns.items()))
    print(f"    unconverged lbfgs fits: {sum(r['unconverged'] for r in main)} points; "
          f"one-class fallback: {sum(r['constant'] for r in main)} points "
          f"of {20*len(main)}")


def weak_penalty_registered(all_rows):
    '''soft at C = 50 against Logit-LIME at alpha = 0.01, the same nominal penalty'''
    ra = load_json('results_ridge_alpha.json')
    pairs = []
    for r in all_rows:
        e = ra.get(f"{r['dataset']}|{r['model']}")
        if r['family'] != 'registered' or e is None or 'error' in e:
            continue
        lk = np.mean(e['scores'][KL]['logit']['0.01'])
        sk = np.mean(e['scores'][KL]['standard']['0.01'])
        pairs.append((r, lk, sk))
    if pairs:
        print(f"\n  weak penalty, registered grid (not registered): soft C = 50 better KL than "
              f"Logit-LIME alpha = 0.01 in {frac(r['soft_kl_weak'] < lk for r, lk, _ in pairs):.1%}"
              f" of {len(pairs)}; than standard alpha = 0.01 in "
              f"{frac(r['soft_kl_weak'] < sk for r, _, sk in pairs):.1%}")


def main():
    all_rows, join_err = rows()
    print(f'join check: standard LIME refitted at query point 0 differs from the stored '
          f'score by at most {join_err:.3g}')
    assert join_err == 0.0, 'the soft sweep did not see the same neighbourhoods'
    sets = grid_split(all_rows)
    reg, reg_d = sets['registered'], sets['registered, differentiable']
    full, full_d = sets['full'], sets['full, differentiable']
    print(f'registered: {len(reg)} (+{len(reg_d)} differentiable); full beyond it: '
          f'{len(full)} (+{len(full_d)})')

    results = {}
    for name, main_rows, diff_rows, rule in (
            ('registered, blind', [r for r in reg if not r['seen']],
             [r for r in reg_d if not r['seen']], 'guarded'),
            ('registered, seen (probe and smoke test)', [r for r in reg if r['seen']],
             [r for r in reg_d if r['seen']], 'guarded'),
            ('full grid beyond the registered datasets', [r for r in full if not r['seen']],
             [r for r in full_d if not r['seen']], 'guarded')):
        p = predictions(main_rows, diff_rows, rule)
        report(name, p)
        results[name] = {k: (v if k in ('values',) or not isinstance(v, tuple)
                             else {'text': v[0], 'held': bool(v[1])}) for k, v in p.items()}
    losses(reg, reg_d, 'the registered grid')
    losses(full, full_d, 'the full grid')
    extras(reg, 'the registered grid')
    extras(full, 'the full grid')
    weak_penalty_registered(all_rows)

    # the full-grid block shows the configurations it is scored on: the seen ones (the four
    # probe-seen Nearest Class Mean rows) are left out, as in predictions()
    full_blind = [r for r in full if not r['seen']]
    full_d_blind = [r for r in full_d if not r['seen']]
    g_reg, g_full = by_group(reg, reg_d), by_group(full_blind, full_d_blind)
    print('\nBY GROUP, registered grid')
    for g, v in g_reg.items():
        print(f"  {g:22s} n={v['n']:3d} adv logit {v['adv_logit']:9.3g} ({v['wins_logit']:2d}) "
              f"soft {v['adv_soft']:9.3g} ({v['wins_soft']:2d})  KL logit/soft "
              f"{v['kl_soft_vs_logit']:8.3g} ({v['kl_wins_vs_logit']:2d})  vs hard "
              f"{v['kl_wins_vs_hard']:2d}  cos {v['cos']}")
    write_table(g_reg, g_full, len(reg), len(full_blind))
    json.dump({'results': results, 'groups_registered': g_reg, 'groups_full': g_full,
               'join_error': join_err},
              open(paths.results('analysis_soft_logistic.json'), 'w'), indent=1, default=float)
    print('written', paths.results('analysis_soft_logistic.json'))


if __name__ == '__main__':
    main()
