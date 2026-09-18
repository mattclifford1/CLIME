'''
Does the probability reading expire inside its own neighbourhood? (-> tables/range.tex)

Scores the fourth pre-registration against results_range.json.  Four questions, in the
order PREREGISTRATION.md asks them:

  P1  how much of the surrogate's own training sample does its unclipped output describe
      with a number that is not a probability?
  P2  how often is the reach - the distance at which that happens - shorter than the
      locality kernel width that defined the neighbourhood in the first place?
  P3  for the black boxes whose true local importance is CONSTANT along the query line
      (group A: exactly linear log-odds), how much does each surrogate's reported
      ||beta|| move anyway?
  P4  for the same black boxes, how wrong is the counterfactual a coefficient implies -
      the distance along a feature at which the decision flips - and does the error grow
      with the black box's confidence?

P3 and P4 are restricted to group A because they need a truth that is known not to move,
and only exactly-linear log-odds supply one.  P4 is further restricted to points where the
black box crosses its own boundary exactly once along that feature within the range
searched: where it crosses more than once there is no single flip distance to be right
about, and where it crosses none the counterfactual does not exist.  Both restrictions cost
points and both are counted here rather than assumed away.

Saturated points - where the black box's probability is 0 or 1 to within logit_ridge's
squash bound - are reported separately throughout, because that is where the logit
surrogate's own coefficient drifts and the honest thing is to show both numbers.

usage:  python analysis/analyse_range.py [results_range.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from scipy.stats import spearmanr

GROUP_LABEL = {'A linear': 'A  linear log-odds',
               'B quadratic': 'B  quadratic',
               'C smooth': 'C  smooth',
               'D piecewise constant': 'D  piecewise constant',
               'E calibrated forest': 'E  calibrated forest',
               'unassigned': 'unassigned'}
SAT_MASS = 0.02          # a point counts as saturated if this much kernel mass is at the
                         # squash bound - the level at which the logit fit starts to bend


def load(path=None):
    d = json.load(open(paths.results(path or 'results_range.json')))
    return {k: v for k, v in d.items() if not k.startswith('_') and 'error' not in v}


def rows(data, group=None):
    '''every query point as a flat record, with its configuration attached'''
    out = []
    for cfg, entry in data.items():
        if group is not None and entry['group'] != group:
            continue
        for p in entry['points']:
            r = dict(p)
            r['config'] = cfg
            r['group'] = entry['group']
            r['k'] = entry['kernel_width']
            r['saturated'] = p['sat_mass'] >= SAT_MASS
            r['mass'] = p['mass_above_1'] + p['mass_below_0']
            out.append(r)
    return out


def quartiles(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return (float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)))


def span(values):
    '''largest over smallest, the factor by which a reported size moves'''
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    return float(v.max()/v.min()) if v.size >= 2 else float('nan')


def pct(x):
    return f'{100*x:.0f}\\%'


def main(path=None):
    data = load(path)
    all_rows = rows(data)
    unsat = [r for r in all_rows if not r['saturated']]
    n_cfg, n_pt = len(data), len(all_rows)

    print(f'\n{n_cfg} configurations, {n_pt} query points '
          f'({len(all_rows)-len(unsat)} saturated, {SAT_MASS:.0%} of kernel mass at the '
          f'squash bound)\n')

    # ---------------------------------------------------------------- P1 and P2, overall
    mass = [r['mass'] for r in all_rows]
    q1, med, q3 = quartiles(mass)
    tiny = float(np.mean([m < 0.02 for m in mass]))
    inside = float(np.mean([r['reach_over_k'] < 1 for r in all_rows]))
    print('P1  mass of its own training sample where g is not a probability')
    print(f'      median {med:.3f}   quartiles {q1:.3f} - {q3:.3f}   '
          f'below 0.02 at {tiny:.1%} of points')
    print(f'      registered: median >= 0.10  -> {"PASS" if med >= 0.10 else "FAIL"};  '
          f'below 0.02 at < 15% -> {"PASS" if tiny < 0.15 else "FAIL"}')
    print('P2  reach shorter than the locality kernel width')
    print(f'      {inside:.1%} of points   registered: >= 66.7% -> '
          f'{"PASS" if inside >= 2/3 else "FAIL"}')

    base_std = np.median([abs(r['g_std'] - r['f_q']) for r in all_rows])
    base_log = np.median([abs(r['g_log'] - r['f_q']) for r in all_rows])
    print(f'      median |g(q) - f(q)|:  standard {base_std:.4f}   '
          f'Logit-LIME {base_log:.4f}')

    # The defect is not uniform over the grid and should not be reported as if it were.
    # The slab's width is 1/||beta||, and ||beta|| is small wherever the black box never
    # commits - so a model that predicts ~0.5 everywhere has a wide slab and an intact
    # reading. Splitting by the black box's own confidence at q says whether the mass
    # appears exactly where explanations are actually asked for.
    conf_all = [abs(np.log(max(r['f_q'], 1e-300)/max(1 - r['f_q'], 1e-300)))
                for r in all_rows]
    edges = [0, 1, 2, 4, np.inf]                      # |logit f(q)|: p = .5, .73, .88, .98
    print('\n    by the black box\'s confidence at q (exploratory)')
    print(f'      {"|logit f(q)|":>14s} {"points":>7s} {"mass med":>9s} {"reach<k":>8s}')
    by_conf = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, c in zip(all_rows, conf_all) if lo <= c < hi]
        if not sel:
            continue
        m = float(np.median([r['mass'] for r in sel]))
        i = float(np.mean([r['reach_over_k'] < 1 for r in sel]))
        by_conf.append({'lo': lo, 'hi': None if hi == np.inf else hi,
                        'n': len(sel), 'mass_median': m, 'reach_inside_k': i})
        label = f'{lo:g}-{hi:g}' if hi != np.inf else f'{lo:g}+'
        print(f'      {label:>14s} {len(sel):7d} {m:9.3f} {i:7.1%}')
    rho_mass = spearmanr(conf_all, [r['mass'] for r in all_rows])[0]
    print(f'      Spearman(|logit f(q)|, mass) = {rho_mass:+.3f} over all '
          f'{len(all_rows)} points')

    # ------------------------------------------------------------------------- per group
    groups = sorted({e['group'] for e in data.values()})
    per_group = {}
    print(f"\n{'group':24s} {'cfgs':>5s} {'mass med':>9s} {'mass IQR':>15s} "
          f"{'reach<k':>8s} {'span std':>9s} {'span log':>9s}")
    for g in groups:
        gr = rows(data, g)
        gq1, gmed, gq3 = quartiles([r['mass'] for r in gr])
        gin = float(np.mean([r['reach_over_k'] < 1 for r in gr]))
        cfgs = sorted({r['config'] for r in gr})
        sp_std, sp_log = [], []
        for c in cfgs:
            pts = [r for r in gr if r['config'] == c and not r['saturated']]
            if len(pts) >= 2:
                sp_std.append(span([p['norm_std'] for p in pts]))
                sp_log.append(span([p['norm_log'] for p in pts]))
        per_group[g] = {'n': len(cfgs), 'mass': (gq1, gmed, gq3), 'inside': gin,
                        'span_std': float(np.nanmedian(sp_std)) if sp_std else float('nan'),
                        'span_log': float(np.nanmedian(sp_log)) if sp_log else float('nan')}
        print(f'{GROUP_LABEL.get(g, g):24s} {len(cfgs):5d} {gmed:9.3f} '
              f'{gq1:7.3f}-{gq3:<7.3f} {gin:7.1%} '
              f'{per_group[g]["span_std"]:9.1f} {per_group[g]["span_log"]:9.2f}')

    # --------------------------------------------------------------- P3, group A configs
    a_all = rows(data, 'A linear')
    a_rows = [r for r in a_all if not r['saturated']]
    a_cfgs = sorted({r['config'] for r in a_all})
    big_std, small_log = [], []
    for c in a_cfgs:
        pts = [r for r in a_rows if r['config'] == c]
        if len(pts) < 2:
            continue
        big_std.append(span([p['norm_std'] for p in pts]) >= 5)
        small_log.append(span([p['norm_log'] for p in pts]) <= 1.5)
    f_std, f_log = float(np.mean(big_std)), float(np.mean(small_log))
    print(f'\nP3  group A ({len(big_std)} of {len(a_cfgs)} configurations; saturated '
          f'points dropped, and a configuration with fewer than two left drops out)')
    print(f'      standard ||beta|| spans >= 5x in {f_std:.0%} of configurations   '
          f'registered >= 90% -> {"PASS" if f_std >= 0.9 else "FAIL"}')
    print(f'      Logit-LIME spans <= 1.5x in {f_log:.0%}   '
          f'registered >= 90% -> {"PASS" if f_log >= 0.9 else "FAIL"}')

    # Diagnosis, exploratory: the registered form of P3 turns on two choices it should not
    # have. Dropping saturated points - done to protect the logit surrogate from its own
    # squash bound - removes exactly the confident points where the standard coefficient
    # collapses; and a black box that never becomes confident along the query line has no
    # confidence range for a coefficient to track in the first place. Both are reported
    # rather than used to restate the prediction in a form that passes.
    print('\n    diagnosis (exploratory, not the registered form)')
    flat = []
    per_cfg = []
    for c in a_cfgs:
        pts = [r for r in a_all if r['config'] == c]
        uns = [r for r in pts if not r['saturated']]
        fr = max(p['f_q'] for p in pts) - min(p['f_q'] for p in pts)
        per_cfg.append({'config': c, 'f_range': float(fr),
                        'n_sat': sum(r['saturated'] for r in pts),
                        'span_std_all': span([p['norm_std'] for p in pts]),
                        'span_std_uns': span([p['norm_std'] for p in uns]),
                        'span_log_all': span([p['norm_log'] for p in pts]),
                        'span_log_uns': span([p['norm_log'] for p in uns])})
        if fr < 0.5:
            flat.append(c)
    all_std = [c['span_std_all'] for c in per_cfg]
    all_log = [c['span_log_all'] for c in per_cfg]
    print(f'      median span, all points:   standard {np.nanmedian(all_std):.1f}x   '
          f'Logit-LIME {np.nanmedian(all_log):.2f}x')
    print(f'      median span, unsaturated:  standard '
          f'{np.nanmedian([c["span_std_uns"] for c in per_cfg]):.1f}x   Logit-LIME '
          f'{np.nanmedian([c["span_log_uns"] for c in per_cfg]):.2f}x')
    print(f'      standard spans >= 5x over all points in '
          f'{np.nanmean([s >= 5 for s in all_std]):.0%} of configurations')
    if flat:
        print(f'      {len(flat)} configuration(s): f spans less than 0.5 over the whole '
              f'query line, so there is no confidence range for a coefficient to track: '
              + ', '.join(flat))
    for c in sorted(per_cfg, key=lambda x: -x['span_std_all'])[:4] + \
             sorted(per_cfg, key=lambda x: x['span_std_all'])[:2]:
        print(f"        {c['config']:38s} f range {c['f_range']:.3f}  "
              f"sat {c['n_sat']:2d}/20  span std {c['span_std_all']:8.1f} "
              f"(uns {c['span_std_uns']:7.1f})  log {c['span_log_all']:7.2f}")

    # --------------------------------------------------------------- P4, the flip distance
    flip = [r for r in a_rows
            if r.get('n_crossings') == 1 and np.isfinite(r.get('flip_scan', np.nan))
            and abs(r['flip_scan']) > 1e-9]
    dropped = len(a_rows) - len(flip)          # crossing count; saturation already gone
    dropped_sat = len(a_all) - len(a_rows)
    conf = [abs(np.log(max(r['f_q'], 1e-300)/max(1 - r['f_q'], 1e-300))) for r in flip]
    err_std = [abs(r['flip_std']/r['flip_scan']) for r in flip]
    err_log = [abs(r['flip_log']/r['flip_scan']) for r in flip]
    rho = spearmanr(conf, err_std)[0] if len(flip) > 2 else float('nan')
    rho_log = spearmanr(conf, err_log)[0] if len(flip) > 2 else float('nan')
    within_log = float(np.mean([abs(e - 1) <= 0.10 for e in err_log])) if flip else float('nan')
    within_std = float(np.mean([abs(e - 1) <= 0.10 for e in err_std])) if flip else float('nan')
    print(f'\nP4  group A flip distance ({len(flip)} points of {len(a_all)}: '
          f'{dropped_sat} dropped as saturated, then {dropped} for a crossing count '
          f'other than one)')
    print(f'      standard error ratio vs |logit f(q)|:  Spearman {rho:+.3f}   '
          f'registered >= +0.50 -> {"PASS" if rho >= 0.5 else "FAIL"}')
    print(f'      Logit-LIME within 10% of the truth at {within_log:.0%}   '
          f'registered >= 90% -> {"PASS" if within_log >= 0.9 else "FAIL"}')
    print(f'      standard within 10% at {within_std:.0%}   '
          f'(Logit-LIME error ratio vs confidence: Spearman {rho_log:+.3f})')
    if flip:
        lo, hi = np.percentile(err_std, [50, 90])
        print(f'      standard overstates the flip distance by {lo:.2f}x (median) '
              f'and {hi:.2f}x (90th percentile)')
        # the analytic flip is exact for group A; the scan is the check on it
        chk = [abs(r['flip_analytic'] - r['flip_scan']) for r in flip
               if np.isfinite(r.get('flip_analytic', np.nan))]
        if chk:
            print(f'      analytic vs scanned truth: max difference {max(chk):.2e} '
                  f'(scan step {2*8.0/3200:.4f})')

    # ------------------------------------------------------------------------- the table
    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \footnotesize',
        r'  \caption{How far a standard LIME coefficient can be carried, over the '
        f'{n_cfg} registered configurations and {n_pt} query points. '
        r'\emph{Not a probability} is the kernel-weighted fraction of the '
        "surrogate's own " + r'$10{,}000$-point training sample on which its unclipped output lies '
        r'outside $[0,1]$ (median, with quartiles). \emph{Reach $<k$} is the fraction of '
        r'query points at which that output leaves $[0,1]$ closer to $q$ than the locality '
        r'kernel width, so the reading expires inside the neighbourhood it describes. '
        r'The reading is intact where the black box never commits --- the slab is wide '
        r'when $\lVert\beta\rVert$ is small --- and narrow exactly where it is confident, '
        r'which is where an explanation is usually wanted (Spearman between '
        r'$|\logit f(q)|$ and the mass, ' + f'${rho_mass:+.2f}$' + r'). '
        r'\emph{Span} is the factor between the largest and smallest '
        r'$\lVert\beta\rVert$ a surrogate reports across the $20$ query points of one '
        r'configuration (median over configurations, saturated points excluded); for '
        r'group~A the true importance vector is constant along that line, so every unit '
        r'of span is the instrument moving and not the truth. Excluding saturated points '
        r'is conservative here, since it removes the confident points where the standard '
        r"coefficient collapses: over all points group~A's medians are "
        f'${np.nanmedian(all_std):.1f}' + r'\times$ and $'
        f'{np.nanmedian(all_log):.2f}' + r'\times$. The first three columns are '
        r'properties of the standard surrogate alone and involve no comparison.}',
        r'  \label{tab:range}',
        r'  \begin{tabular}{@{}lrrrrr@{}}',
        r'    \toprule',
        r'    & & \multicolumn{2}{c}{not a probability} & & '
        r'\multicolumn{1}{c}{span of $\lVert\beta\rVert$} \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){6-6}',
        r'    & points & median & quartiles & reach $<k$ & standard / logit \\',
        r'    \midrule',
        r'    \multicolumn{6}{@{}l}{\emph{by the black box\textquoteright s confidence at '
        r'$q$}} \\',
    ]
    # the confidence split goes first: it is the answer to "when does this bite", and the
    # honest reply to the configurations where the mass is near zero
    for b in by_conf:
        label = (rf"\quad $|\logit f(q)| \ge {b['lo']:g}$" if b['hi'] is None
                 else rf"\quad ${b['lo']:g} \le |\logit f(q)| < {b['hi']:g}$")
        lines.append(f"    {label} & {b['n']} & ${b['mass_median']:.3f}$ & & "
                     f"{pct(b['reach_inside_k'])} & \\\\")
    lines += [
        r'    \midrule',
        r'    \multicolumn{6}{@{}l}{\emph{by black box family}} \\',
    ]
    for g in groups:
        s = per_group[g]
        q1g, medg, q3g = s['mass']
        sp = (f"${s['span_std']:.1f}\\times$ / ${s['span_log']:.2f}\\times$"
              if np.isfinite(s['span_std']) else '---')
        lines.append(f"    \\quad {GROUP_LABEL.get(g, g)} & {20*s['n']} & ${medg:.2f}$ & "
                     f"${q1g:.2f}$--${q3g:.2f}$ & {pct(s['inside'])} & {sp} \\\\")
    lines += [
        r'    \midrule',
        f'    all & {n_pt} & ${med:.2f}$ & ${q1:.2f}$--${q3:.2f}$ & {pct(inside)} & '
        r'\\',
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    open(paths.table('range.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\nwritten tables/range.tex')

    json.dump({'n_configs': n_cfg, 'n_points': n_pt,
               'mass_median': med, 'mass_q1': q1, 'mass_q3': q3, 'mass_below_002': tiny,
               'reach_inside_k': inside, 'by_confidence': by_conf,
               'rho_confidence_mass': float(rho_mass),
               'base_error_std': float(base_std), 'base_error_log': float(base_log),
               'group': per_group,
               'P3_std_span_ge5': f_std, 'P3_log_span_le15': f_log,
               'P3_span_std_median_all': float(np.nanmedian(all_std)),
               'P3_span_log_median_all': float(np.nanmedian(all_log)),
               'P3_std_span_ge5_all_points':
                   float(np.nanmean([s >= 5 for s in all_std])),
               'P3_flat_configs': flat, 'P3_per_config': per_cfg,
               'P4_n': len(flip), 'P4_dropped': dropped, 'P4_rho': float(rho),
               'P4_log_within10': within_log, 'P4_std_within10': within_std,
               'P4_std_median_ratio': float(np.median(err_std)) if flip else None,
               'P4_std_p90_ratio': float(np.percentile(err_std, 90)) if flip else None},
              open(paths.results('range_summary.json'), 'w'), indent=1)
    print('written results/range_summary.json')


if __name__ == '__main__':
    main(*sys.argv[1:])
