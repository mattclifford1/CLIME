'''
Two questions from results_gradient_truth.json.

1. Which surrogate recovers the black box's true local importances, now that the question
   can be asked of every differentiable black box rather than only the three whose
   log-odds are exactly linear?

2. Does fidelity predict explanation correctness? The study measures Brier and KL, but
   what a user reads is a feature ranking, and the assumption that one tracks the other
   has never been checked. It can be checked here, because both are recorded from the
   same surrogate at the same query point.

The second is the load-bearing one: if a better Brier score did not come with a better
explanation, the rest of the study would be measuring something nobody wants.

usage:  python analysis/analyse_gradient_truth.py [results_gradient_truth.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from scipy.stats import spearmanr, wilcoxon

BRIER, KL = 'Brier score (local)', 'KL divergence (local)'
LABELS = {'standard': 'standard LIME', 'logit': 'Logit-LIME',
          'logreg': 'logistic-regression LIME'}
GROUP_ORDER = ['A linear', 'B quadratic', 'C smooth', 'unassigned']


def load(path=None):
    path = paths.results(path or 'results_gradient_truth.json')
    raw = json.load(open(path))
    rows = []
    for key, v in raw.items():
        if key.startswith('_') or 'error' in v or not v.get('points'):
            continue
        rows.append(dict(dataset=key.split('|')[0], model=key.split('|')[1], **v))
    return rows


def pooled(rows, field, label):
    v = [p[label][field] for r in rows for p in r['points']]
    return np.array([x for x in v if np.isfinite(x)])


def summarise(rows, title):
    print(f'\n{title}   ({len(rows)} configurations, '
          f'{sum(len(r["points"]) for r in rows)} query points)')
    print(f"{'surrogate':<28s} {'cos':>8s} {'rank rho':>10s} {'top-1':>8s} "
          f"{'Brier':>11s} {'KL':>11s}")
    for label, nice in LABELS.items():
        print(f'{nice:<28s} '
              f'{np.nanmean([r[f"cos_{label}"] for r in rows]):>8.3f} '
              f'{np.nanmean([r[f"rho_{label}"] for r in rows]):>10.3f} '
              f'{np.nanmean([r[f"top1_{label}"] for r in rows]):>8.2f} '
              f'{np.nanmean([r[f"{BRIER}_{label}"] for r in rows]):>11.2e} '
              f'{np.nanmean([r[f"{KL}_{label}"] for r in rows]):>11.2e}')


def paired(rows, field, a='standard', b='logit'):
    x = np.array([r[f'{field}_{a}'] for r in rows], dtype=float)
    y = np.array([r[f'{field}_{b}'] for r in rows], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    try:
        p = wilcoxon(y, x, zero_method='zsplit').pvalue
    except ValueError:
        p = float('nan')
    return int(np.sum(y > x)), int(np.sum(y == x)), len(x), float(np.median(y - x)), p


def fidelity_tracks_explanation(rows, metric):
    '''
    does the surrogate with the better fidelity give the better explanation?

    Reported three ways, because each can be criticised on its own:
      - agreement: per query point, do the two rankings name the same winner
      - paired rank correlation across configurations, of the fidelity gain against the
        explanation gain, which removes any between-configuration difficulty effect
      - pooled rank correlation within each surrogate, which does not
    '''
    agree, total = 0, 0
    material_agree, material_total = 0, 0
    for r in rows:
        for p in r['points']:
            fa, fb = p['standard'][metric], p['logit'][metric]
            ca, cb = p['standard']['cos'], p['logit']['cos']
            if not all(np.isfinite(v) for v in (fa, fb, ca, cb)) or fa == fb or ca == cb:
                continue
            total += 1
            same = int((fb < fa) == (cb > ca))
            agree += same
            # a point where both surrogates are within a hair of each other on both
            # counts is a coin toss that says nothing either way
            if abs(cb - ca) > 0.01 and max(fa, fb)/max(min(fa, fb), 1e-300) > 1.1:
                material_total += 1
                material_agree += same

    # the same question asked of whole configurations rather than single points
    cfg_agree, cfg_total = 0, 0
    for r in rows:
        fa, fb = r[f'{metric}_standard'], r[f'{metric}_logit']
        ca, cb = r['cos_standard'], r['cos_logit']
        if not all(np.isfinite(v) for v in (fa, fb, ca, cb)) or fa == fb or ca == cb:
            continue
        cfg_total += 1
        cfg_agree += int((fb < fa) == (cb > ca))

    d_fid, d_cos = [], []
    for r in rows:
        fa, fb = r[f'{metric}_standard'], r[f'{metric}_logit']
        ca, cb = r['cos_standard'], r['cos_logit']
        if min(fa, fb) <= 0 or not all(np.isfinite(v) for v in (fa, fb, ca, cb)):
            continue
        d_fid.append(np.log10(fa) - np.log10(fb))     # >0: logit fits better
        d_cos.append(cb - ca)                         # >0: logit explains better
    rho_paired = spearmanr(d_fid, d_cos)

    pool_f, pool_c = [], []
    for label in ('standard', 'logit'):
        f = pooled(rows, metric, label)
        c = pooled(rows, 'cos', label)
        n = min(len(f), len(c))
        keep = f[:n] > 0
        pool_f.append(np.log10(f[:n][keep]))
        pool_c.append(c[:n][keep])
    rho_pooled = spearmanr(np.concatenate(pool_f), np.concatenate(pool_c))

    print(f'\n--- does {metric} predict explanation correctness? ---')
    print(f'  DIRECTION')
    print(f'    same winner in {cfg_agree}/{cfg_total} configurations '
          f'({cfg_agree/cfg_total:.1%})')
    print(f'    same winner at {agree}/{total} query points ({agree/total:.1%}), '
          f'and {material_agree}/{material_total} ({material_agree/material_total:.1%}) '
          f'of the points where both differences are material')
    print(f'  LEVEL')
    print(f'    pooled over surrogates and points, rho(log {metric}, cos) = '
          f'{rho_pooled.statistic:+.3f} (p = {rho_pooled.pvalue:.1e}, '
          f'n = {len(np.concatenate(pool_c))}) - negative is the expected sign, since a '
          f'lower score is better')
    print(f'  MAGNITUDE')
    print(f'    paired across configurations, rho(fidelity gain, explanation gain) = '
          f'{rho_paired.statistic:+.3f} (p = {rho_paired.pvalue:.1e}, n = {len(d_fid)})')
    return agree/total, rho_paired


if __name__ == '__main__':
    rows = load(sys.argv[1] if len(sys.argv) > 1 else None)
    summarise(rows, 'ALL DIFFERENTIABLE BLACK BOXES')
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if sub:
            summarise(sub, f'group {g}')

    print('\nPAIRED, Logit-LIME against standard LIME')
    for field, nice in (('cos', 'cosine'), ('rho', 'rank rho'), ('top1', 'top-1')):
        w, t, n, med, p = paired(rows, field)
        print(f'  {nice:<9s} logit better on {w}/{n} (ties {t})  '
              f'median difference {med:+.3f}  Wilcoxon p = {p:.2g}')

    for metric in (BRIER, KL):
        fidelity_tracks_explanation(rows, metric)

    print('\n--- the instrument reduces to the coefficient one where both are defined ---')
    const = [r for r in rows if r.get('truth_is_constant')]
    print(f'  gradient constant in x (so equal to coef_) for {len(const)} of {len(rows)} '
          f'configurations')
    by_model = sorted({r['model'] for r in const})
    print(f'  models: {", ".join(by_model)}')

    print('\nBY BLACK BOX')
    print(f"{'model':<30s} {'n':>3s} {'cos std':>9s} {'cos logit':>10s} "
          f"{'top1 std':>9s} {'top1 logit':>11s}")
    for model in sorted({r['model'] for r in rows}):
        sub = [r for r in rows if r['model'] == model]
        print(f'{model:<30s} {len(sub):>3d} '
              f'{np.nanmean([r["cos_standard"] for r in sub]):>9.3f} '
              f'{np.nanmean([r["cos_logit"] for r in sub]):>10.3f} '
              f'{np.nanmean([r["top1_standard"] for r in sub]):>9.2f} '
              f'{np.nanmean([r["top1_logit"] for r in sub]):>11.2f}')

    # a surrogate that returns an all-zero coefficient vector has no direction to score,
    # which happens in degenerate neighbourhoods (FINDINGS.md B12, B13)
    for label in LABELS:
        n = sum(1 for r in rows if not np.isfinite(r[f'cos_{label}']))
        if n:
            print(f'  note: {LABELS[label]} has no scorable direction in {n} '
                  f'configuration(s)')
