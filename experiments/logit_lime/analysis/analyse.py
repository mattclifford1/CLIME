'''
Assess the pre-registered predictions (PREREGISTRATION.md) against the sweep results,
and emit the LaTeX tables for the paper.

usage:  python analyse.py [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import json
import glob
import numpy as np
from scipy.stats import spearmanr

E = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']
GROUP_ORDER = ['A linear', 'B quadratic', 'C smooth',
               'D piecewise constant', 'E calibrated forest', 'unassigned']
GROUP_LABEL = {'A linear': 'A. linear log-odds',
               'B quadratic': 'B. quadratic log-odds',
               'C smooth': 'C. smooth, non-polynomial',
               'D piecewise constant': 'D. piecewise constant',
               'E calibrated forest': 'E. calibrated forest',
               'unassigned': 'gradient boosting (unassigned)'}


def load(path):
    d = json.load(open(path))
    rows = []
    for key, v in d.items():
        if key.startswith('_') or 'error' in v:
            continue
        dataset, model = key.split('|')
        b = v['metrics']['Brier score (local)']
        k = v['metrics']['KL divergence (local)']
        rows.append(dict(
            dataset=dataset, model=model, group=v['group'],
            gap=v['diagnostic']['gap'], sat=v['diagnostic']['saturation'],
            brier_std=b[E[0]]['mean'], brier_logit=b[E[1]]['mean'], brier_lr=b[E[2]]['mean'],
            kl_std=k[E[0]]['mean'], kl_logit=k[E[1]]['mean'], kl_lr=k[E[2]]['mean'],
            adv=b[E[0]]['mean']/max(b[E[1]]['mean'], 1e-30),
            acc=v.get('model_stats', {}).get('test accurracy', float('nan'))))
    errors = {key: v['error'] for key, v in d.items()
              if not key.startswith('_') and 'error' in v}
    return rows, errors


# Both R^2 are at most 1, so a meaningful gap cannot exceed 1 either. Beyond that the
# logit fit has been driven by probability clipping rather than by any geometry.
GAP_PLAUSIBLE = 1.0


def degenerate(rows):
    '''
    Configurations where the gap is not a meaningful quantity, dropped from anything
    involving it and reported separately rather than silently imputed.

    Two cases:

    - the gap is not finite. The black box is constant over the whole neighbourhood, the
      probability target has zero variance and R^2_p is undefined.
    - the gap is finite but absurd. When the probabilities differ only in their last bits,
      R^2_p is defined but the clipped logit target is garbage: Ionosphere/Gaussian naive
      Bayes gives a gap of -179 on one stack and -108 on another, with an identical
      advantage of 1.5e11 either way. Both R^2 are bounded above by 1, so any |gap| > 1
      means the logit fit was determined by where we clipped.

    N.B. do NOT test saturation instead. A decision tree saturates every sampled point -
    every leaf value is exactly 0 or 1 - yet the target still *varies* across the
    neighbourhood, so R^2_p is perfectly well defined and the gap is a real 0.000.
    Excluding on saturation throws away most of group D, which is the evidence that logit
    space actively harms piecewise-constant black boxes.
    '''
    return [r for r in rows
            if not np.isfinite(r['gap']) or abs(r['gap']) > GAP_PLAUSIBLE]


def by_group(rows):
    out = {}
    for g in GROUP_ORDER:
        sub = [r for r in rows if r['group'] == g]
        if not sub:
            continue
        out[g] = dict(n=len(sub),
                      gap=np.nanmedian([r['gap'] for r in sub]),
                      adv=np.median([r['adv'] for r in sub]),
                      wins=sum(r['adv'] > 1 for r in sub),
                      n_degenerate=len(degenerate(sub)),
                      models=sorted({r['model'] for r in sub}))
    return out


def assess(rows):
    '''check each pre-registered statement'''
    g = by_group(rows)
    print('\n' + '='*78)
    print('PRE-REGISTERED PREDICTIONS (PREREGISTRATION.md)')
    print('='*78)

    def med(model, field='adv'):
        sub = [r[field] for r in rows if r['model'] == model]
        return np.nanmedian(sub) if sub else float('nan')

    # 1: LDA behaves like logistic regression
    lda_gap, lda_adv = med('LDA', 'gap'), med('LDA')
    ok1 = lda_gap > 0.35 and lda_adv >= 10
    print(f"\n1. LDA lands in group A (gap > 0.35 and benefit >= 10x)")
    print(f"   LDA: median gap = {lda_gap:+.3f}, median benefit = {lda_adv:.2f}x"
          f"   -> {'CONFIRMED' if ok1 else 'NOT CONFIRMED'}")

    # 2: decision tree and kNN show no benefit
    print(f"\n2. Decision tree and kNN land in group D (gap < 0.15, benefit < 1.5x)")
    ok2 = True
    for m in ['Decision Tree', 'k Nearest Neighbours']:
        gp, ad = med(m, 'gap'), med(m)
        good = gp < 0.15 and ad < 1.5
        ok2 &= good
        print(f"   {m:24s} median gap = {gp:+.3f}, median benefit = {ad:.2f}x"
              f"   -> {'as predicted' if good else 'NOT as predicted'}")

    # 3: quadratic models sit strictly between A and D
    a, b, d = g.get('A linear'), g.get('B quadratic'), g.get('D piecewise constant')
    ok3 = a and b and d and (d['adv'] < b['adv'] < a['adv'])
    print(f"\n3. Quadratic-log-odds models (QDA, Gaussian NB) sit strictly between A and D")
    if a and b and d:
        print(f"   median benefit  A={a['adv']:.2f}x   B={b['adv']:.2f}x   D={d['adv']:.2f}x"
              f"   -> {'CONFIRMED' if ok3 else 'NOT CONFIRMED'}")

    # 4: group ordering
    present = [x for x in ['A linear', 'B quadratic', 'C smooth', 'D piecewise constant']
               if x in g]
    advs = [g[x]['adv'] for x in present]
    ok4 = all(advs[i] >= advs[i+1] for i in range(len(advs)-1))
    print(f"\n4. Group ordering A > B > C > D by median benefit")
    print('   ' + '   '.join(f'{x[0]}={g[x]["adv"]:.2f}x' for x in present)
          + f"   -> {'CONFIRMED' if ok4 else 'NOT CONFIRMED'}")
    return dict(s1=ok1, s2=ok2, s3=ok3, s4=ok4)


def summary(rows):
    print('\n' + '='*78)
    print(f'SUMMARY  (n = {len(rows)} dataset x black box combinations)')
    print('='*78)
    g = by_group(rows)
    print(f"\n{'group':<32s} {'n':>4s} {'median gap':>11s} {'median benefit':>15s} {'better':>8s}")
    for name in GROUP_ORDER:
        if name not in g:
            continue
        v = g[name]
        print(f"{GROUP_LABEL[name]:<32s} {v['n']:>4d} {v['gap']:>+11.3f} "
              f"{v['adv']:>14.2f}x {v['wins']}/{v['n']:>3d}")

    deg = degenerate(rows)
    if deg:
        print(f"\n{len(deg)} degenerate configuration(s) - the black box is constant, or "
              f"near enough, over the whole neighbourhood, so the gap is not a meaningful "
              f"quantity. Excluded from the gap correlation:")
        for r in deg:
            print(f"   {r['dataset']}|{r['model']:22s} saturation = {r['sat']:6.1%}  "
                  f"gap = {r['gap']:+9.3f}  benefit = {r['adv']:.3g}x")

    ok = [r for r in rows if r not in deg]
    rg, pg = spearmanr([r['gap'] for r in ok], [r['adv'] for r in ok])
    rs, ps = spearmanr([r['sat'] for r in rows], [r['adv'] for r in rows])
    print(f"\nSpearman(gap, benefit)        rho = {rg:+.3f}  p = {pg:.2g}  (n = {len(ok)})")
    print(f"Spearman(saturation, benefit) rho = {rs:+.3f}  p = {ps:.2g}  (n = {len(rows)})")

    for label, idx in [('Brier score (local)', ('brier_std', 'brier_logit', 'brier_lr')),
                       ('KL divergence (local)', ('kl_std', 'kl_logit', 'kl_lr'))]:
        wins = [0, 0, 0]
        for r in rows:
            wins[int(np.argmin([r[i] for i in idx]))] += 1
        print(f"\nbest surrogate by {label}: standard={wins[0]}  logit={wins[1]}  log.reg={wins[2]}")
    return dict(rho_gap=rg, p_gap=pg, rho_sat=rs, p_sat=ps, n=len(rows))


def seed_spread(prefix='results'):
    '''spread across repeated trials, if the seed sweep has run'''
    files = sorted(glob.glob(paths.results(f'{prefix}_seed*.json')))
    if len(files) < 2:
        return None
    per_seed = {}
    for p in files:
        rows, _ = load(p)
        for r in rows:
            per_seed.setdefault((r['dataset'], r['model']), []).append(r['adv'])
    print('\n' + '='*78)
    print(f'REPEATED TRIALS  ({len(files)} seeds)')
    print('='*78)
    by_model = {}
    for (ds, m), vals in per_seed.items():
        if len(vals) == len(files):
            by_model.setdefault(m, []).append(vals)
    print(f"\n{'model':<24s} {'median benefit':>15s} {'across-seed spread':>22s}")
    for m, groups in sorted(by_model.items()):
        allv = np.array(groups)                      # datasets x seeds
        med = np.median(allv)
        rel = np.median(np.std(allv, axis=1)/np.maximum(np.mean(allv, axis=1), 1e-12))
        print(f"{m:<24s} {med:>14.2f}x {rel:>21.1%}")
    return by_model


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json')
    rows, errors = load(path)
    if errors:
        print(f'{len(errors)} configurations failed:')
        for k, v in list(errors.items())[:10]:
            print(f'   {k:55s} {v[:60]}')
    summary(rows)
    assess(rows)
    seed_spread()
