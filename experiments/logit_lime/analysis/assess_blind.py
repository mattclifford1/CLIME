'''
The third pre-registration, assessed (PREREGISTRATION.md).

Predictions 5-7 were registered over the 70-configuration extension - the five extended
grid black boxes that have a gradient ground truth but had never been through the fidelity
sweep - specifically because the 84 overlapping configurations had already been looked at
while the claim was being formed.  So the two sets are assessed separately here and never
pooled: pooling them would let the seen half carry the blind half.

Predictions 8-10 concern the null explainer, and are assessed over the whole registered
grid of 168, which had not been run at all.

usage:  python analysis/assess_blind.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from analysis.analyse_fidelity_explanation import (
    load, pooled_level, direction, crowns, explanation_quality, INSTRUMENTS)

EXTENSION_MODELS = ['Bagged Logistic', 'Bayes Optimal', 'Nearest Class Mean',
                    'Polynomial Logistic (deg 2)', 'RBF Logistic (Nystroem)']
FID_TEST = 'fidelity | test data'


def verdict(ok):
    return 'HELD' if ok else 'FAILED'


def assess_fidelity_predictions(rows, label):
    n_cfg = len({r['config'] for r in rows})
    print(f'\n--- {label}: {n_cfg} configurations, {len(rows)} query points ---')

    # 5. ties
    _, _, ties_f, total_f = direction(rows, FID_TEST)
    _, _, ties_kl, total_kl = direction(rows, 'KL')
    tie_rate, kl_tie_rate = ties_f/max(total_f, 1), ties_kl/max(total_kl, 1)
    print(f'  P5  fidelity ties at {tie_rate:.1%} of comparisons (registered >= 30%), '
          f'KL at {kl_tie_rate:.1%} (registered < 10%)   '
          f'{verdict(tie_rate >= 0.30 and kl_tie_rate < 0.10)}')

    # 6. the hard-label surrogate is fidelity's favourite, and explains worst
    counts, n = crowns(rows, FID_TEST)
    share = counts['logreg']/max(n, 1)
    q = explanation_quality(rows)
    worst_cos = min(q, key=lambda s: q[s][0]) == 'logreg'
    print(f'  P6  hard-label crowned by test-data fidelity in {counts["logreg"]}/{n} '
          f'({share:.1%}, registered >= 40%); its mean cosine {q["logreg"][0]:.3f} vs '
          f'standard {q["standard"][0]:.3f}, logit {q["logit"][0]:.3f} '
          f'-> lowest: {worst_cos}   {verdict(share >= 0.40 and worst_cos)}')

    # 7. fidelity is not uninformative
    rho_f, _, _ = pooled_level(rows, FID_TEST)
    rho_kl, _, _ = pooled_level(rows, 'KL')
    gap = abs(abs(rho_f) - abs(rho_kl))
    print(f'  P7  level rho: fidelity {rho_f:+.3f}, KL {rho_kl:+.3f}; '
          f'difference in magnitude {gap:.3f} (registered <= 0.10)   {verdict(gap <= 0.10)}')
    if gap > 0.10 and abs(rho_f) > abs(rho_kl):
        print('      note: the prediction fails because fidelity tracks cosine MORE '
              'closely than KL does, not less.')

    counts_all = {i: crowns(rows, i)[0] for i in INSTRUMENTS}
    print(f"      crowns   {'':<12s}" + ' '.join(f'{s:>12s}' for s in
                                                 ('standard', 'logit', 'logreg')))
    for i, c in counts_all.items():
        print(f'      {i:<22s}' + ' '.join(f'{c[s]:>12d}' for s in
                                           ('standard', 'logit', 'logreg')))


def assess_null():
    path = paths.results('results_null.json')
    if not os.path.exists(path):
        print('\n(no results_null.json yet)')
        return
    raw = json.load(open(path))
    cells = raw['_meta']['cells']
    have_kl = any(c.startswith('KL') for c in cells)
    rows = [v for k, v in raw.items() if not k.startswith('_') and 'error' not in v]
    print(f'\n--- the null explainer: {len(rows)} configurations of the registered grid ---')

    local = np.array([v for r in rows for v in r['cells']['fidelity | local sample']])
    test = np.array([v for r in rows for v in r['cells']['fidelity | test data']])
    print(f'  P8  mean fidelity: local sample {local.mean():.4f}, test data '
          f'{test.mean():.4f} (registered >= 0.90 on the local sample)   '
          f'{verdict(local.mean() >= 0.90)}')
    if local.mean() < 0.90:
        print(f'      it scores >= 0.90 in {np.mean([np.mean(r["cells"]["fidelity | local sample"]) >= 0.9 for r in rows]):.1%} '
              f'of configurations and a perfect 1.000 at '
              f'{np.mean(local == 1.0):.1%} of all query points')

    # P9/P10 need the fitted surrogates alongside, which the join supplies
    joined, _ = load(extended=False)
    have = [r for r in joined if ('fidelity | local sample', 'null') in r]
    if have:
        beats = np.mean([r[('fidelity | local sample', 'null')]
                         >= r[('fidelity | local sample', 'standard')] for r in have])
        print(f'  P9  null >= standard LIME on local-sample fidelity at {beats:.1%} of '
              f'{len(have)} points (registered >= 30%)   {verdict(beats >= 0.30)}')
        kl_ok = [r for r in have if np.isfinite(r[('KL', 'null')])]
        if have_kl and kl_ok:
            worst = np.mean([
                r[('KL', 'null')] > max(r[('KL', s)]
                                        for s in ('standard', 'logit', 'logreg'))
                for r in kl_ok])
            print(f'  P10 KL ranks the null explainer worst of the four at {worst:.1%} '
                  f'of {len(kl_ok)} points (registered > 95%)   {verdict(worst > 0.95)}')
        else:
            print('  P10 not assessable: results_null.json carries no KL cell. '
                  'Re-run sweeps/sweep_null.py.')

    # the number the section quotes: how good is "good" for an explainer that says nothing
    perfect = [k for k, v in raw.items()
               if not k.startswith('_') and 'error' not in v
               and np.mean(v['cells']['fidelity | test data']) == 1.0]
    print(f'      it scores a mean test-data fidelity of exactly 1.000 in '
          f'{len(perfect)} configurations'
          + (f': {", ".join(perfect[:3])}' if perfect else ''))
    by_cfg = sorted(((np.mean(v['cells']['fidelity | local sample']), k)
                     for k, v in raw.items()
                     if not k.startswith('_') and 'error' not in v), reverse=True)
    print('      highest and lowest by mean local fidelity: '
          + ', '.join(f'{k} {s:.3f}' for s, k in by_cfg[:2] + by_cfg[-2:]))


def main():
    rows, mismatch = load(extended=True)
    assert mismatch <= 1e-12, f'the join is invalid: Brier disagrees by {mismatch:.3g}'
    seen = [r for r in rows if r['model'] not in EXTENSION_MODELS]
    blind = [r for r in rows if r['model'] in EXTENSION_MODELS]

    print('=' * 78)
    print('THIRD PRE-REGISTRATION, ASSESSED')
    print('=' * 78)
    assess_fidelity_predictions(blind, 'THE BLIND EXTENSION (never looked at)')
    assess_fidelity_predictions(seen, 'the 84 already seen (recomputation, not a test)')
    assess_null()


if __name__ == '__main__':
    main()
