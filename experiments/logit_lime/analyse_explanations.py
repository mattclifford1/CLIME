'''
Do the two surrogates give the same explanation?

Restricted to datasets with enough features for the question to mean anything: on 2
features a rank correlation can only be +/-1 and a top-3 overlap is always 1.0.

usage:  python analyse_explanations.py [results_explanations.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import json
import numpy as np
from analyse import GROUP_ORDER, GROUP_LABEL

MIN_FEATURES = 6


def load(path):
    d = json.load(open(path))
    return [dict(dataset=k.split('|')[0], model=k.split('|')[1], **v)
            for k, v in d.items() if not k.startswith('_') and 'error' not in v]


def report(rows, label):
    if not rows:
        return
    def m(f):
        v = [r[f] for r in rows if np.isfinite(r.get(f, np.nan))]
        return np.mean(v) if v else float('nan')
    print(f"{label:<34s} {len(rows):>4d} {m('rho'):>8.3f} {m('overlap3'):>9.2f} "
          f"{m('overlap5'):>9.2f} {m('top1_agree'):>8.2f} {m('sign_agree'):>8.2f}")


if __name__ == '__main__':
    all_rows = load(sys.argv[1] if len(sys.argv) > 1 else 'results_explanations.json')
    rows = [r for r in all_rows if r.get('n_features', 0) >= MIN_FEATURES]
    print(f'{len(all_rows)} configurations, {len(rows)} on datasets with '
          f'>= {MIN_FEATURES} features\n')
    print(f"{'':<34s} {'n':>4s} {'rank rho':>8s} {'top3':>9s} {'top5':>9s} "
          f"{'top1':>8s} {'sign':>8s}")
    report(rows, 'ALL')
    print()
    for g in GROUP_ORDER:
        report([r for r in rows if r['group'] == g], GROUP_LABEL[g])
    print()
    for ds in sorted({r['dataset'] for r in rows}):
        sub = [r for r in rows if r['dataset'] == ds]
        report(sub, f"  {ds} ({sub[0]['n_features']}f)")

    print('\n--- how often is the top feature different? ---')
    t1 = np.array([r['top1_agree'] for r in rows])
    print(f'  mean agreement on the single most important feature: {t1.mean():.1%}')
    print(f'  configurations where they disagree at >1 in 4 query points: '
          f'{int(np.sum(t1 < 0.75))}/{len(t1)}')
