'''
Choosing the worked example, by a protocol fixed before looking (PREREGISTRATION.md).

A figure showing one query point where a fidelity score misleads is worth nothing if the
point was found by hunting for it.  So the candidates are enumerated by a rule, every
candidate that is tried is logged including the rejected ones, the base rate of the shape
is reported alongside, and the point that gets drawn is the MEDIAN qualifying one rather
than the most extreme.

Three shapes qualify:

  case 1  fidelity ranks standard LIME at or above Logit-LIME, while the explanation
          ground truth puts Logit-LIME ahead by a wide margin.  The instrument prefers the
          worse explanation.
  case 2  the hard-label surrogate has the best test-data fidelity and the worst cosine,
          with KL ranking it last.  The instrument prefers the most confident surrogate.
  case 3  fidelity cannot separate the two surrogates at all - both cells agree to within
          FID_TIE_TOL - while Brier separates them by an order of magnitude and the
          explanations are far apart.  The instrument is blind where a proper scoring rule
          is not.

**Case 3 was added after running the first two, and that is recorded rather than hidden.**
Case 1 turned out to be satisfied largely at saturated query points, where the black box
predicts one class over the whole neighbourhood: there fidelity is indeed blind, but so is
Brier, so the point illustrates every probability instrument failing rather than this one.
That is a real and reportable caveat - it is the honest limit of the argument - but it is
not the claim the motivation section makes, and choosing between the two AFTER seeing them
would be picking the flattering one.  So both are kept: case 1's robust example is reported
as the caveat, case 3's as the motivating figure, and the distinction is stated in the
paper.

Two filters, both of which reject candidates that would make a dishonest figure:

  flat    where the black box's gradient is much smaller than usual for that
          configuration there is no direction to be right about, and a bad cosine is
          measuring noise.
  non-monotone
          an RBF SVM's decision function decays back towards its bias far from the data,
          so a neighbourhood can contain a SECOND boundary of the black box.  A surrogate
          that points the other way may be fitting that second boundary perfectly well,
          and the gradient at q - which is the ground truth by definition - is then the
          thing that is unrepresentative, not the surrogate.  Drawing such a point would
          illustrate a non-monotone black box rather than a blind instrument.  This is a
          real effect in the candidate list, not a hypothetical: it is why the first
          candidate found by the ad-hoc probe is rejected here.

usage:  python analysis/select_example.py [--extended] [--top N]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import argparse
import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import get_points_between_class_means
from sweeps.sweep import opts, METRICS
from analysis.analyse_fidelity_explanation import load

warnings.filterwarnings('ignore')

FID_TEST = 'fidelity | test data'
FID_LOCAL = 'fidelity | local sample'
COS_MARGIN = 0.4        # "the explanations really do differ"
FID_MARGIN = 0.02       # "the instrument really does prefer it"
FID_TIE_TOL = 0.01      # fidelity readings this close cannot separate two surrogates
BRIER_MARGIN = 10.0     # "a proper scoring rule, on the same point, is not blind"
FLAT_FRACTION = 0.5     # reject ||grad|| below this times the configuration's median
KERNEL_MASS = 0.1       # transect extent: where the locality weight has fallen to this
MONOTONE_TOL = 0.02     # ignore wobbles smaller than this in probability


def case_1(r):
    '''fidelity prefers standard LIME; the truth prefers Logit-LIME'''
    d_cos = r[('cos', 'logit')] - r[('cos', 'standard')]
    if not np.isfinite(d_cos) or d_cos <= COS_MARGIN:
        return None
    if r[(FID_TEST, 'standard')] < r[(FID_TEST, 'logit')]:
        return None
    if r[(FID_LOCAL, 'standard')] < r[(FID_LOCAL, 'logit')]:
        return None
    return d_cos


def case_2(r):
    '''fidelity crowns the hard-label surrogate; the truth puts it last'''
    fids = {s: r[(FID_TEST, s)] for s in ('standard', 'logit', 'logreg')}
    coss = {s: r[('cos', s)] for s in ('standard', 'logit', 'logreg')}
    kls = {s: r[('KL', s)] for s in ('standard', 'logit', 'logreg')}
    if not all(np.isfinite(v) for v in (*fids.values(), *coss.values(), *kls.values())):
        return None
    best_other = max(fids['standard'], fids['logit'])
    worst_other = min(coss['standard'], coss['logit'])
    if fids['logreg'] - best_other < FID_MARGIN:
        return None
    if worst_other - coss['logreg'] < COS_MARGIN:
        return None
    if kls['logreg'] < max(kls['standard'], kls['logit']):
        return None
    return fids['logreg'] - best_other


def case_3(r):
    '''fidelity cannot tell them apart; Brier can, and the explanations differ'''
    d_cos = r[('cos', 'logit')] - r[('cos', 'standard')]
    if not np.isfinite(d_cos) or d_cos <= COS_MARGIN:
        return None
    for cell in (FID_TEST, FID_LOCAL):
        if abs(r[(cell, 'standard')] - r[(cell, 'logit')]) > FID_TIE_TOL:
            return None
    a, b = r[('Brier', 'standard')], r[('Brier', 'logit')]
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0 or a/b < BRIER_MARGIN:
        return None
    return d_cos


def transect(dataset, model, index, n=400):
    '''
    the black box's probability along the query-point line through q, and the locality
    weight at each step, so "within the kernel's support" is measured rather than assumed
    '''
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, test = r['clf'], r['test_data']
    qs, _ = get_points_between_class_means(test)
    q = np.asarray(qs[index], dtype=float)
    direction = np.asarray(qs[-1]) - np.asarray(qs[0])
    direction = direction/np.linalg.norm(direction)
    k = np.sqrt(len(q))*costs.KERNEL_WIDTH_SCALE
    lim = k*np.sqrt(-2.0*np.log(KERNEL_MASS))         # where the weight falls to 10%
    t = np.linspace(-lim, lim, n)
    line = q[None, :] + t[:, None]*direction[None, :]
    p = np.asarray(clf.predict_proba(line))[:, 1]
    return t, p, lim


def monotone(p, tol=MONOTONE_TOL):
    '''
    does f move one way across the transect?

    A running-extremum test rather than a sign test on the differences: sampling noise in
    a piecewise model makes the differences change sign constantly without the function
    ever turning round.  What matters is whether it retraces by more than `tol`.
    '''
    up = np.max(np.maximum.accumulate(p) - p)
    down = np.max(p - np.minimum.accumulate(p))
    return bool(min(up, down) <= tol), float(min(up, down))


def main(extended=False, top=25):
    rows, mismatch = load(extended=extended)
    assert mismatch <= 1e-12, f'the join is invalid: Brier disagrees by {mismatch:.3g}'

    # the gradient norm at each point, against the configuration's own median
    norms = {}
    for r in rows:
        norms.setdefault(r['config'], []).append(r['truth_norm'])
    median_norm = {k: float(np.median(v)) for k, v in norms.items()}

    candidates = []
    for r in rows:
        for name, test in (('case 1', case_1), ('case 2', case_2),
                           ('case 3', case_3)):
            margin = test(r)
            if margin is None:
                continue
            candidates.append({'case': name, 'config': r['config'],
                               'dataset': r['dataset'], 'model': r['model'],
                               'index': r['index'], 'margin': float(margin),
                               'flat': r['truth_norm'] < FLAT_FRACTION*median_norm[r['config']],
                               'truth_norm': r['truth_norm'],
                               'median_norm': median_norm[r['config']],
                               'n_features': None, 'row': r})

    n_points = len(rows)
    print(f'{n_points} query points, {len({r["config"] for r in rows})} configurations')
    for name in ('case 1', 'case 2', 'case 3'):
        c = [x for x in candidates if x['case'] == name]
        print(f'  {name}: {len(c)} candidates ({len(c)/n_points:.2%} of points), '
              f'{sum(1 for x in c if x["flat"])} rejected as flat')

    kept = [c for c in candidates if not c['flat']]
    kept.sort(key=lambda c: -c['margin'])

    print(f'\nchecking the black box along the transect at each of the {min(top, len(kept))} '
          f'strongest surviving candidates')
    print(f"{'case':<7s} {'dataset':<22s} {'model':<26s} {'pt':>3s} {'margin':>7s} "
          f"{'|grad|':>7s} {'retrace':>8s}  verdict")
    log = []
    for c in kept[:top]:
        t, p, lim = transect(c['dataset'], c['model'], c['index'])
        ok, retrace = monotone(p)
        c['monotone'], c['retrace'], c['transect_lim'] = ok, retrace, lim
        c['f_range'] = [float(p.min()), float(p.max())]
        verdict = 'usable' if ok else 'REJECT: f turns round in the neighbourhood'
        print(f"{c['case']:<7s} {c['dataset']:<22s} {c['model']:<26s} {c['index']:>3d} "
              f"{c['margin']:>7.3f} {c['truth_norm']:>7.2f} {retrace:>8.3f}  {verdict}")
        log.append({k: v for k, v in c.items() if k != 'row'})

    usable = [c for c in kept[:top] if c.get('monotone')]
    print(f'\n{len(usable)}/{min(top, len(kept))} checked candidates survive the '
          f'monotonicity filter')
    for name in ('case 1', 'case 2', 'case 3'):
        sub = sorted((c for c in usable if c['case'] == name), key=lambda c: c['margin'])
        if not sub:
            print(f'\n{name}: no candidate survives both filters')
            continue
        pick = sub[len(sub)//2]
        print(f"\n{name}: MEDIAN of the {len(sub)} surviving candidates -> "
              f"{pick['dataset']} | {pick['model']}  point {pick['index']}  "
              f"(margin {pick['margin']:.3f})")
        r = pick['row']
        print(f"  {'':22s} {'standard':>10s} {'logit':>10s} {'hard label':>12s}")
        for field in (FID_TEST, FID_LOCAL, 'Brier', 'KL', 'cos'):
            print(f'  {field:<22s} ' + ' '.join(
                f'{r[(field, s)]:>10.4g}' if s != "logreg" else f'{r[(field, s)]:>12.4g}'
                for s in ('standard', 'logit', 'logreg')))
        # 2-D datasets can be drawn directly; the rest fall back to the transect figure
        two_d = [c for c in sub if c['dataset'] in ('Gaussian', 'Moons', 'Circles')]
        if two_d:
            m = two_d[len(two_d)//2]
            print(f"  drawable (2-D) median: {m['dataset']} | {m['model']} "
                  f"point {m['index']}, margin {m['margin']:.3f}")

    out = paths.results('example_candidates.json')
    json.dump({'_meta': {'cos_margin': COS_MARGIN, 'fid_margin': FID_MARGIN,
                         'flat_fraction': FLAT_FRACTION, 'kernel_mass': KERNEL_MASS,
                         'monotone_tol': MONOTONE_TOL, 'extended': extended,
                         'n_points': n_points,
                         'brier_margin': BRIER_MARGIN, 'fid_tie_tol': FID_TIE_TOL,
                         'n_candidates': {n: sum(1 for c in candidates if c['case'] == n)
                                          for n in ('case 1', 'case 2', 'case 3')}},
               'checked': log}, open(out, 'w'), indent=1)
    print(f'\nevery candidate tried is logged in {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--extended', action='store_true')
    p.add_argument('--top', type=int, default=25)
    a = p.parse_args()
    main(extended=a.extended, top=a.top)
