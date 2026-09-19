'''
What the log-odds linearity diagnostic is actually measuring: checks C1-C5 of the fifth
registration (PREREGISTRATION.md).

The paper's diagnostic (sweep.py::diagnostic) fits two unregularised locality-weighted
least-squares lines at each query point, one to f's probabilities and one to its log-odds
clipped at 1e-9, on 2,000 points from the *evaluation* sampler, and reports the weighted R²
of each. This recomputes that at the same points and adds, per query point:

  C1  R²_logit at clip eps in {1e-3, 1e-6, 1e-9, 1e-12}, and with the exact transform
      Logit-LIME trains on (logit_regression.logits: rescale into [1e-9, 1 - 1e-8]).
      Does the number measure the black box or the clipping constant?
  C2  the gain in weighted R² of the log-odds fit from adding squared terms - a lack-of-fit
      test within one space, where Δ compares two. Diagonal terms only: full quadratics
      are d²/2 parameters, ill-posed on 2,000 points at d = 279.
  C3  the locality-weighted relative dispersion of grad logit f over the neighbourhood,
      for black boxes with an analytic gradient (common/gradients.py). No fit at all, so
      this is what "needs only the black box" would honestly mean. Exactly 0 for linear
      log-odds.
  C4  the in-sample Brier score of each of the two OLS fits, scored in probability space
      on the points they were fitted to. Their ratio is what any number derived from these
      two fits can at best predict.
  C5  R²_logit and R²_p again on a sample with its own salt, 'diagnostic sample'. The paper
      drew the diagnostic with the evaluation salt, which CLAUDE.md says a new draw site
      must not reuse.

Also recorded: the weighted variance of f's probabilities over the neighbourhood, which is
the stated cause of every degenerate configuration and replaces the |Δ| > 1 exclusion.

usage:  python sweep_diagnostic_checks.py [out.json] [--datasets A,B] [--models A,B]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients

import argparse
import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.models.logit_regression import logits as logit_lime_transform
from clime.evaluation.key_points import get_local_points
from sweeps import sweep, full_grid

warnings.filterwarnings('ignore')

EPSILONS = [1e-3, 1e-6, 1e-9, 1e-12]
N_SAMPLES = 2000          # as sweep.diagnostic
OWN_SALT = 'diagnostic sample'


# R² is undefined when the target is constant to within rounding. sweep.weighted_r2 tests
# ss_tot > 0, which a constant target passes: its weighted mean carries rounding error, so
# ss_tot comes out near 1e-28 rather than 0, and R² is then a ratio of two rounding errors -
# -303 on Arrhythmia with a logistic black box, where every sampled probability clips to the
# same bound. This is the mechanism behind every |Δ| > 1 configuration the paper excludes.
# The tolerance is relative to the target's own magnitude, so a real but small variation
# (a logistic model's log-odds far from its boundary, exactly linear and tiny in p) is kept.
REL_TOL = sweep.REL_TOL        # one tolerance, defined in sweep.py next to guarded_r2


def wls(A, target, w):
    '''weighted least squares: (coefficients, fitted values, weighted R², or nan if undefined)'''
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A*sw[:, None], target*sw, rcond=None)
    fit = A @ coef
    mean = np.sum(w*target)/np.sum(w)
    ss_tot = np.sum(w*(target - mean)**2)
    scale = REL_TOL*max(1.0, abs(mean))
    if not ss_tot/np.sum(w) > scale**2:
        return coef, fit, float('nan')
    return coef, fit, float(1 - np.sum(w*(target - fit)**2)/ss_tot)


def clipped_logit(p, eps):
    pc = np.clip(p, eps, 1 - eps)
    return np.log(pc/(1 - pc))


def wmean(x, w):
    return float(np.sum(w*x)/np.sum(w))


def point_stats(clf, model, test_data, q):
    out = {}
    X = get_local_points(test_data, q, samples=N_SAMPLES)['X']
    p = np.asarray(clf.predict_proba(X))[:, 1].astype(np.float64)
    w = costs.weights_based_on_distance(q, X)
    A = np.c_[X, np.ones(len(X))]

    out['saturation'] = float(((p <= 1e-6) | (p >= 1 - 1e-6)).mean())
    out['var_p'] = float(np.sum(w*(p - wmean(p, w))**2)/np.sum(w))

    _, fit_p, out['r2_prob'] = wls(A, p, w)
    for eps in EPSILONS:
        out[f'r2_logit_eps{eps:.0e}'] = wls(A, clipped_logit(p, eps), w)[2]
    out['r2_logit'] = out['r2_logit_eps1e-09']          # the paper's definition, guarded
    # exactly as sweep.diagnostic computes them, unguarded, to show what the guard changes
    out['r2_logit_paper'] = float(sweep.weighted_r2(A, clipped_logit(p, 1e-9), w))
    out['r2_prob_paper'] = float(sweep.weighted_r2(A, p, w))
    out['r2_logit_rescaled'] = wls(A, logit_lime_transform(p), w)[2]

    # C2: squared terms, same space
    Aq = np.c_[X, X**2, np.ones(len(X))]
    out['r2_logit_quad'] = wls(Aq, clipped_logit(p, 1e-9), w)[2]
    out['curvature_gain'] = out['r2_logit_quad'] - out['r2_logit']

    # C4: each OLS fit scored in probability space on its own training points
    _, fit_l, _ = wls(A, clipped_logit(p, 1e-9), w)
    g_prob = np.clip(fit_p, 0, 1)
    g_logit = 1/(1 + np.exp(-fit_l))
    out['insample_brier_prob'] = wmean((g_prob - p)**2, w)
    out['insample_brier_logit'] = wmean((g_logit - p)**2, w)

    # C3: gradient dispersion, no fit
    if model in gradients.ANALYTIC_GRADIENTS:
        G = np.asarray(gradients.grad_logit(clf, model, X), dtype=np.float64)
        ok = np.all(np.isfinite(G), axis=1)
        if ok.sum() > 1:
            Gw, ww = G[ok], w[ok]
            gbar = np.sum(ww[:, None]*Gw, axis=0)/np.sum(ww)
            spread = np.sum(ww*np.sum((Gw - gbar)**2, axis=1))/np.sum(ww)
            denom = float(gbar @ gbar)
            out['grad_dispersion'] = float(spread/denom) if denom > 0 else np.nan
            out['grad_finite_frac'] = float(ok.mean())

    # C5: its own random stream
    X2 = get_local_points(test_data, q, samples=N_SAMPLES, salt=OWN_SALT)['X']
    p2 = np.asarray(clf.predict_proba(X2))[:, 1].astype(np.float64)
    w2 = costs.weights_based_on_distance(q, X2)
    A2 = np.c_[X2, np.ones(len(X2))]
    out['own_r2_prob'] = wls(A2, p2, w2)[2]
    out['own_r2_logit'] = wls(A2, clipped_logit(p2, 1e-9), w2)[2]
    return out


def run_config(dataset, model):
    r = clime.pipeline.run_pipeline(sweep.opts(dataset, model, sweep.EXPLAINERS[0],
                                               sweep.METRICS[0]), parallel_eval=False)
    clf, test_data = r['clf'], r['test_data']
    qs = sweep.query_points(test_data)
    points = [point_stats(clf, model, test_data, np.asarray(q, dtype=np.float64))
              for q in qs]
    entry = {'group': full_grid.GROUP_OF.get(model, 'unassigned'),
             'n_features': int(np.asarray(test_data['X']).shape[1]),
             'n_test': int(np.asarray(test_data['X']).shape[0]),
             'points': points}
    keys = sorted({k for pt in points for k in pt})
    for k in keys:
        vals = np.array([pt.get(k, np.nan) for pt in points], dtype=float)
        entry[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        if k.startswith('r2_') or k.startswith('own_r2'):
            entry[f'n_defined_{k}'] = int(np.isfinite(vals).sum())
    return entry


def run(out_path, datasets, models):
    out_path = paths.results(out_path)
    out = {'_meta': {'epsilons': EPSILONS, 'n_samples': N_SAMPLES, 'own_salt': OWN_SALT,
                     'query points': sweep.QUERY_POINTS}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} already done', flush=True)
    for dataset in datasets:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                e = run_config(dataset, model)
                print(f"{dataset:26s} {model:36s} R2logit={e['r2_logit']:+.3f} "
                      f"eps1e-3={e['r2_logit_eps1e-03']:+.3f} curv={e['curvature_gain']:+.3f} "
                      f"disp={e.get('grad_dispersion', float('nan')):.3g}", flush=True)
            except Exception as exc:
                e = {'error': f'{type(exc).__name__}: {exc}'}
                print(f'{key:62s} FAILED {e["error"][:60]}', flush=True)
            out[key] = e
            json.dump(out, open(out_path, 'w'))
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_diagnostic_checks.json')
    p.add_argument('--datasets', type=str, default=None)
    p.add_argument('--models', type=str, default=None)
    a = p.parse_args()
    full_grid.configure()
    run(a.out, a.datasets.split(',') if a.datasets else full_grid.DATASETS,
        a.models.split(',') if a.models else full_grid.CHECK_MODELS)
