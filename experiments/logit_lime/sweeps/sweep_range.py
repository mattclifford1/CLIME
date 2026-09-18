'''
How far a surrogate's reported coefficient can be carried, over the registered grid.

Every other sweep here scores a surrogate against the black box.  This one measures a
property of the number the surrogate hands to a user, and needs no notion of fit quality
at all.  A standard LIME surrogate is a linear model of a probability, so it is a
probability only inside the slab

    0 <= g(q) + beta.(x - q) <= 1,

whose half-width along beta is (1 - g(q))/||beta|| on one side and g(q)/||beta|| on the
other.  Carry the coefficient further than that - "the probability rises by beta_j for each
unit of x_j" - and the claim is arithmetic nonsense.  Whether that matters is an empirical
question about how the slab compares with the neighbourhood the surrogate was fitted on,
and this sweep answers it by recording, per query point:

  reach       the distance from q, toward the confident side, at which the standard
              surrogate's unclipped output leaves [0,1], and that distance as a fraction
              of the locality kernel width k = 0.75 sqrt(d) that defined "local"
  mass        the kernel-weighted fraction of the surrogate's OWN 10,000 point training
              sample on which its unclipped output is not a probability, split by side
  base        g(q) against f(q), for both surrogates: the value the reading starts from
  norm        ||beta|| for both surrogates, so that a per-configuration span over the 20
              query points says whether the reported size tracks importance or saturation
  flip        for black boxes with an analytic gradient, the distance along the black
              box's own top feature at which each model changes its decision, against the
              truth - the counterfactual a coefficient implies

The flip is only well posed where f is monotone along that feature within the range
searched, which is automatic for the exactly-linear families and is NOT automatic for an
RBF SVM, whose decision function can turn back inside the neighbourhood.  `n_crossings` is
recorded so the analysis can require exactly one rather than assume it.

Nothing here depends on the evaluation metric, so no metric is run.

usage:  python sweeps/sweep_range.py [results_range.json] [--datasets N] [--models M,...]
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
from sweeps.sweep import opts, DATASETS, MODELS, GROUP_OF, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

SAMPLES = 10000          # the surrogate's own training sample size, as bLIMEy uses
SEARCH = 8.0             # the flip is looked for within this many sd of q
SEARCH_N = 3201
SAT_TOL = 1e-9           # logit_ridge's squash bound; matches gradients.saturated


def surrogate_logodds(expl, X):
    '''logit g for the logit surrogate, read off the ridge before the sigmoid'''
    m = expl.surrogate_model
    return float(m.intercept_) + np.atleast_2d(X) @ np.atleast_2d(m.coef_)[-1, :]


def scan_flip(clf, q, j):
    '''
    signed distance along feature j to the black box's own boundary, nearest q, plus how
    many times it crosses in range.  Scanned rather than solved: f need not be monotone
    along an axis, and a solver would silently return one root of several.
    '''
    t = np.linspace(-SEARCH, SEARCH, SEARCH_N)
    X = np.repeat(np.asarray(q, dtype=float)[None, :], t.size, axis=0)
    X[:, j] += t
    s = np.sign(np.asarray(clf.predict_proba(X))[:, 1] - 0.5)
    cross = np.where(np.diff(s) != 0)[0]
    nearest = float(t[cross[np.argmin(np.abs(t[cross]))]]) if cross.size else float('nan')
    return nearest, int(cross.size)


def run_config(dataset, model):
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train, test = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test)
    qs = np.asarray(qs, dtype=float)
    d = qs.shape[1]
    k = costs.KERNEL_WIDTH_SCALE*np.sqrt(d)
    cov = np.cov(test['X'].T)
    differentiable = gradients.has_gradient(model)

    entry = {'group': GROUP_OF[model], 'n_features': int(d), 'kernel_width': float(k),
             'differentiable': bool(differentiable), 'points': []}
    E = clime.explainer.AVAILABLE_EXPLAINERS

    for q in qs:
        e_std = E['bLIMEy (normal)'](clf, query_point=q, train_data=train, test_data=test)
        e_log = E['bLIMEy (logit)'](clf, query_point=q, train_data=train, test_data=test)
        b_std = np.asarray(e_std.get_explanation(), dtype=float)
        b_log = np.asarray(e_log.get_explanation(), dtype=float)

        f_q = float(np.asarray(clf.predict_proba(q[None, :]))[0, 1])
        g_q = float(e_std.surrogate_model.predict(q[None, :])[0, 1])       # unclipped
        l_q = float(surrogate_logodds(e_log, q[None, :])[0])
        n_std = float(np.linalg.norm(b_std))
        n_log = float(np.linalg.norm(b_log))

        # the surrogate's own training neighbourhood, same draw site (same salt) as
        # bLIMEy._sample_locally, so this is mass it was actually fitted on
        rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
        X = rng.multivariate_normal(q, cov, SAMPLES)
        w = costs.weights_based_on_distance(q, X)
        g_n = e_std.surrogate_model.predict(X)[:, 1]
        p_n = np.asarray(clf.predict_proba(X))[:, 1]
        sw = float(np.sum(w))

        # toward the confident side: the direction a user reading the explanation would
        # push the feature to strengthen the prediction the black box already makes
        reach = ((1 - g_q)/n_std if g_q >= 0.5 else g_q/n_std) if n_std > 0 else float('inf')

        row = {
            'f_q': f_q, 'g_std': g_q, 'g_log': float(1/(1 + np.exp(-np.clip(l_q, -700, 700)))),
            'norm_std': n_std, 'norm_log': n_log,
            'reach': float(reach), 'reach_over_k': float(reach/k),
            'mass_above_1': float(np.sum(w*(g_n > 1))/sw),
            'mass_below_0': float(np.sum(w*(g_n < 0))/sw),
            'sat_point': bool((f_q < SAT_TOL) or (f_q > 1 - SAT_TOL)),
            'sat_mass': float(np.sum(w*((p_n < SAT_TOL) | (p_n > 1 - SAT_TOL)))/sw),
        }

        if differentiable:
            truth = gradients.grad_logit(clf, model, q)[0]
            if np.all(np.isfinite(truth)) and np.linalg.norm(truth) > 0:
                j = int(np.argmax(np.abs(truth)))
                nearest, n_cross = scan_flip(clf, q, j)
                row.update({
                    'feature': j, 'truth_norm': float(np.linalg.norm(truth)),
                    'truth_j': float(truth[j]),
                    'flip_std': float((0.5 - g_q)/b_std[j]) if b_std[j] != 0 else float('nan'),
                    'flip_log': float(-l_q/b_log[j]) if b_log[j] != 0 else float('nan'),
                    'flip_scan': nearest, 'n_crossings': n_cross,
                    # exact only where the log-odds are linear in x; the analysis uses it
                    # for group A alone and checks it against the scan
                    'flip_analytic': float(-np.log(max(f_q, 1e-300)/max(1 - f_q, 1e-300))
                                           / truth[j]) if truth[j] != 0 else float('nan'),
                })
        entry['points'].append(row)
    return entry


def run(out_path, datasets, models, scales=None):
    '''
    `scales` reruns everything at other locality kernel widths, keyed `scale|dataset|model`.

    Reach and mass both depend on the kernel width by construction - a narrower kernel
    makes the fitted chord more like a tangent, which pushes the exit further out in units
    of k - so the registered numbers are only meaningful next to that dependence.  The
    default (None) is a single run at the module default, keyed `dataset|model`.
    '''
    out_path = paths.results(out_path)
    default_scale = costs.KERNEL_WIDTH_SCALE
    out = {'_meta': {'seed': clime.RANDOM_SEED, 'samples': SAMPLES,
                     'kernel_width_scale': default_scale, 'scales': scales,
                     'search': SEARCH, 'sat_tol': SAT_TOL}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} configurations already done', flush=True)

    try:
        for scale in (scales or [None]):
            if scale is not None:
                # set on the module so the surrogate's own training weights use it, and
                # drop the pipeline cache, which keys on opts and so cannot see this
                costs.KERNEL_WIDTH_SCALE = scale
                clime.pipeline.run_pipeline.cache_clear()
            tag = '' if scale is None else f'{scale}|'
            for dataset in datasets:
                for model in models:
                    key = f'{tag}{dataset}|{model}'
                    if key in out:
                        continue
                    try:
                        entry = run_config(dataset, model)
                        pts = entry['points']
                        mass = np.mean([p['mass_above_1'] + p['mass_below_0'] for p in pts])
                        inside = np.mean([p['reach_over_k'] < 1 for p in pts])
                        print(f'{tag:6s}{dataset:26s} {model:36s} mass={mass:.3f} '
                              f'reach<k at {inside:.0%}', flush=True)
                    except Exception as e:
                        entry = {'error': f'{type(e).__name__}: {e}'}
                        print(f'{key:64s} FAILED {entry["error"][:60]}', flush=True)
                    out[key] = entry
                    json.dump(out, open(out_path, 'w'))
    finally:
        # leave the module as it was found: this constant is global and other scripts in
        # the same process would silently inherit it
        costs.KERNEL_WIDTH_SCALE = default_scale
        clime.pipeline.run_pipeline.cache_clear()

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_range.json')
    p.add_argument('--datasets', type=int, default=len(DATASETS))
    p.add_argument('--models', default=None,
                   help='comma separated subset of the registered black boxes')
    p.add_argument('--dataset-names', default=None,
                   help='comma separated subset of the registered datasets')
    p.add_argument('--scales', default=None,
                   help='comma separated locality kernel width scales; keys gain a prefix')
    a = p.parse_args()
    chosen = MODELS if a.models is None else [m.strip() for m in a.models.split(',')]
    sets = (DATASETS[:a.datasets] if a.dataset_names is None
            else [d.strip() for d in a.dataset_names.split(',')])
    widths = None if a.scales is None else [float(s) for s in a.scales.split(',')]
    run(a.out, sets, chosen, widths)
