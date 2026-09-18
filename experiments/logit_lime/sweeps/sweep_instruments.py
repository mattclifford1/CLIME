'''
What can each instrument see?  A probe with no fit in the loop.

Every other sweep here compares fitted surrogates, which makes it hard to say what an
instrument is responding to: a fit moves its direction, its scale and its intercept at
once, and the three are confounded.  This sweep removes the fit.  At a query point q it
builds the one-parameter-at-a-time family

    logit g_{theta,s}(x) = logit f(q) + s * R(theta) grad logit f(q) . (x - q)

(common/surrogates.py) and scores every member with every instrument.  By construction
the cosine of g's explanation to the ground truth is cos(theta) and the ratio of its
coefficient norm to the truth's is s, so the two axes ARE the two ways an explanation can
be wrong - direction and magnitude - and the response of an instrument along each axis is
exactly what that instrument can see.

There are three axes, because there are three ways to be wrong and the instruments do not
treat them alike (common/surrogates.py::family_member):

  theta    the direction of the explanation, at the true scale.
  scale    the slope alone, leaving g(q) = f(q) correct.  This moves the surrogate's
           class boundary, so a threshold instrument does respond - to the boundary
           sliding past, not to the slope.
  sharpen  the whole log-odds, so the class boundary is FIXED and only the confidence
           either side of it changes.  This is the axis Logit-LIME differs from standard
           LIME on, and the arithmetic says a threshold at 0.5 must be exactly blind to
           it: the set {g = 1/2} does not move, so not one prediction changes.

The predictions (PREREGISTRATION.md, third registration) are therefore arithmetic rather
than empirical, and a violation would be a bug rather than a finding: fidelity is exactly
constant along `sharpen`, fidelity-at-f(q) is exactly constant along `scale`, and both
proper scoring rules are minimised at the truth on all three.  In theta the thresholded
instruments should show a V at a query point on the decision boundary, flattening as the
boundary leaves the neighbourhood.

Three query points per configuration, chosen as the points of the standard between-means
line whose f(q) is nearest 0.5, 0.9 and 0.99: the boundary case LIME is usually drawn
for, and the confident cases it is usually used on.

usage:  python sweeps/sweep_instruments.py [results_instruments.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, surrogates

import argparse
import json
import warnings
import numpy as np
import clime
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

# the 2-D case is the one that can be drawn; the 30-D one says whether the shapes are a
# property of the instruments or of a two-dimensional toy
CONFIGS = [('Gaussian', 'Logistic'), ('Breast Cancer', 'Logistic')]

# (name, metric key, evaluation data) - the ladder, from the coarsest instrument to the
# finest.  The first two differ only in where they are measured, which is the other axis
# of sweep_fidelity.py's 2x2.
INSTRUMENTS = [
    ('fidelity | local sample', 'fidelity (local)', 'local'),
    ('fidelity | test data', 'fidelity (local)', 'test'),
    ('fidelity at f(q)', 'fidelity (local query probs)', 'local'),
    ('Spearman', 'spearman', 'local'),
    ('Brier', 'Brier score (local)', 'local'),
    ('KL', 'KL divergence (local)', 'local'),
]

# lower is better for these; the rest are agreements, where higher is better
LOWER_IS_BETTER = {'Brier', 'KL'}

TARGET_PROBS = [0.5, 0.9, 0.99]
THETAS = np.linspace(-np.pi, np.pi, 73)          # 5 degree steps
SCALES = np.logspace(-1, 1, 41)                  # 0.1x to 10x the true norm
PLANE_SEED = 0                                   # the rotation plane, when d > 2

# the three one-parameter sweeps, as keyword overrides of the family's (theta, scale,
# sharpen) = (0, 1, 1) truth
AXES = {'theta': [dict(theta=t) for t in THETAS],
        'scale': [dict(scale=s) for s in SCALES],
        'sharpen': [dict(sharpen=s) for s in SCALES]}


def score(metric, expl, clf, data, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = metric(expl, black_box_model=clf, data=data, query_point=q)
    return float(v)


def run_point(clf, test_data, q, truth, metrics, n_eval):
    '''the full response surface of every instrument at one query point'''
    eval_sets = {'local': get_local_points(test_data, q, samples=n_eval),
                 'test': test_data}
    intercept = float(surrogates._logit(
        np.asarray(clf.predict_proba(q[None, :]))[0, 1]))
    partner = surrogates.orthonormal_partner(truth, seed=PLANE_SEED)

    out = {}
    for name, key, where in INSTRUMENTS:
        metric, data = metrics[key], eval_sets[where]
        out[name] = {
            axis: [score(metric,
                         surrogates.family_member(q, intercept, truth, partner=partner,
                                                  **kw),
                         clf, data, q)
                   for kw in settings]
            for axis, settings in AXES.items()}
    return out


def fitted_surrogates(clf, train_data, test_data, q, truth):
    '''
    where the real surrogates sit on the theta axis.

    Only the angle is recorded, not the norm: standard LIME regresses probabilities, so
    its coefficients are a gradient of p rather than of logit p and its norm is not on the
    same scale as the truth's (common/gradients.py).  The angle is scale free, so it is
    comparable for all three.
    '''
    out = {}
    for label, name in (('standard', 'bLIMEy (normal)'), ('logit', 'bLIMEy (logit)'),
                        ('logreg', 'bLIMEy (logistic regression)')):
        c = np.asarray(clime.explainer.AVAILABLE_EXPLAINERS[name](
            clf, q, train_data=train_data, test_data=test_data).get_explanation(),
            dtype=float)
        n = np.linalg.norm(c)
        out[label] = float(c @ truth/(n*np.linalg.norm(truth))) if n > 0 else float('nan')
    return out


def run_config(dataset, model, metrics, n_eval):
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test_data)
    Q = np.asarray(qs, dtype=np.float64)
    p1 = np.asarray(clf.predict_proba(Q))[:, 1]
    truth = gradients.grad_logit(clf, model, Q)

    entry = {'n_features': int(Q.shape[1]), 'n_test': int(len(test_data['X'])),
             'points': []}
    for target in TARGET_PROBS:
        i = int(np.argmin(np.abs(p1 - target)))
        q = Q[i]
        entry['points'].append({
            'index': i, 'target': target, 'f_q': float(p1[i]),
            'truth_norm': float(np.linalg.norm(truth[i])),
            'cos_fitted': fitted_surrogates(clf, train_data, test_data, q, truth[i]),
            'curves': run_point(clf, test_data, q, truth[i], metrics, n_eval)})
        print(f'  point {i:2d}  f(q)={p1[i]:.4f}  |grad|={np.linalg.norm(truth[i]):.3f}',
              flush=True)
    return entry


def run(out_path, n_eval):
    out_path = paths.results(out_path)
    metrics = {key: clime.evaluation.AVAILABLE_EVALUATION_METRICS[key]
               for _, key, _ in INSTRUMENTS}

    out = {'_meta': {'seed': clime.RANDOM_SEED, 'n_eval': n_eval,
                     'thetas_deg': np.degrees(THETAS).tolist(),
                     'scales': SCALES.tolist(), 'axes': list(AXES),
                     'plane_seed': PLANE_SEED,
                     'instruments': [list(i) for i in INSTRUMENTS],
                     'lower_is_better': sorted(LOWER_IS_BETTER),
                     'target_probs': TARGET_PROBS}}
    for dataset, model in CONFIGS:
        print(f'{dataset} | {model}', flush=True)
        out[f'{dataset}|{model}'] = run_config(dataset, model, metrics, n_eval)
        json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_instruments.json')
    p.add_argument('--n-eval', type=int, default=100,
                   help='evaluation points per query point (100 is the study protocol)')
    a = p.parse_args()
    run(a.out, a.n_eval)
