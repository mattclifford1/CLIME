'''
one pass per configuration: build each explainer once, score it against everything

`clime.pipeline.run_pipeline` keys its cache on the whole options dict, so asking for a
second metric - or a second evaluation distribution - rebuilds every explainer from
scratch. This study wants 3 metrics x 2 evaluation distributions x an agreement-with-truth
score for each of 6 weighting schemes, which through the pipeline would be 36 builds of
each explainer instead of 1.

So the evaluation loop is written out here: build the black box once per configuration,
then for each query point build each surrogate once and score that one object against
everything. Every registry lookup still goes through clime, so the schemes and metrics are
the same objects the rest of the repo uses.

Two deliberate departures from experiments/logit_lime:

  - the local evaluation sample is 500 points, not the 100 of
    `key_points.get_local_points`. Scores here are compared between weighting schemes at
    the same query point, and 100 points put more noise on that difference than the
    difference itself in the tails
  - evaluation on `test data` and on `sample locally` happens in the same pass, on the same
    surrogate, so the two can be differenced per query point rather than per configuration
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import json
import numpy as np
from scipy.stats import spearmanr
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import (get_points_between_class_means, get_data_grid,
                                         get_local_points)

from . import weights as weighting

# the six schemes of PREREGISTRATION.md, in reporting order
SCHEMES = ('bLIMEy (normal)',
           'bLIMEy (cost sensitive sampled)',
           'bLIMEy (cost sensitive class)',
           'bLIMEy (local y)',
           'bLIMEy (local yhat)',
           'bLIMEy (density ratio)')

METRICS = ('fidelity (local)', 'Brier score (local)', 'KL divergence (local)')
EVAL_DATA = ('test data', 'sample locally')
LOCAL_EVAL_SAMPLES = 500
SURROGATE_SAMPLES = 10000   # bLIMEy's default, restated so the diagnostic can match it

_CACHE = {}


def _opts(dataset, model, rebalancing, data_params):
    return {'dataset': dataset, 'data params': data_params or {},
            'standardise data': True, 'dataset rebalancing': rebalancing,
            'model': model, 'model balancer': 'none'}


def get_data_and_model(dataset, model, rebalancing='none', data_params=None):
    '''fit once per process: several query points and six schemes share one black box'''
    key = (dataset, model, rebalancing, json.dumps(data_params or {}, sort_keys=True))
    if key not in _CACHE:
        pipeline = clime.pipeline.construct(_opts(dataset, model, rebalancing, data_params))
        _CACHE[key] = pipeline.get_data_and_model()
    return _CACHE[key]


def query_points(train_data, test_data, eval_points='between_class_means', num_points=20):
    if eval_points == 'between_class_means':
        points, _ = get_points_between_class_means(test_data, num_samples=num_points)
        return [np.asarray(q, dtype=np.float64) for q in points]
    if eval_points == 'grid':
        grid = get_data_grid(train_data, test_data, num_samples=num_points)
        return [np.asarray(q, dtype=np.float64) for q in grid]
    raise ValueError(f'unsupported evaluation points: {eval_points}')


def _weighted_rate(mask, w):
    total = w.sum()
    return float((w*mask).sum()/total) if total > 0 else float('nan')


def diagnostics(clf, train_data, test_data, query_point):
    '''
    how far apart P(y|x) and P(yhat|x) are near this query point, and whether CIKM's class
    weights look like a density ratio

    The sampled points are regenerated with the same salt the surrogates use, so the
    weight comparison is made on exactly the points a surrogate would be trained on.
    '''
    w = costs.weights_based_on_distance(query_point, test_data['X'])
    y = np.asarray(test_data['y']).astype(np.int64)
    yhat = np.asarray(clf.predict(test_data['X'])).astype(np.int64)
    out = {'local y balance': _weighted_rate(y == 1, w),
           'local yhat balance': _weighted_rate(yhat == 1, w),
           'local disagreement': _weighted_rate(yhat != y, w),
           'black box vs truth': _weighted_rate(yhat == y, w)}

    rng = clime.utils.rng_from_point(query_point, salt='surrogate training sample')
    cov = np.cov(test_data['X'].T)
    X_s = rng.multivariate_normal(query_point, cov, SURROGATE_SAMPLES)
    yhat_s = np.asarray(clf.predict(X_s)).astype(np.int64)
    out['sample yhat balance'] = float((yhat_s == 1).mean())

    # the two corrections, on the same points: CIKM's class weights and the density ratio
    cikm = weighting.class_weights_from_labels(yhat_s)[yhat_s]
    ratio = weighting.density_ratio_weights(X_s, train_data['X'])
    if np.ptp(cikm) == 0 or np.ptp(ratio) == 0:
        out['weight agreement'] = None   # one correction is constant: no rank to compare
    else:
        rho = spearmanr(cikm, ratio).statistic
        out['weight agreement'] = float(rho) if np.isfinite(rho) else None
    return out


def evaluate_config(dataset, model, schemes=SCHEMES, *, rebalancing='none',
                    data_params=None, eval_points='between_class_means', num_points=20,
                    metrics=METRICS, local_samples=LOCAL_EVAL_SAMPLES,
                    with_diagnostics=True):
    '''
    every number this study records for one (dataset, black box) pair

    returns per-query-point lists, not means: P1 and P2 are about how the scores vary
    ALONG the line, so a configuration mean would average the effect away
    '''
    train_data, test_data, clf = get_data_and_model(dataset, model, rebalancing, data_params)
    points = query_points(train_data, test_data, eval_points, num_points)

    out = {'dataset': dataset, 'model': model, 'rebalancing': rebalancing,
           'eval_points': eval_points, 'n_query_points': len(points),
           'n_features': int(test_data['X'].shape[1]),
           'n_train': int(train_data['X'].shape[0]),
           'n_test': int(test_data['X'].shape[0]),
           'model_stats': {k: float(v) for k, v in
                           clime.utils.get_model_stats(clf, train_data, test_data).items()},
           'schemes': {s: {**{d: {m: [] for m in metrics} for d in EVAL_DATA},
                           'surrogate vs truth': []} for s in schemes},
           'diagnostics': {}, 'failures': 0}

    diag_rows = []
    for query_point in points:
        local = get_local_points(test_data, query_point, samples=local_samples)
        eval_sets = {'test data': test_data, 'sample locally': local}
        y_test = np.asarray(test_data['y']).astype(np.int64)
        w_test = costs.weights_based_on_distance(query_point, test_data['X'])

        for scheme in schemes:
            record = out['schemes'][scheme]
            try:
                expl = clime.explainer.AVAILABLE_EXPLAINERS[scheme](
                    black_box_model=clf, query_point=query_point,
                    train_data=train_data, test_data=test_data)
            except Exception as e:                      # noqa: BLE001 - recorded, not raised
                out['failures'] += 1
                out.setdefault('error', f'{type(e).__name__}: {e}')
                for data_name in EVAL_DATA:
                    for metric in metrics:
                        record[data_name][metric].append(None)
                record['surrogate vs truth'].append(None)
                continue

            for data_name, data in eval_sets.items():
                for metric in metrics:
                    score = clime.evaluation.AVAILABLE_EVALUATION_METRICS[metric](
                        expl, black_box_model=clf, data=data, query_point=query_point)
                    score = float(score)
                    record[data_name][metric].append(score if np.isfinite(score) else None)
            # the other objective: agreement with the TRUE labels rather than with f
            g_pred = np.asarray(expl.predict(test_data['X'])).astype(np.int64)
            record['surrogate vs truth'].append(_weighted_rate(g_pred == y_test, w_test))

        if with_diagnostics:
            diag_rows.append(diagnostics(clf, train_data, test_data, query_point))

    if diag_rows:
        out['diagnostics'] = {k: [row[k] for row in diag_rows] for k in diag_rows[0]}
    out['query_points'] = [q.tolist() for q in points]
    return out
