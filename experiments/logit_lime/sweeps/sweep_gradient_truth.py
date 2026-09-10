'''
Explanation ground truth for every differentiable black box, from the analytic gradient.

sweep_ground_truth.py can only ask "which surrogate is right?" for the three black boxes
whose log-odds are exactly linear, because only there does a constant coefficient vector
exist to be right about. The local generalisation - the gradient of the log-odds at the
query point - is defined for every differentiable black box, and reduces to those same
coefficients when the log-odds are linear. common/gradients.py derives it in closed form
for eleven of them and validate_gradients.py checks each against finite differences.

This sweep does two things at once, deliberately:

  1. scores each surrogate's explanation against that truth (cosine, rank correlation,
     top-1), which extends the ground truth question from group A to groups B and C;

  2. records the surrogate's local Brier score and KL divergence at the same query point,
     from the same fitted surrogate object. That pairing is what lets us ask whether
     fidelity - the quantity the rest of the study measures - actually predicts
     explanation correctness. It is the justification for using Brier and KL as proxies
     at all, and it can only be checked where a ground truth exists.

Both quantities come from one surrogate fit per (configuration, surrogate, query point):
the metric is computed here exactly as get_key_points_score does it, on the same locally
sampled evaluation points, rather than taken from a separate pipeline run.

usage:  python sweeps/sweep_gradient_truth.py [results_gradient_truth.json] [--datasets ...]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients

import argparse
import json
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweeps.sweep import opts, DATASETS, GROUP_OF, METRICS
from sweeps.sweep_extended import NEW_GROUPS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)',
              'logreg': 'bLIMEy (logistic regression)'}
MODELS = list(gradients.ANALYTIC_GRADIENTS)
GROUPS = {**GROUP_OF, **NEW_GROUPS, 'Bayes Optimal': 'B quadratic'}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else float('nan')


def evaluate(clf, train_data, test_data, q, truth, metrics):
    '''one query point: fit each surrogate once, score its explanation and its fidelity'''
    eval_data = get_local_points(test_data, q)
    out = {}
    for label, name in SURROGATES.items():
        expl = clime.explainer.AVAILABLE_EXPLAINERS[name](
            clf, q, train_data=train_data, test_data=test_data)
        c = np.asarray(expl.get_explanation(), dtype=float)
        row = {'cos': cosine(c, truth),
               'top1': float(np.argmax(np.abs(c)) == np.argmax(np.abs(truth)))}
        rho = spearmanr(np.abs(c), np.abs(truth))[0] if c.size > 1 else np.nan
        row['rho'] = float(rho) if np.isfinite(rho) else float('nan')
        for metric_name, metric in metrics.items():
            row[metric_name] = float(metric(expl, black_box_model=clf, data=eval_data,
                                            query_point=q))
        out[label] = row
    return out


def run_config(dataset, model, metrics):
    r = clime.pipeline.run_pipeline(opts(dataset, model, SURROGATES['standard'],
                                         METRICS[0]), parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test_data)
    Q = np.asarray(qs, dtype=np.float64)

    truth = gradients.grad_logit(clf, model, Q)
    p1 = np.asarray(clf.predict_proba(Q))[:, 1]
    entry = {'group': GROUPS.get(model, 'unassigned'),
             'n_features': int(Q.shape[1]),
             'n_points': int(Q.shape[0]),
             'saturation': float(np.mean((p1 < 1e-9) | (p1 > 1 - 1e-9))),
             'truth_norm': [float(np.linalg.norm(g)) for g in truth],
             'points': []}
    # where the log-odds are exactly linear the gradient is constant and equal to coef_,
    # which is the check that this instrument extends the coefficient one
    entry['truth_is_constant'] = bool(
        np.allclose(truth, truth[0], rtol=1e-9, atol=1e-12))

    for i, q in enumerate(qs):
        if np.linalg.norm(truth[i]) == 0 or not np.all(np.isfinite(truth[i])):
            continue          # no direction to be right about
        entry['points'].append(evaluate(clf, train_data, test_data, np.array(q),
                                        truth[i], metrics))
    for label in SURROGATES:
        for stat in ('cos', 'rho', 'top1', *metrics):
            vals = [p[label][stat] for p in entry['points']]
            vals = [v for v in vals if np.isfinite(v)]
            entry[f'{stat}_{label}'] = float(np.mean(vals)) if vals else float('nan')
    return entry


def run(out_path, datasets, models):
    out_path = paths.results(out_path)
    metrics = {m: clime.evaluation.AVAILABLE_EVALUATION_METRICS[m] for m in METRICS}

    out = {}
    if os.path.exists(out_path):
        out = {k: v for k, v in json.load(open(out_path)).items()
               if not k.startswith('_') and 'error' not in v}
        print(f'resuming: {len(out)} already done', flush=True)

    print(f'{len(datasets)} datasets x {len(models)} black boxes '
          f'= {len(datasets)*len(models)} configurations', flush=True)
    for dataset in datasets:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                entry = run_config(dataset, model, metrics)
                print(f"{dataset:26s} {model:28s} n={len(entry['points']):3d} "
                      f"const={str(entry['truth_is_constant']):5s} "
                      f"cos std={entry['cos_standard']:+.3f} "
                      f"logit={entry['cos_logit']:+.3f}", flush=True)
            except Exception as e:
                entry = {'error': f'{type(e).__name__}: {e}'}
                print(f'{key:56s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_gradient_truth.json')
    p.add_argument('--datasets', type=int, default=len(DATASETS))
    p.add_argument('--models', type=str, default=None,
                   help='comma separated subset of the gradient black boxes')
    a = p.parse_args()
    run(a.out, DATASETS[:a.datasets],
        a.models.split(',') if a.models else MODELS)
