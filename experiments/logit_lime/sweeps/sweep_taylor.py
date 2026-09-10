'''
What does an explanation-optimal surrogate cost in fidelity?

Registered in PREREGISTRATION.md (second registration) before running. The claim under
test is the account given for why logit space improves the explanation of a group B or C
black box without improving its fidelity: that a fitted surrogate trades slope for
intercept to repair its probabilities over the neighbourhood, and the slope is what the
user reads.

If that is right, the surrogate that keeps the slope and accepts whatever the intercept
gives - the first-order Taylor expansion of the log-odds at q - should be worse on Brier
and KL than a fitted surrogate on exactly those groups, and equal to it on group A where
the expansion is the black box.

Four surrogates per query point, all scored the same way as everywhere else in this study:

  standard LIME, Logit-LIME     fitted, as before
  Taylor (analytic)             oracle: coefficients ARE the ground truth, so its cosine
                                is 1 by construction and only its fidelity is a result
  Taylor (finite difference)    model-agnostic, 2d black-box queries, so both its
                                fidelity and its explanation accuracy are results

usage:  python sweeps/sweep_taylor.py [results_taylor.json] [--datasets N] [--models ...]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients
from common.taylor import AnalyticTaylor, FiniteDifferenceTaylor

import argparse
import json
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweeps.sweep import opts, DATASETS, METRICS
from sweeps.sweep_gradient_truth import GROUPS, SURROGATES, cosine
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

FITTED = {'standard': SURROGATES['standard'], 'logit': SURROGATES['logit']}
MODELS = list(gradients.ANALYTIC_GRADIENTS)


def score(expl, clf, eval_data, q, truth, metrics):
    c = np.asarray(expl.get_explanation(), dtype=float)
    rho = spearmanr(np.abs(c), np.abs(truth))[0] if c.size > 1 else np.nan
    row = {'cos': cosine(c, truth),
           'rho': float(rho) if np.isfinite(rho) else float('nan'),
           'top1': float(np.argmax(np.abs(c)) == np.argmax(np.abs(truth)))
                   if np.any(c) else 0.0}
    for name, metric in metrics.items():
        row[name] = float(metric(expl, black_box_model=clf, data=eval_data,
                                 query_point=q))
    return row


def run_config(dataset, model, metrics):
    r = clime.pipeline.run_pipeline(opts(dataset, model, SURROGATES['standard'],
                                         METRICS[0]), parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test_data)
    Q = np.asarray(qs, dtype=np.float64)
    truth = gradients.grad_logit(clf, model, Q)
    p1 = np.asarray(clf.predict_proba(Q))[:, 1]
    sat = (p1 < 1e-9) | (p1 > 1 - 1e-9)

    entry = {'group': GROUPS.get(model, 'unassigned'),
             'n_features': int(Q.shape[1]), 'saturation': float(np.mean(sat)),
             'points': []}
    for i, q in enumerate(qs):
        if np.linalg.norm(truth[i]) == 0 or not np.all(np.isfinite(truth[i])):
            continue
        q = np.asarray(q, dtype=np.float64)
        eval_data = get_local_points(test_data, q)
        point = {'saturated': bool(sat[i])}
        for label, name in FITTED.items():
            expl = clime.explainer.AVAILABLE_EXPLAINERS[name](
                clf, q, train_data=train_data, test_data=test_data)
            point[label] = score(expl, clf, eval_data, q, truth[i], metrics)
        an = AnalyticTaylor(clf, q, model_name=model)
        point['taylor'] = score(an, clf, eval_data, q, truth[i], metrics)
        fd = FiniteDifferenceTaylor(clf, q)
        point['taylor_fd'] = score(fd, clf, eval_data, q, truth[i], metrics)
        point['taylor_fd']['degenerate'] = bool(fd.degenerate)
        point['taylor_fd']['step'] = float(fd.step_used)
        point['n_queries_fd'] = int(fd.n_queries)
        entry['points'].append(point)

    for label in ('standard', 'logit', 'taylor', 'taylor_fd'):
        for stat in ('cos', 'rho', 'top1', *metrics):
            vals = [p[label][stat] for p in entry['points']
                    if np.isfinite(p[label][stat])]
            entry[f'{stat}_{label}'] = float(np.mean(vals)) if vals else float('nan')
    entry['fd_degenerate'] = float(np.mean([p['taylor_fd']['degenerate']
                                            for p in entry['points']]))
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
                e = run_config(dataset, model, metrics)
                print(f"{dataset:24s} {model:28s} "
                      f"KL logit={e[f'{METRICS[1]}_logit']:.2e} "
                      f"taylor={e[f'{METRICS[1]}_taylor']:.2e}   "
                      f"cos std={e['cos_standard']:.3f} logit={e['cos_logit']:.3f} "
                      f"fd={e['cos_taylor_fd']:.3f}", flush=True)
            except Exception as exc:
                e = {'error': f'{type(exc).__name__}: {exc}'}
                print(f'{key:54s} FAILED {e["error"][:60]}', flush=True)
            out[key] = e
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_taylor.json')
    p.add_argument('--datasets', type=int, default=len(DATASETS))
    p.add_argument('--models', type=str, default=None)
    a = p.parse_args()
    run(a.out, DATASETS[:a.datasets], a.models.split(',') if a.models else MODELS)
