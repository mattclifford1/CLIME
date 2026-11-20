'''
Explanations of each differentiable black box before and after its decision threshold is moved.

The move adds a constant to the log-odds, so the gradient of the log-odds, the truth
of sweep_gradient_truth.py, is the same before and after.

usage:  python sweeps/sweep_threshold.py [results_threshold.json] [--datasets ...] [--jobs N]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, threshold

import argparse
import json
import warnings
import numpy as np
import clime
from sweeps.sweep import opts, DATASETS, GROUP_OF, METRICS
from sweeps.sweep_extended import NEW_GROUPS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)'}
MODELS = list(gradients.ANALYTIC_GRADIENTS)
GROUPS = {**GROUP_OF, **NEW_GROUPS, 'Bayes Optimal': 'B quadratic'}
THRESHOLDS = (0.99, 0.9, 0.1, 0.01)     # moved to, from 0.5


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else float('nan')


def coefficients(box, q, train_data, test_data):
    return {label: np.asarray(clime.explainer.AVAILABLE_EXPLAINERS[name](
                box, q, train_data=train_data, test_data=test_data).get_explanation(), dtype=float)
            for label, name in SURROGATES.items()}


def run_config(dataset, model):
    r = clime.pipeline.run_pipeline(opts(dataset, model, SURROGATES['standard'], METRICS[0]),
                                    parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs = np.asarray(get_points_between_class_means(test_data)[0], dtype=np.float64)
    truth = gradients.grad_logit(clf, model, qs)
    entry = {'group': GROUPS.get(model, 'unassigned'), 'n_features': int(qs.shape[1]), 'points': []}
    for q, g in zip(qs, truth):
        before = coefficients(threshold.Shifted(clf, 0), q, train_data, test_data)
        point = {'log_odds': float(threshold.Shifted(clf, 0).log_odds(q[None, :])[0]), 'moves': {}}
        for t in THRESHOLDS:
            after = coefficients(threshold.at_threshold(clf, t), q, train_data, test_data)
            point['moves'][str(t)] = {
                label: {'norm_ratio': float(np.linalg.norm(after[label])/np.linalg.norm(before[label])),
                        'cos': cosine(after[label], before[label]),
                        'same_top': float(np.argmax(np.abs(after[label]))
                                          == np.argmax(np.abs(before[label]))),
                        'cos_truth': cosine(after[label], g)}
                for label in SURROGATES}
        point['cos_truth'] = {label: cosine(before[label], g) for label in SURROGATES}
        entry['points'].append(point)
    return entry


def run(out_path, datasets, models=MODELS):
    out_path = paths.results(out_path)
    out = {}
    if os.path.exists(out_path):
        out = {k: v for k, v in json.load(open(out_path)).items() if 'error' not in v}
        print(f'resuming: {len(out)} already done', flush=True)
    for dataset in datasets:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                entry = run_config(dataset, model)
                ratio = np.median([abs(np.log(p['moves']['0.1']['standard']['norm_ratio']))
                                   for p in entry['points']])
                print(f'{dataset:26s} {model:28s} LIME size x{np.exp(ratio):.2f} at 0.1', flush=True)
            except Exception as e:
                entry = {'error': f'{type(e).__name__}: {e}'}
                print(f'{key:56s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='results_threshold.json')
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    if args.jobs > 1:
        from sweeps import parallel
        parallel.run(name=args.out, datasets=args.datasets, jobs=args.jobs,
                     setup='from sweeps import sweep_threshold as m',
                     call='m.run({out!r}, {datasets!r})')
    else:
        run(args.out, args.datasets)
