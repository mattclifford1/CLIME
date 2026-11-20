'''
Cosine of each explanation with the weights of an exactly-linear black box, against the
number of samples in the neighbourhood.

usage:  python sweeps/sweep_samples.py [results_samples.json] [--datasets ...] [--jobs N]
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
from sweeps.sweep import opts, DATASETS, METRICS
from sweeps.sweep_gradient_truth import GROUPS, cosine
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)'}
MODELS = [m for m in gradients.ANALYTIC_GRADIENTS if GROUPS.get(m) == 'A linear']
SAMPLES = (100, 300, 1000, 3000, 10000, 30000, 100000)


def run_config(dataset, model):
    r = clime.pipeline.run_pipeline(opts(dataset, model, SURROGATES['standard'], METRICS[0]),
                                    parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs = np.asarray(get_points_between_class_means(test_data)[0], dtype=np.float64)
    truth = gradients.grad_logit(clf, model, qs)
    entry = {'n_features': int(qs.shape[1]), 'cos': {label: {} for label in SURROGATES}}
    for label, name in SURROGATES.items():
        for n in SAMPLES:
            entry['cos'][label][str(n)] = [
                cosine(np.asarray(clime.explainer.AVAILABLE_EXPLAINERS[name](
                    clf, q, train_data=train_data, test_data=test_data,
                    samples=n).get_explanation(), dtype=float), g)
                for q, g in zip(qs, truth)]
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
                print(f'{dataset:26s} {model:20s} at 1,000 samples: ' + '  '.join(
                    f'{label} {np.nanmean(entry["cos"][label]["1000"]):.3f}'
                    for label in SURROGATES), flush=True)
            except Exception as e:
                entry = {'error': f'{type(e).__name__}: {e}'}
                print(f'{key:56s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='results_samples.json')
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    if args.jobs > 1:
        from sweeps import parallel
        parallel.run(name=args.out, datasets=args.datasets, jobs=args.jobs,
                     setup='from sweeps import sweep_samples as m',
                     call='m.run({out!r}, {datasets!r})')
    else:
        run(args.out, args.datasets)
