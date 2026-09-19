'''
Does the comparison depend on the ridge penalty? Check R of the fifth registration.

Both surrogates use alpha = 1. The logit target runs to about +/-20 while the probability
target lives in [0, 1], so the same alpha is a relatively weaker penalty in logit space.
With 10,000 samples the shrinkage is small either way, but that is an argument, and this
measures it: each surrogate is refitted at several alphas on *the same* sampled
neighbourhood and weights it was built from (bLIMEy's own sampler, same seed), and scored
with the same metric on the same evaluation points as the main sweep. At alpha = 1 this
reproduces results_taxonomy.json, which run_config checks.

usage:  python sweep_ridge_alpha.py [out.json] [--datasets A,B]
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
from clime.evaluation.key_points import get_local_points
from sweeps import sweep

warnings.filterwarnings('ignore')

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)'}


def refit(expl, clf, alpha):
    '''refit expl's surrogate at another alpha, on the neighbourhood it was built from'''
    sampled = expl._sample_locally(clf)
    w = expl._get_sampled_weights(sampled)
    expl.surrogate_model.alpha = alpha
    expl.surrogate_model.fit(sampled['X'], sampled['p(y|x)'], sample_weight=w)
    return expl


def run_config(dataset, model, metrics):
    r = clime.pipeline.run_pipeline(sweep.opts(dataset, model, SURROGATES['standard'],
                                               sweep.METRICS[0]), parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs = sweep.query_points(test_data)
    scores = {m: {s: {str(a): [] for a in ALPHAS} for s in SURROGATES} for m in metrics}
    for q in qs:
        q = np.asarray(q, dtype=np.float64)
        eval_data = get_local_points(test_data, q)
        for label, name in SURROGATES.items():
            expl = clime.explainer.AVAILABLE_EXPLAINERS[name](
                clf, query_point=q, train_data=train_data, test_data=test_data)
            for a in ALPHAS:
                refit(expl, clf, a)
                for mname, metric in metrics.items():
                    scores[mname][label][str(a)].append(float(metric(
                        expl, black_box_model=clf, data=eval_data, query_point=q)))
    entry = {'scores': scores}
    for mname in metrics:
        entry[f'advantage {mname}'] = {
            str(a): float(np.mean(scores[mname]['standard'][str(a)]) /
                          max(np.mean(scores[mname]['logit'][str(a)]), 1e-30))
            for a in ALPHAS}
    return entry


def run(out_path, datasets, models):
    out_path = paths.results(out_path)
    metrics = {m: clime.evaluation.AVAILABLE_EVALUATION_METRICS[m] for m in sweep.METRICS}
    out = {'_meta': {'alphas': ALPHAS, 'metrics': sweep.METRICS}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
    for dataset in datasets:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                e = run_config(dataset, model, metrics)
                adv = e[f'advantage {sweep.METRICS[0]}']
                print(f'{dataset:26s} {model:36s} ' +
                      ' '.join(f'a={a}:{adv[str(a)]:9.3g}x' for a in ALPHAS), flush=True)
            except Exception as exc:
                e = {'error': f'{type(exc).__name__}: {exc}'}
                print(f'{key:62s} FAILED {e["error"][:60]}', flush=True)
            out[key] = e
            json.dump(out, open(out_path, 'w'))
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_ridge_alpha.json')
    p.add_argument('--datasets', type=str, default=None)
    p.add_argument('--models', type=str, default=None)
    a = p.parse_args()
    run(a.out, a.datasets.split(',') if a.datasets else sweep.DATASETS,
        a.models.split(',') if a.models else sweep.MODELS)
