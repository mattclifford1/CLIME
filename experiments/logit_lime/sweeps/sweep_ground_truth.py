'''
When the two surrogates disagree about which feature matters, which one is right?

For most black boxes there is no ground truth to check against. For black boxes whose
log-odds are exactly linear there is: the model's own coefficients ARE the local feature
importances, everywhere in the space. Logistic regression, LDA and Nearest Class Mean all
qualify, so we can score each surrogate's explanation against the truth rather than
against each other.

This is the step that turns "the two surrogates disagree" into "and one of them is wrong".

Reported per (dataset, black box), averaged over query points:
  rho_*     Spearman rank correlation between |surrogate coefficients| and |true|
  top1_*    does the surrogate pick the black box's actual most important feature
  cos_*     cosine similarity of the coefficient vectors, which unlike the rank measures
            is sensitive to getting the relative magnitudes right

usage:  python sweep_ground_truth.py [results_ground_truth.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
import json
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweep import opts, METRICS
from clime.data.loaders.exported_npz import available_exported
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

# black boxes with exactly linear log-odds, so their coefficients are the ground truth
LINEAR_MODELS = ['Logistic', 'LDA', 'Nearest Class Mean']
SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)',
              'logreg': 'bLIMEy (logistic regression)'}


def _unwrap(clf, attr):
    '''
    the pipeline wraps the estimator twice - the clime model wrapper, then the model
    balancer - so .model has to be followed until the attribute actually appears
    '''
    seen = 0
    while clf is not None and seen < 5:
        if hasattr(clf, attr):
            return clf
        clf = getattr(clf, 'model', None)
        seen += 1
    raise AttributeError(f'no {attr} found by unwrapping .model')


def true_coefficients(clf, model_name):
    '''the black box's own class-1 weights, in feature space'''
    if model_name == 'Nearest Class Mean':
        # log-odds = 2 x.(m1 - m0) + const, so the weight vector is 2(m1 - m0)
        inner = _unwrap(clf, 'means_')
        return 2.0*(inner.means_[1] - inner.means_[0])
    coef = np.atleast_2d(_unwrap(clf, 'coef_').coef_)
    return coef[-1, :]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


def run(out_path):
    datasets = list(dict.fromkeys(
        __import__('sweep').DATASETS + sorted(available_exported())))
    out = {}
    if os.path.exists(out_path):
        out = {k: v for k, v in json.load(open(out_path)).items() if 'error' not in v}
        print(f'resuming: {len(out)} done', flush=True)

    for dataset in datasets:
        for model in LINEAR_MODELS:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            entry = {}
            try:
                r = clime.pipeline.run_pipeline(
                    opts(dataset, model, SURROGATES['standard'], METRICS[0]),
                    parallel_eval=False)
                truth = np.asarray(true_coefficients(r['clf'], model), dtype=float)
                qs, _ = get_points_between_class_means(r['test_data'])
                acc = {k: {'rho': [], 'top1': [], 'cos': []} for k in SURROGATES}
                for q in qs:
                    q = np.array(q)
                    for label, name in SURROGATES.items():
                        e = clime.explainer.AVAILABLE_EXPLAINERS[name](
                            r['clf'], q, test_data=r['test_data'])
                        c = np.asarray(e.get_explanation(), dtype=float)
                        if c.shape != truth.shape:
                            continue
                        rho = spearmanr(np.abs(c), np.abs(truth))[0]
                        acc[label]['rho'].append(rho if np.isfinite(rho) else np.nan)
                        acc[label]['top1'].append(
                            float(np.argmax(np.abs(c)) == np.argmax(np.abs(truth))))
                        acc[label]['cos'].append(cosine(c, truth))
                for label in SURROGATES:
                    for stat in ('rho', 'top1', 'cos'):
                        vals = acc[label][stat]
                        entry[f'{stat}_{label}'] = (float(np.nanmean(vals))
                                                    if vals else float('nan'))
                entry['n_features'] = int(truth.size)
                entry['n_points'] = len(qs)
                print(f"{dataset:26s} {model:20s} d={truth.size:3d}  "
                      f"top1 std={entry['top1_standard']:.2f} logit={entry['top1_logit']:.2f}"
                      f"   cos std={entry['cos_standard']:+.2f} logit={entry['cos_logit']:+.2f}",
                      flush=True)
            except Exception as e:
                entry['error'] = f'{type(e).__name__}: {e}'
                print(f'{key:50s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'results_ground_truth.json')
