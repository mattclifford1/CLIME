'''
Does the choice of surrogate space change the explanation, or only its fidelity?

Everything else in this study measures how well the surrogate reproduces the black box.
That is not what a LIME user consumes: they read off feature importances. If standard LIME
and Logit-LIME rank features identically, the fidelity gains reported elsewhere are of no
practical consequence, and we need to know that.

Coefficient magnitudes are not comparable across the two surrogates - one regresses
probabilities, the other log-odds, so the units differ by a factor that varies with the
local slope. Rankings are comparable, and rankings are what get shown to users. We report

  - Spearman rank correlation between the two coefficient vectors
  - top-k set overlap (k = 3, and k = 5 where the data has enough features)
  - sign agreement on the shared top-k

usage:  python sweep_explanations.py <output.json>
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import os
import json
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweeps.sweep import opts, DATASETS, MODELS, GROUP_OF, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

STANDARD, LOGIT = 'bLIMEy (normal)', 'bLIMEy (logit)'


def topk_overlap(a, b, k):
    '''fraction of the top-k features by |coefficient| that the two surrogates share'''
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb)/k, sorted(ta & tb)


def compare(clf, train_data, test_data, query_points):
    rho, ov3, ov5, sign, top1 = [], [], [], [], []
    for q in query_points:
        q = np.array(q)
        e_std = clime.explainer.AVAILABLE_EXPLAINERS[STANDARD](
            clf, q, test_data=test_data)
        e_log = clime.explainer.AVAILABLE_EXPLAINERS[LOGIT](
            clf, q, test_data=test_data)
        a = np.asarray(e_std.get_explanation(), dtype=float)
        b = np.asarray(e_log.get_explanation(), dtype=float)
        if a.shape != b.shape or a.size < 2:
            continue
        r = spearmanr(np.abs(a), np.abs(b))[0]
        rho.append(float(r) if np.isfinite(r) else np.nan)

        f3, shared3 = topk_overlap(a, b, min(3, a.size))
        ov3.append(f3)
        if a.size >= 5:
            ov5.append(topk_overlap(a, b, 5)[0])
        # do the shared important features at least point the same way?
        if shared3:
            sign.append(float(np.mean(np.sign(a[shared3]) == np.sign(b[shared3]))))
        top1.append(float(np.argmax(np.abs(a)) == np.argmax(np.abs(b))))

    def m(x):
        return float(np.nanmean(x)) if len(x) else float('nan')
    return dict(rho=m(rho), overlap3=m(ov3), overlap5=m(ov5),
                sign_agree=m(sign), top1_agree=m(top1), n_points=len(rho))


def run(out_path):
    out_path = paths.results(out_path)   # a bare name lands in results/

    out = {}
    if os.path.exists(out_path):
        out = {k: v for k, v in json.load(open(out_path)).items()
               if not k.startswith('_') and 'error' not in v}
        print(f'resuming: {len(out)} already done', flush=True)

    for dataset in DATASETS:
        for model in MODELS:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            entry = {'group': GROUP_OF[model]}
            try:
                r = clime.pipeline.run_pipeline(
                    opts(dataset, model, STANDARD, METRICS[0]), parallel_eval=False)
                qs, _ = get_points_between_class_means(r['test_data'])
                entry.update(compare(r['clf'], r['train_data'], r['test_data'], qs))
                entry['n_features'] = int(np.asarray(r['test_data']['X']).shape[1])
                print(f"{dataset:26s} {model:36s} rho={entry['rho']:+.3f} "
                      f"top3={entry['overlap3']:.2f} top1={entry['top1_agree']:.2f} "
                      f"sign={entry['sign_agree']:.2f}", flush=True)
            except Exception as e:
                entry['error'] = f'{type(e).__name__}: {e}'
                print(f'{key:60s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_explanations.json'))
