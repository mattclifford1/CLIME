'''
Main Logit-LIME sweep: datasets x black boxes x surrogates x metrics.

For every (dataset, black box) pair we record
  (a) the quality of each surrogate along the between-class-means line, and
  (b) the diagnostic computed from the BLACK BOX ALONE - how much better a locality
      weighted linear model fits its log-odds than its probabilities.

The black boxes are grouped by the a priori geometry of their log-odds; see
PREREGISTRATION.md, written before this was run.

usage:  python sweep.py <output.json> [seed]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASETS = ['Gaussian', 'Moons', 'Circles',
            'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes',
            'Iris', 'Wine', 'Sonar Rocks vs Mines', 'Ionosphere', 'Wheat Seeds',
            'Abalone Gender', 'Credit Scoring 1', 'Direct Marketing']

# grouped by the predicted geometry of their log-odds (PREREGISTRATION.md)
MODEL_GROUPS = {
    'A linear': ['Logistic', 'LDA'],
    'B quadratic': ['QDA', 'Gaussian Naive Bayes'],
    'C smooth': ['MLP', 'SVM'],
    'D piecewise constant': ['Decision Tree', 'Random Forest', 'k Nearest Neighbours'],
    'E calibrated forest': ['Random Forest (Platt calibrated)',
                            'Random Forest (isotonic calibrated)'],
    'unassigned': ['Gradient Boosting'],
}
MODELS = [m for group in MODEL_GROUPS.values() for m in group]
GROUP_OF = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}

EXPLAINERS = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']
METRICS = ['Brier score (local)', 'KL divergence (local)']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def opts(dataset, model, explainer, metric):
    return {'dataset': dataset, 'data params': DATA_PARAMS, 'standardise data': True,
            'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none',
            'explainer': explainer, 'evaluation metric': metric,
            'evaluation points': 'between_class_means', 'evaluation data': 'sample locally'}


def weighted_r2(A, target, w):
    '''R^2 of a distance weighted linear fit - the same fit the surrogate performs'''
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A*sw[:, None], target*sw, rcond=None)
    resid = target - A @ coef
    ss_res = np.sum(w*resid**2)
    mean = np.sum(w*target)/np.sum(w)
    ss_tot = np.sum(w*(target-mean)**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


def diagnostic(clf, test_data, query_points):
    '''how linear are the black box's log-odds vs its probabilities, locally?'''
    r2_logit, r2_prob, saturation = [], [], []
    for q in query_points:
        X = get_local_points(test_data, q, samples=2000)['X']
        p = clf.predict_proba(X)[:, 1].astype(np.float64)
        w = costs.weights_based_on_distance(q, X)
        saturation.append(float(((p <= 1e-6) | (p >= 1-1e-6)).mean()))
        pc = np.clip(p, 1e-9, 1-1e-9)
        A = np.c_[X, np.ones(len(X))]
        r2_logit.append(weighted_r2(A, np.log(pc/(1-pc)), w))
        r2_prob.append(weighted_r2(A, p, w))
    return (float(np.nanmean(r2_logit)), float(np.nanmean(r2_prob)),
            float(np.mean(saturation)))


def run(out_path, seed=None):
    if seed is not None:
        # models and dataset splits read this at construction time
        clime.RANDOM_SEED = int(seed)
        np.random.seed(int(seed))

    out = {'_meta': {'seed': seed if seed is not None else clime.RANDOM_SEED,
                     'groups': MODEL_GROUPS}}
    # resume: results are deterministic per (dataset, model) since B10 was fixed, so a
    # run interrupted part way can pick up where it stopped rather than recompute
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} configurations already done', flush=True)

    for dataset in DATASETS:
        for model in MODELS:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            entry = {'group': GROUP_OF[model], 'metrics': {}}
            try:
                for metric in METRICS:
                    entry['metrics'][metric] = {}
                    for expl in EXPLAINERS:
                        # serial: parallel evaluation is slower here and the pipeline
                        # cache makes repeated black box fits cheap
                        r = clime.pipeline.run_pipeline(opts(dataset, model, expl, metric),
                                                        parallel_eval=False)
                        s = np.array(r['score']['scores'])
                        entry['metrics'][metric][expl] = {'mean': float(s.mean()),
                                                          'scores': s.tolist()}
                        entry['model_stats'] = {k: float(v) for k, v in r['model_stats'].items()}
                base = clime.pipeline.run_pipeline(opts(dataset, model, EXPLAINERS[0], METRICS[0]),
                                                   parallel_eval=False)
                qs, _ = get_points_between_class_means(base['test_data'])
                rl, rp, sat = diagnostic(base['clf'], base['test_data'], qs)
                entry['diagnostic'] = {'r2_logit': rl, 'r2_prob': rp, 'gap': rl-rp,
                                       'saturation': sat}
            except Exception as e:
                entry['error'] = f'{type(e).__name__}: {e}'
                print(f'{key:60s} FAILED {entry["error"][:60]}', flush=True)
                out[key] = entry
                continue

            out[key] = entry
            m = entry['metrics']['Brier score (local)']
            adv = m['bLIMEy (normal)']['mean']/max(m['bLIMEy (logit)']['mean'], 1e-30)
            print(f"{dataset:26s} {model:36s} [{entry['group'][0]}] "
                  f"gap={rl-rp:+.3f} sat={sat:5.1%} advantage={adv:10.2f}x", flush=True)
            json.dump(out, open(out_path, 'w'))   # checkpoint as we go

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    run(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else None)
