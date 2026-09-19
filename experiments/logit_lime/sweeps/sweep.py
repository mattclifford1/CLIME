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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import os
import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import (get_points_between_class_means, get_local_points,
                                        get_random_test_points)

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


# where the query points go. Every registered result uses the line between the class
# means; sweep_querypoints.py switches this to test the diagnostic somewhere else. The
# diagnostic in run() follows the same setting, so it is always computed at the points
# the surrogates are scored at
QUERY_POINTS = 'between_class_means'


def query_points(test_data):
    if QUERY_POINTS == 'between_class_means':
        return get_points_between_class_means(test_data)[0]
    if QUERY_POINTS == 'random_test_points':
        return get_random_test_points(test_data)
    raise ValueError(f'no query points for {QUERY_POINTS!r}')


def opts(dataset, model, explainer, metric):
    return {'dataset': dataset, 'data params': DATA_PARAMS, 'standardise data': True,
            'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none',
            'explainer': explainer, 'evaluation metric': metric,
            'evaluation points': QUERY_POINTS, 'evaluation data': 'sample locally'}


def weighted_r2(A, target, w):
    '''R^2 of a distance weighted linear fit - the same fit the surrogate performs'''
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A*sw[:, None], target*sw, rcond=None)
    resid = target - A @ coef
    ss_res = np.sum(w*resid**2)
    mean = np.sum(w*target)/np.sum(w)
    ss_tot = np.sum(w*(target-mean)**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


# R² is undefined when the target is constant to within rounding. weighted_r2 above tests
# ss_tot > 0, which a constant target passes: its weighted mean carries rounding error, so
# ss_tot comes out near 1e-28 rather than 0 and R² becomes a ratio of two rounding errors
# (-303 at one Arrhythmia point where every sampled probability clips to the same bound).
# That is the mechanism behind every |gap| > 1 configuration analyse.degenerate excludes.
# weighted_r2 is kept as it was so the registered numbers reproduce; guarded_r2 is what the
# fifth registration reports. The tolerance is relative to the target's own magnitude, so a
# real but small variation - a logistic model's log-odds far from its boundary - is kept.
REL_TOL = 1e-9


def guarded_r2(A, target, w):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A*sw[:, None], target*sw, rcond=None)
    mean = np.sum(w*target)/np.sum(w)
    ss_tot = np.sum(w*(target-mean)**2)
    if not ss_tot/np.sum(w) > (REL_TOL*max(1.0, abs(mean)))**2:
        return np.nan
    return 1 - np.sum(w*(target - A @ coef)**2)/ss_tot


def diagnostic_detail(clf, test_data, query_points):
    '''
    how linear are the black box's log-odds vs its probabilities, locally?

    'r2_logit', 'r2_prob', 'gap' and 'saturation' are exactly what the registered sweep
    recorded. The '_guarded' values drop query points whose target is numerically constant
    and average over the rest; 'n_defined_*' says how many were left.
    '''
    cols = {k: [] for k in ('r2_logit', 'r2_prob', 'r2_logit_guarded', 'r2_prob_guarded',
                            'saturation')}
    for q in query_points:
        X = get_local_points(test_data, q, samples=2000)['X']
        p = clf.predict_proba(X)[:, 1].astype(np.float64)
        w = costs.weights_based_on_distance(q, X)
        cols['saturation'].append(float(((p <= 1e-6) | (p >= 1-1e-6)).mean()))
        pc = np.clip(p, 1e-9, 1-1e-9)
        A = np.c_[X, np.ones(len(X))]
        lo = np.log(pc/(1-pc))
        cols['r2_logit'].append(weighted_r2(A, lo, w))
        cols['r2_prob'].append(weighted_r2(A, p, w))
        cols['r2_logit_guarded'].append(guarded_r2(A, lo, w))
        cols['r2_prob_guarded'].append(guarded_r2(A, p, w))
    out = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
           for k, v in ((k, np.asarray(v, dtype=float)) for k, v in cols.items())}
    out['gap'] = out['r2_logit'] - out['r2_prob']
    out['n_defined_logit'] = int(np.isfinite(cols['r2_logit_guarded']).sum())
    out['n_defined_prob'] = int(np.isfinite(cols['r2_prob_guarded']).sum())
    return out


def diagnostic(clf, test_data, query_points):
    '''(r2_logit, r2_prob, saturation) as registered - see diagnostic_detail'''
    d = diagnostic_detail(clf, test_data, query_points)
    return d['r2_logit'], d['r2_prob'], d['saturation']


def run(out_path, seed=None):
    out_path = paths.results(out_path)   # a bare name lands in results/

    if seed is not None:
        # models and dataset splits read this at construction time
        clime.RANDOM_SEED = int(seed)
        np.random.seed(int(seed))

    out = {'_meta': {'seed': seed if seed is not None else clime.RANDOM_SEED,
                     'groups': MODEL_GROUPS, 'query points': QUERY_POINTS}}
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
            entry = {'group': GROUP_OF.get(model, 'unassigned'), 'metrics': {}}
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
                qs = query_points(base['test_data'])
                entry['diagnostic'] = diagnostic_detail(base['clf'], base['test_data'], qs)
                rl, rp, sat = (entry['diagnostic'][k] for k in ('r2_logit', 'r2_prob',
                                                                'saturation'))
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
