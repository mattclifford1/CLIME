'''
The soft-label logistic surrogate on the grid: Logit-LIME's model class, fitted by
cross-entropy against the black box's probabilities (sixth registration,
PREREGISTRATION.md).

Only the new surrogate is fitted here. Standard LIME, Logit-LIME and the hard-label
surrogate are already on disk for exactly the same query points and evaluation samples,
because sampling is seeded per query point (FINDINGS.md B10):

    Brier, KL on the local sample     results_full.json (results_taxonomy.json on the
                                      registered grid; the two agree to 1e-11)
    the CIKM'23 fidelity cells        results_fidelity_full.json
    explanation against the truth     results_gradient_truth_full.json

so the analysis joins against those rather than refitting them. The join is checked rather
than assumed: at every configuration's first query point this sweep also refits standard
LIME and records its local Brier score, which must equal the stored one exactly.

Per query point, for the surrogate at C = 0.5 (the registered fit, the nominal equivalent
of the other surrogates' ridge alpha = 1):

    Brier | local sample, KL | local sample          the main protocol
    Brier | test data, fidelity | local sample,
    fidelity | test data                            the rest of sweep_fidelity.py's 2x2
    cos, rho, top1                                  against grad logit f(q), where it exists
    converged, constant                             whether lbfgs converged, and whether the
                                                    neighbourhood had one class only

and Brier and KL again at C = 50, the nominal equivalent of alpha = 0.01, without a
prediction.

usage:  python sweeps/sweep_soft_logistic.py [out.json] [--datasets A,B] [--models A,B]
        (the full grid runs one dataset per process: sweeps/run_full.py soft_logistic)
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
from clime.evaluation.key_points import get_local_points
from sweeps import sweep, full_grid

warnings.filterwarnings('ignore')

SOFT = 'bLIMEy (soft-label logistic regression)'
C_WEAK = 50.0
EVAL = clime.evaluation.AVAILABLE_EVALUATION_METRICS
CELLS = {'Brier | local sample': ('Brier score (local)', 'local'),
         'KL | local sample': ('KL divergence (local)', 'local'),
         'Brier | test data': ('Brier score (local)', 'test'),
         'fidelity | local sample': ('fidelity (local)', 'local'),
         'fidelity | test data': ('fidelity (local)', 'test')}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else float('nan')


def score(expl, clf, data, q):
    return {cell: float(EVAL[m](expl, black_box_model=clf, data=data[where], query_point=q))
            for cell, (m, where) in CELLS.items()}


def refit(expl, clf, C):
    '''the same surrogate at another penalty, on the neighbourhood it was built from'''
    sampled = expl._sample_locally(clf)
    w = expl._get_sampled_weights(sampled)
    expl.surrogate_model.C = C
    expl.surrogate_model.fit(sampled['X'], sampled['p(y|x)'], sample_weight=w)
    return expl


def run_config(dataset, model):
    r = clime.pipeline.run_pipeline(sweep.opts(dataset, model, 'bLIMEy (normal)',
                                               sweep.METRICS[0]), parallel_eval=False)
    clf, train_data, test_data = r['clf'], r['train_data'], r['test_data']
    qs = np.asarray(sweep.query_points(test_data), dtype=np.float64)
    differentiable = gradients.has_gradient(model)
    truth = gradients.grad_logit(clf, model, qs) if differentiable else None

    entry = {'group': full_grid.GROUP_OF.get(model, 'unassigned'),
             'n_features': int(qs.shape[1]), 'differentiable': bool(differentiable),
             'points': []}
    for i, q in enumerate(qs):
        data = {'local': get_local_points(test_data, q), 'test': test_data}
        expl = clime.explainer.AVAILABLE_EXPLAINERS[SOFT](
            clf, query_point=q, train_data=train_data, test_data=test_data)
        m = expl.surrogate_model
        row = {'index': i, **score(expl, clf, data, q),
               'constant': m.constant is not None,
               'converged': bool(m.constant is not None or m.n_iter_[0] < m.max_iter),
               'norm': float(np.linalg.norm(expl.get_explanation()))}
        if differentiable and np.linalg.norm(truth[i]) > 0 and np.all(np.isfinite(truth[i])):
            c = np.asarray(expl.get_explanation(), dtype=float)
            rho = spearmanr(np.abs(c), np.abs(truth[i]))[0] if c.size > 1 else np.nan
            # an all-zero explanation (the one-class fallback) has no top feature: argmax
            # would return feature 0 and score it, so it is NaN like cos and rho
            row.update(cos=cosine(c, truth[i]),
                       rho=float(rho) if np.isfinite(rho) else float('nan'),
                       top1=(float(np.argmax(np.abs(c)) == np.argmax(np.abs(truth[i])))
                             if np.linalg.norm(c) > 0 else float('nan')))
        weak = refit(expl, clf, C_WEAK)
        row[f'Brier | local sample | C={C_WEAK:g}'] = float(EVAL['Brier score (local)'](
            weak, black_box_model=clf, data=data['local'], query_point=q))
        row[f'KL | local sample | C={C_WEAK:g}'] = float(EVAL['KL divergence (local)'](
            weak, black_box_model=clf, data=data['local'], query_point=q))
        if i == 0:
            # the join check: standard LIME refitted here must reproduce the stored score
            std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
                clf, query_point=q, train_data=train_data, test_data=test_data)
            entry['check_standard_brier_q0'] = float(EVAL['Brier score (local)'](
                std, black_box_model=clf, data=data['local'], query_point=q))
        entry['points'].append(row)

    for k in sorted({k for pt in entry['points'] for k in pt} - {'index'}):
        vals = np.array([p.get(k, np.nan) for p in entry['points']], dtype=float)
        entry[f'mean {k}'] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
    return entry


def run(out_path, datasets, models):
    out_path = paths.results(out_path)
    out = {'_meta': {'surrogate': SOFT, 'C': 0.5, 'C_weak': C_WEAK,
                     'query points': sweep.QUERY_POINTS, 'cells': list(CELLS)}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} already done', flush=True)
    for dataset in datasets:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                e = run_config(dataset, model)
                print(f"{dataset:28s} {model:36s} Brier={e['mean Brier | local sample']:.3e} "
                      f"KL={e['mean KL | local sample']:.3e} "
                      f"cos={e.get('mean cos', float('nan')):+.3f} "
                      f"unconverged={sum(not p['converged'] for p in e['points'])}",
                      flush=True)
            except Exception as exc:
                e = {'error': f'{type(exc).__name__}: {exc}'}
                print(f'{key:66s} FAILED {e["error"][:60]}', flush=True)
            out[key] = e
            json.dump(out, open(out_path, 'w'))
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_soft_logistic_full.json')
    p.add_argument('--datasets', type=str, default=None)
    p.add_argument('--models', type=str, default=None)
    a = p.parse_args()
    full_grid.configure()
    run(a.out, a.datasets.split(',') if a.datasets else full_grid.DATASETS,
        a.models.split(',') if a.models else full_grid.FIDELITY_MODELS)
