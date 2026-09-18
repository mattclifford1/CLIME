'''
Does the result depend on how we chose to evaluate?

The main experiments differ from the evaluation used by clifford2023reconciling (the CIKM
paper this work follows on from) in TWO ways at once, and a reader is entitled to ask
which of them is doing the work:

  metric     they measure fidelity - the locality weighted agreement between the class
             predictions of g and f, both thresholded at 0.5.  We measure a proper
             scoring rule (Brier) on the probabilities.
  eval data  they score the surrogate over the real test set X_test.  We score it over
             points sampled locally around q, from the same distribution the surrogate
             was trained on (but drawn with a different seed).

So this sweeps the 2x2 of those choices over the registered grid.  Their protocol is the
(fidelity, test data) cell; ours is the (Brier, sample locally) cell, which lets the
(Brier, sample locally) numbers be checked against the main sweep as a side effect.

The claim under test is the one Section 2.2 of the paper makes in prose: fidelity cannot
see this effect, because thresholding at 0.5 throws away exactly the calibration that
Logit-LIME changes. If that is right, the fidelity cells should show the two surrogates
as near enough identical while the Brier cells show orders of magnitude.

The extension of the same 2x2 to the five extended-grid black boxes that have gradient
ground truth but were never swept here (--models, writing a separate file) is what lets
the join in analysis/analyse_fidelity_explanation.py be tested on configurations that were
not looked at while the claim was being formed.  It writes its own output file rather than
adding to this one: results_fidelity.json is the registered 168 and stays that way.

usage:  python sweep_fidelity.py <output.json> [--seed N] [--models A,B]
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
from sweeps.sweep import DATASETS, MODELS, GROUP_OF, MODEL_GROUPS, EXPLAINERS, DATA_PARAMS
from sweeps.sweep_extended import NEW_GROUPS

warnings.filterwarnings('ignore')

# the extended-grid black boxes carry their own group labels; 'Bayes Optimal' is
# quadratic by construction, as in sweep_gradient_truth.py
GROUPS = {**GROUP_OF, **NEW_GROUPS, 'Bayes Optimal': 'B quadratic'}

# (metric, evaluation data) - the four cells, named for the table
CELLS = {
    'Brier | local sample': ('Brier score (local)', 'sample locally'),
    'Brier | test data':    ('Brier score (local)', 'test data'),
    'fidelity | local sample': ('fidelity (local)', 'sample locally'),
    'fidelity | test data':    ('fidelity (local)', 'test data'),
}


def opts(dataset, model, explainer, metric, eval_data):
    return {'dataset': dataset, 'data params': DATA_PARAMS, 'standardise data': True,
            'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none',
            'explainer': explainer, 'evaluation metric': metric,
            'evaluation points': 'between_class_means', 'evaluation data': eval_data}


def run(out_path, seed=None, models=None):
    out_path = paths.results(out_path)   # a bare name lands in results/
    models = models or MODELS

    if seed is not None:
        clime.RANDOM_SEED = int(seed)
        np.random.seed(int(seed))

    out = {'_meta': {'seed': seed if seed is not None else clime.RANDOM_SEED,
                     'groups': MODEL_GROUPS, 'models': models,
                     'cells': {k: list(v) for k, v in CELLS.items()}}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} configurations already done', flush=True)

    for dataset in DATASETS:
        for model in models:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            entry = {'group': GROUPS.get(model, 'unassigned'), 'cells': {}}
            try:
                for cell, (metric, eval_data) in CELLS.items():
                    entry['cells'][cell] = {}
                    for expl in EXPLAINERS:
                        r = clime.pipeline.run_pipeline(
                            opts(dataset, model, expl, metric, eval_data),
                            parallel_eval=False)
                        s = np.array(r['score']['scores'])
                        entry['cells'][cell][expl] = {'mean': float(s.mean()),
                                                      'scores': s.tolist()}
            except Exception as e:
                entry['error'] = f'{type(e).__name__}: {e}'
                print(f'{key:60s} FAILED {entry["error"][:60]}', flush=True)
                out[key] = entry
                json.dump(out, open(out_path, 'w'))
                continue

            out[key] = entry
            bits = []
            for cell in CELLS:
                c = entry['cells'][cell]
                std, log = c['bLIMEy (normal)']['mean'], c['bLIMEy (logit)']['mean']
                if cell.startswith('Brier'):
                    bits.append(f'{std/max(log, 1e-30):9.2f}x')
                else:
                    bits.append(f'{std:.3f}/{log:.3f}')
            print(f"{dataset:26s} {model:36s} [{entry['group'][0]}] " + '  '.join(bits),
                  flush=True)
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--models', type=str, default=None,
                   help='comma separated subset of black boxes (default: the registered 12)')
    a = p.parse_args()
    run(a.out, seed=a.seed, models=a.models.split(',') if a.models else None)
