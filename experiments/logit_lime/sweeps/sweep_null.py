'''
The base rate: what does an explainer that explains nothing score?

Every fidelity number in this study is reported without a floor, which makes it impossible
to read.  Is 0.97 good?  Against what?  This sweep supplies the floor.  The null explainer
is the constant

    g(x) = locality weighted mean of f over the neighbourhood

- the constant that minimises the local Brier score, so not a straw man - whose explanation
is the zero vector.  It says nothing about any feature, by construction, and it is scored
by the same four (metric, evaluation data) cells as sweep_fidelity.py so that its readings
sit directly alongside the three fitted surrogates'.

The argument it supports is the off-boundary one.  A threshold instrument asks only which
side of the surrogate's boundary each evaluation point fell on; where the black box predicts
one class over the whole neighbourhood, any surrogate that agrees on that class - including
one with no boundary at all - scores a perfect 1.  How often that happens on this grid is
an empirical question, and this answers it.

usage:  python sweeps/sweep_null.py [results_null.json] [--datasets N]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, surrogates

import argparse
import json
import warnings
import numpy as np
import clime
from sweeps.sweep import opts, DATASETS, MODELS, GROUP_OF, METRICS
from sweeps.sweep_fidelity import CELLS as FIDELITY_CELLS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

# sweep_fidelity.py's 2x2, plus KL: the registered prediction is about which instrument
# ranks the null explainer last, and KL is the one this study actually uses
CELLS = {**FIDELITY_CELLS, 'KL | local sample': ('KL divergence (local)', 'sample locally')}


def run_config(dataset, model, metrics):
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, test_data = r['clf'], r['test_data']
    qs, _ = get_points_between_class_means(test_data)

    entry = {'group': GROUP_OF[model], 'cells': {c: [] for c in CELLS},
             'p_bar': [], 'f_q': []}
    for q in qs:
        q = np.asarray(q, dtype=np.float64)
        expl = surrogates.null_explainer(clf, q, test_data)
        eval_data = {'local': get_local_points(test_data, q), 'test': test_data}
        entry['p_bar'].append(expl.p_bar)
        entry['f_q'].append(float(np.asarray(clf.predict_proba(q[None, :]))[0, 1]))
        for cell, (metric_name, eval_name) in CELLS.items():
            data = eval_data['local' if eval_name == 'sample locally' else 'test']
            entry['cells'][cell].append(float(metrics[metric_name](
                expl, black_box_model=clf, data=data, query_point=q)))
    for cell in CELLS:
        entry[f'mean {cell}'] = float(np.mean(entry['cells'][cell]))
    return entry


def run(out_path, datasets):
    out_path = paths.results(out_path)
    metrics = {m: clime.evaluation.AVAILABLE_EVALUATION_METRICS[m]
               for m, _ in CELLS.values()}

    out = {'_meta': {'seed': clime.RANDOM_SEED,
                     'cells': {k: list(v) for k, v in CELLS.items()}}}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} configurations already done', flush=True)

    for dataset in datasets:
        for model in MODELS:
            key = f'{dataset}|{model}'
            if key in out:
                continue
            try:
                entry = run_config(dataset, model, metrics)
                print(f"{dataset:26s} {model:36s} "
                      f"fid(local)={entry['mean fidelity | local sample']:.4f} "
                      f"fid(test)={entry['mean fidelity | test data']:.4f} "
                      f"Brier={entry['mean Brier | local sample']:.4f}", flush=True)
            except Exception as e:
                entry = {'error': f'{type(e).__name__}: {e}'}
                print(f'{key:60s} FAILED {entry["error"][:60]}', flush=True)
            out[key] = entry
            json.dump(out, open(out_path, 'w'))

    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', nargs='?', default='results_null.json')
    p.add_argument('--datasets', type=int, default=len(DATASETS))
    a = p.parse_args()
    run(a.out, DATASETS[:a.datasets])
