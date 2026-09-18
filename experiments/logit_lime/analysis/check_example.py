'''
Is the worked example a property of the configuration, or of one random draw?

select_example.py picks the point to draw; this asks whether it survives being drawn
again.  A single query point of a single fitted model is exactly the sort of evidence that
looks compelling and replicates at chance, so before the figure is believed the same
configuration is rebuilt under five random seeds - a different train/test split and a
different fitted black box each time - and at three locality kernel widths.

The shape being checked is the one the example is chosen to show (PREREGISTRATION.md):
the two fidelity cells cannot separate the surrogates, while the explanation ground truth
separates them by a wide margin.

Note that the query point is an INDEX into the between-class-means line, not a fixed
location: under a new seed the split changes, so the class means move and point 12 is a
different point of a different model.  That is the right test - a shape that only exists
for one particular fitted model is not worth a figure - and it is why the numbers move as
much as they do.

usage:  python analysis/check_example.py [--dataset D] [--model M] [--index I]
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
from clime.data.utils import costs
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASET, MODEL, INDEX = 'Breast Cancer', 'MLP', 12
SEEDS = [1, 2, 3, 4, 42]                       # as sweep_seeds.py
KERNEL_SCALES = [0.5, 0.75, 1.25]              # the default, and one either side
SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)',
              'logreg': 'bLIMEy (logistic regression)'}
CELLS = {'fidelity | local sample': ('fidelity (local)', 'local'),
         'fidelity | test data': ('fidelity (local)', 'test'),
         'Brier': ('Brier score (local)', 'local'),
         'KL': ('KL divergence (local)', 'local')}

FID_TIE_TOL = 0.01      # fidelity readings this close cannot separate the surrogates
COS_MARGIN = 0.4        # the explanations, however, are this far apart
BRIER_MARGIN = 10.0     # while a proper scoring rule separates them by this much
COS_MATERIAL = 0.1      # "the explanations differ at all", as opposed to by a lot


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else float('nan')


def measure(dataset, model, index):
    '''every instrument and the explanation truth at one query point of one fitted model'''
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train, test = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test)
    q = np.asarray(qs[index], dtype=float)
    truth = gradients.grad_logit(clf, model, q[None, :])[0]
    data = {'local': get_local_points(test, q), 'test': test}
    metrics = clime.evaluation.AVAILABLE_EVALUATION_METRICS

    out = {'f_q': float(np.asarray(clf.predict_proba(q[None, :]))[0, 1]),
           'truth_norm': float(np.linalg.norm(truth))}
    for label, name in SURROGATES.items():
        expl = clime.explainer.AVAILABLE_EXPLAINERS[name](
            clf, q, train_data=train, test_data=test)
        row = {'cos': cosine(np.asarray(expl.get_explanation(), dtype=float), truth)}
        for cell, (metric, where) in CELLS.items():
            row[cell] = float(metrics[metric](expl, black_box_model=clf,
                                              data=data[where], query_point=q))
        out[label] = row
    return out


def holds(m):
    '''
    the shape, as three separate criteria.

    Kept separate deliberately.  The conjunction is what the selection protocol asks for,
    but it can fail because one component moved slightly across an arbitrary threshold
    while the qualitative picture is unchanged, and a caption that says "holds in 1 of 5
    seeds" when two of the three components hold in all five would be as misleading as one
    that says it always holds.  Both numbers are reported.
    '''
    d_cos = m['logit']['cos'] - m['standard']['cos']
    ratio = m['standard']['Brier']/max(m['logit']['Brier'], 1e-30)
    parts = {
        'fidelity tied': all(abs(m['standard'][c] - m['logit'][c]) <= FID_TIE_TOL
                             for c in ('fidelity | local sample', 'fidelity | test data')),
        'Brier separates': bool(np.isfinite(ratio) and ratio >= BRIER_MARGIN),
        'explanations differ': bool(np.isfinite(d_cos) and d_cos > COS_MATERIAL),
        'by a wide margin': bool(np.isfinite(d_cos) and d_cos > COS_MARGIN),
    }
    return parts, d_cos, ratio


def shapes(parts):
    '''
    the three shapes a candidate can hold, nested from strongest to weakest.

    A candidate selected as case 1 will usually fail `case 3` for a reason that is itself
    the finding - the black box is saturated, so Brier cannot separate the surrogates
    either - so reporting only the case-3 verdict for it would read as "does not
    replicate" when what is happening is "replicates, and shows something else".
    '''
    return {
        'case 3, strict': all(parts.values()),
        'case 3': (parts['fidelity tied'] and parts['Brier separates']
                   and parts['explanations differ']),
        'case 1': parts['fidelity tied'] and parts['by a wide margin'],
    }


def show(tag, m):
    parts, d_cos, ratio = holds(m)
    got = shapes(parts)
    strict, loose = got['case 3, strict'], got['case 3']
    marks = ''.join('.+'[parts[k]] for k in
                    ('fidelity tied', 'Brier separates', 'explanations differ',
                     'by a wide margin'))
    print(f"{tag:<16s} f(q)={m['f_q']:.4f}  "
          f"fid(test) {m['standard']['fidelity | test data']:.4f}/"
          f"{m['logit']['fidelity | test data']:.4f}  "
          f"fid(local) {m['standard']['fidelity | local sample']:.4f}/"
          f"{m['logit']['fidelity | local sample']:.4f}  "
          f"Brier {ratio:>9.1f}x  "
          f"cos {m['standard']['cos']:+.3f}/{m['logit']['cos']:+.3f} "
          f"(d={d_cos:+.3f})  [{marks}] "
          + ', '.join(k for k, v in got.items() if v))
    return parts, strict, loose, got


def main(dataset=DATASET, model=MODEL, index=INDEX):
    print(f'worked example: {dataset} | {model}, query point {index}\n')
    results = {'_meta': {'dataset': dataset, 'model': model, 'index': index,
                         'seeds': SEEDS, 'kernel_scales': KERNEL_SCALES,
                         'fid_tie_tol': FID_TIE_TOL, 'cos_margin': COS_MARGIN}}

    print('  key: [fidelity tied | Brier separates | explanations differ | '
          'by a wide margin]\n')
    print('SEEDS  (a new split and a newly fitted black box each time)')
    base_seed = clime.RANDOM_SEED
    tallies = {'strict': 0, 'shape': 0}
    parts_count, shape_count = {}, {}
    d_cos_seen = []
    results['seeds'] = {}
    for s in SEEDS:
        clime.RANDOM_SEED = s
        np.random.seed(s)
        clime.pipeline.run_pipeline.cache_clear()     # opts does not carry the seed
        m = measure(dataset, model, index)
        results['seeds'][str(s)] = m
        parts, strict, loose, got = show(f'  seed {s}', m)
        tallies['strict'] += strict
        tallies['shape'] += loose
        for k, v in got.items():
            shape_count[k] = shape_count.get(k, 0) + v
        d_cos_seen.append(m['logit']['cos'] - m['standard']['cos'])
        for k, v in parts.items():
            parts_count[k] = parts_count.get(k, 0) + v
    clime.RANDOM_SEED = base_seed
    np.random.seed(base_seed)
    clime.pipeline.run_pipeline.cache_clear()

    n = len(SEEDS)
    for k, v in shape_count.items():
        flag = '' if v >= 4 else '   <-- below the 4/5 the protocol requires'
        print(f'\n  {k:<16s} holds in {v}/{n} seeds{flag}' if k == 'case 3, strict'
              else f'  {k:<16s} holds in {v}/{n} seeds{flag}')
    print('  by criterion:')
    for k, v in parts_count.items():
        print(f'    {k:<22s} {v}/{n}')
    print(f'  the explanation gap ranges over {min(d_cos_seen):+.3f} to '
          f'{max(d_cos_seen):+.3f} across seeds')

    print('\nKERNEL WIDTH  (seed 42, the default split)')
    base_scale, kernel_ok, kernel_shape = costs.KERNEL_WIDTH_SCALE, 0, 0
    results['kernel'] = {}
    for scale in KERNEL_SCALES:
        costs.KERNEL_WIDTH_SCALE = scale
        clime.pipeline.run_pipeline.cache_clear()     # nor the kernel width
        m = measure(dataset, model, index)
        results['kernel'][str(scale)] = m
        _, strict, loose, _ = show(f'  scale {scale}', m)
        kernel_ok += strict
        kernel_shape += loose
    costs.KERNEL_WIDTH_SCALE = base_scale
    clime.pipeline.run_pipeline.cache_clear()
    print(f'\n  all four criteria: {kernel_ok}/{len(KERNEL_SCALES)} kernel widths; '
          f'without the wide-margin clause {kernel_shape}/{len(KERNEL_SCALES)}')

    results['_meta'].update(
        {'seeds_strict': tallies['strict'], 'seeds_shape': tallies['shape'],
         'seeds_by_criterion': parts_count, 'seeds_by_shape': shape_count,
         'd_cos_range': [float(min(d_cos_seen)), float(max(d_cos_seen))],
         'kernels_strict': kernel_ok, 'kernels_shape': kernel_shape})
    out = paths.results(f"example_robustness_{model.replace(chr(32),chr(95))}_{index}.json")
    json.dump(results, open(out, 'w'), indent=1)
    print(f'\nwritten {out}')
    return tallies, kernel_ok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=DATASET)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--index', type=int, default=INDEX)
    a = p.parse_args()
    main(a.dataset, a.model, a.index)
