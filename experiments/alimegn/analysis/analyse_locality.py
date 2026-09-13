'''
is "local" fidelity on the test set actually local?

An observation that came out of the curves rather than out of a prediction. LIME's kernel
width is k = 0.75*sqrt(D), which grows with the number of features, while standardised data
does not spread out at the same rate. On a wide dataset the exponential kernel is then so
broad that every test point carries comparable weight, and a "local" score computed over
the test set is close to a global one.

This measures it. For each dataset, at every query point on the line, it records the
effective number of test points the locality kernel actually weights,

    ESS = (sum w)^2 / sum w^2,

which is n when the weights are uniform and 1 when a single point dominates. ESS/n near 1
means the metric is global.

It matters for this study because P1 contrasts scoring on the test set with scoring on a
local sample - and if the first is really a global score, then what P1 measures is partly
local-versus-global rather than one marginal versus another. It matters for CIKM'23 for the
same reason.

usage:  uv run python analysis/analyse_locality.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import spearmanr

import common_analysis as ca
from common import evaluate
from clime.data.utils import costs


def locality(dataset, data_params):
    train_data, test_data, _ = evaluate.get_data_and_model(
        dataset, 'Logistic', data_params=data_params)
    points = evaluate.query_points(train_data, test_data, 'between_class_means', 20)
    n = test_data['X'].shape[0]
    d = test_data['X'].shape[1]
    ess, nearest = [], []
    for query_point in points:
        w = costs.weights_based_on_distance(query_point, test_data['X'])
        ess.append((w.sum()**2)/np.sum(w**2))
        distances = np.linalg.norm(test_data['X'] - query_point, axis=1)
        nearest.append(np.median(distances))
    kernel_width = np.sqrt(d)*costs.KERNEL_WIDTH_SCALE
    return {'features': d, 'test points': n, 'kernel width': kernel_width,
            'median distance': float(np.mean(nearest)),
            'ESS': float(np.mean(ess)), 'ESS fraction': float(np.mean(ess))/n}


if __name__ == '__main__':
    results, meta, _ = ca.load('results_marginal.json')
    data_params = meta.get('data_params')
    datasets = sorted({e['dataset'] for e in results.values()})

    print('how local is a locality-weighted score over the test set?\n')
    print(f"{'dataset':26s} {'D':>4s} {'n test':>7s} {'kernel k':>9s} "
          f"{'median dist':>12s} {'ESS':>8s} {'ESS/n':>7s}")
    rows = {}
    for dataset in datasets:
        try:
            row = locality(dataset, data_params)
        except Exception as e:                     # noqa: BLE001
            print(f'{dataset:26s} failed: {type(e).__name__}: {e}')
            continue
        rows[dataset] = row
        print(f"{dataset:26s} {row['features']:4d} {row['test points']:7d} "
              f"{row['kernel width']:9.2f} {row['median distance']:12.2f} "
              f"{row['ESS']:8.1f} {row['ESS fraction']:7.2f}")

    # does it explain how big the marginal effect looks?
    fraction, variation = [], []
    for key, entry in results.items():
        row = rows.get(entry['dataset'])
        if row is None:
            continue
        v = ca.variation(entry, 'bLIMEy (normal)', 'fidelity (local)', 'test data')
        if np.isfinite(v):
            fraction.append(row['ESS fraction'])
            variation.append(v)
    rho = spearmanr(fraction, variation)
    print(f'\nESS/n vs the size of the collapse on test data: rho = {rho.statistic:+.3f}, '
          f'p = {rho.pvalue:.2g}, n = {len(fraction)}')
    print(f'median ESS/n over datasets: {np.median([r["ESS fraction"] for r in rows.values()]):.2f}')
