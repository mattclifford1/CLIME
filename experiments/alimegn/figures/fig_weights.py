'''
P5: is CIKM's class weighting a density-ratio correction in disguise?

The class trick assigns one of two weights to every sampled point, by the class the black
box gives it. An estimated density ratio assigns a continuous weight to each point. If the
first is a crude version of the second, the two should agree in rank - and they should
agree *more* where the neighbourhood is one-sided, which is where the class trick has
something to correct.

(a) and (b) the two corrections point by point, at a boundary query point and at a tail
    query point of one configuration
(c) the rank agreement along the line, pooled over every configuration in the sweep

Panels (a) and (b) recompute the sampled neighbourhood - the sweep stores the rank
correlation, not 10,000 weights per query point - using the same salt the surrogates use,
so these are the points a surrogate would have been trained on.

usage:  uv run python figures/fig_weights.py [dataset] [model]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import clime
from common import paths, style, evaluate, weights as weighting
import common_analysis as ca

DEFAULT = ('Breast Cancer', 'Random Forest')


def _corrections(clf, train_data, test_data, query_point):
    '''the two weightings, on the surrogate's own sample'''
    rng = clime.utils.rng_from_point(query_point, salt='surrogate training sample')
    cov = np.cov(test_data['X'].T)
    X_s = rng.multivariate_normal(query_point, cov, evaluate.SURROGATE_SAMPLES)
    yhat = np.asarray(clf.predict(X_s)).astype(np.int64)
    cikm = weighting.class_weights_from_labels(yhat)[yhat]
    ratio = weighting.density_ratio_weights(X_s, train_data['X'])
    return cikm, ratio, yhat


def panel_points(ax, cikm, ratio, yhat, title):
    subsample = np.linspace(0, len(cikm)-1, min(len(cikm), 3000)).astype(int)
    for cls, colour in ((0, style.BLUE), (1, style.ORANGE)):
        mask = yhat[subsample] == cls
        ax.scatter(ratio[subsample][mask], cikm[subsample][mask], s=4, alpha=0.35,
                   color=colour, edgecolor='none', label=f'$\\hat{{y}} = {cls}$')
    rho = spearmanr(cikm, ratio).statistic
    ax.set_xscale('log')
    ax.set_xlabel('estimated density ratio')
    ax.set_ylabel('CIKM class weight')
    ax.set_title(f'{title}\n' + (r'$\rho = %+.2f$' % rho if np.isfinite(rho)
                                 else 'constant correction'))
    ax.legend(loc='upper left', fontsize=7, markerscale=2)


def panel_along_line(ax, results):
    '''rank agreement as a function of position along the line, over every configuration'''
    curves = []
    for entry in results.values():
        row = entry.get('diagnostics', {}).get('weight agreement')
        if not row:
            continue
        curves.append([np.nan if v is None else v for v in row])
    if not curves:
        ax.set_visible(False)
        return
    curves = np.array(curves, dtype=np.float64)
    x = np.arange(curves.shape[1])
    median = np.nanmedian(curves, axis=0)
    lo = np.nanpercentile(curves, 25, axis=0)
    hi = np.nanpercentile(curves, 75, axis=0)
    ax.fill_between(x, lo, hi, color=style.PLUM, alpha=0.18, lw=0)
    ax.plot(x, median, color=style.PLUM, marker='o', ms=3)
    ax.axhline(0, color=style.MUTED, lw=1, ls='--')
    ax.set_xlabel('query point (class 0 mean $\\to$ class 1 mean)')
    ax.set_ylabel('rank agreement of the two corrections')
    ax.set_title(f'(c) {curves.shape[0]} configurations (IQR shaded)')


def main(dataset, model):
    results, _, _ = ca.load('results_marginal.json')
    key = f'{dataset}|{model}'
    if key not in results:
        raise SystemExit(f'{key} not in results')
    entry = results[key]
    train_data, test_data, clf = evaluate.get_data_and_model(
        dataset, model, data_params=ca.load('results_marginal.json')[1].get('data_params'))
    points = [np.asarray(q) for q in entry['query_points']]

    boundary = ca.boundary_index(entry)
    tail = 0 if boundary > entry['n_query_points']//2 else entry['n_query_points']-1

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, index, label in ((axes[0], boundary, '(a) at the decision boundary'),
                             (axes[1], tail, '(b) in the tail')):
        cikm, ratio, yhat = _corrections(clf, train_data, test_data, points[index])
        panel_points(ax, cikm, ratio, yhat, f'{label} (point {index})')
    panel_along_line(axes[2], results)
    fig.suptitle(f'{dataset} — {model}', fontsize=9, y=1.02)
    out = paths.fig('fig_weights.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    print('written', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0] if args else DEFAULT[0], args[1] if len(args) > 1 else DEFAULT[1])
