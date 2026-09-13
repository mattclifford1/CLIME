'''
the whole study in one picture: two distributions, and what each scheme does about them

(a) the setup. The surrogate is trained on a Gaussian cloud around the query point and
    judged against the test set, which lies somewhere else. Away from the boundary the
    black box predicts one class over almost all of that cloud.
(b-d) the same 10,000 sampled points, coloured by the weight each scheme gives them, with
    the test data overlaid so that where the weight goes can be compared against where
    the data are. The locality kernel alone; CIKM's class weights, which take one of two
    values according to the class the black box assigns; and the estimated density ratio.

usage:  uv run python figures/fig_schematic.py [dataset] [model] [query point index]
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

import clime
from common import paths, style, evaluate, weights as weighting

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}
DEFAULT = ('Gaussian', 'Logistic', 6)
SHOW = 2500        # sampled points drawn, of the 10,000 used


def _sample(clf, test_data, query_point):
    rng = clime.utils.rng_from_point(query_point, salt='surrogate training sample')
    cov = np.cov(test_data['X'].T)
    X_s = rng.multivariate_normal(query_point, cov, evaluate.SURROGATE_SAMPLES)
    return X_s, np.asarray(clf.predict(X_s)).astype(np.int64)


def _limits(test_data, pad=1.2):
    X = test_data['X']
    return ((X[:, 0].min()-pad, X[:, 0].max()+pad),
            (X[:, 1].min()-pad, X[:, 1].max()+pad))


def panel_setup(ax, clf, test_data, query_point, X_s, limits):
    (x0, x1), (y0, y1) = limits
    grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    probs = clf.predict_proba(np.c_[grid_x.ravel(), grid_y.ravel()])[:, 1]
    ax.contourf(grid_x, grid_y, probs.reshape(grid_x.shape), levels=12, cmap='Greys',
                alpha=0.3)
    ax.contour(grid_x, grid_y, probs.reshape(grid_x.shape), levels=[0.5],
               colors=[style.INK], linewidths=1.2)

    sub = np.linspace(0, len(X_s)-1, SHOW).astype(int)
    ax.scatter(X_s[sub, 0], X_s[sub, 1], s=4, color=style.PLUM, alpha=0.18,
               edgecolor='none', label='surrogate\'s training sample')
    for cls, colour in ((0, style.BLUE), (1, style.ORANGE)):
        mask = np.asarray(test_data['y']) == cls
        ax.scatter(test_data['X'][mask, 0], test_data['X'][mask, 1], s=9, color=colour,
                   alpha=0.9, edgecolor='none', label=f'test data, class {cls}')
    ax.scatter(*query_point, s=110, marker='*', color=style.AQUA,
               edgecolor=style.INK, linewidth=0.7, zorder=5, label='query point $q$')
    ax.set_title('(a) trained on one distribution,\njudged against another')
    legend = ax.legend(loc='lower left', fontsize=5.8, markerscale=1.4, frameon=True,
                       borderpad=0.3, handletextpad=0.4, labelspacing=0.3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(style.MUTED)
    legend.get_frame().set_alpha(0.9)
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')


def panel_weights(ax, X_s, w, test_data, title, subtitle):
    # the data, faint, so that where the weight goes can be read against where the data are
    ax.scatter(test_data['X'][:, 0], test_data['X'][:, 1], s=7, color=style.MUTED,
               alpha=0.35, edgecolor='none', zorder=1)
    sub = np.linspace(0, len(X_s)-1, SHOW).astype(int)
    order = np.argsort(w[sub])           # heaviest points on top
    mesh = ax.scatter(X_s[sub][order, 0], X_s[sub][order, 1], s=5,
                      c=w[sub][order]/w[sub].max(), cmap='magma_r', vmin=0, vmax=1,
                      edgecolor='none', zorder=2)
    ax.set_title(f'{title}\n{subtitle}')
    ax.set_xlabel('feature 1')
    return mesh


def main(dataset, model, index):
    train_data, test_data, clf = evaluate.get_data_and_model(
        dataset, model, data_params=DATA_PARAMS)
    points = evaluate.query_points(train_data, test_data, 'between_class_means', 20)
    query_point = points[index]
    X_s, yhat_s = _sample(clf, test_data, query_point)
    limits = _limits(test_data)

    kernel = weighting.kernel_weights(query_point, X_s)
    class_weights = weighting.class_weights_from_labels(yhat_s)[yhat_s]
    ratio = weighting.density_ratio_weights(X_s, train_data['X'])
    majority = max((yhat_s == 0).mean(), (yhat_s == 1).mean())

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    panel_setup(axes[0], clf, test_data, query_point, X_s, limits)
    panel_weights(axes[1], X_s, kernel, test_data, '(b) locality kernel',
                  'standard LIME')
    panel_weights(axes[2], X_s, kernel*class_weights, test_data,
                  '(c) $\\times$ class weights',
                  f'CIKM\'23; {majority:.0%} of the sample is one class')
    mesh = panel_weights(axes[3], X_s, kernel*ratio, test_data,
                         '(d) $\\times$ density ratio',
                         'pulls weight back towards the data')
    bar = fig.colorbar(mesh, ax=axes[1:], fraction=0.03, pad=0.01)
    bar.set_label('weight, relative to the largest in the panel', fontsize=7)
    for ax in axes:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    out = paths.fig('fig_schematic.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    print('written', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0] if args else DEFAULT[0],
         args[1] if len(args) > 1 else DEFAULT[1],
         int(args[2]) if len(args) > 2 else DEFAULT[2])
