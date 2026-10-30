'''
The aLIMEgn figures in the CLIME extensions (clime:fig:alimegn-*), redrawn in the thesis
style from experiments/alimegn's cached results and helpers.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

import draw
import style

ALIMEGN = os.path.join(os.path.dirname(style.HERE), 'experiments', 'alimegn')
sys.path[:0] = [ALIMEGN, os.path.join(ALIMEGN, 'analysis')]
import clime  # noqa: E402
import common_analysis as ca  # noqa: E402
from common import evaluate, weights as weighting  # noqa: E402

METRIC = 'fidelity (local)'
NORMAL, CIKM = 'bLIMEy (normal)', 'bLIMEy (cost sensitive sampled)'


def schematic(dataset='Gaussian', model='Logistic', index=6, show=2500):
    '''Training sample against test set at one query point, and the weight each scheme gives the sample.'''
    data_params = {'class_samples': [200, 200], 'percent_of_data': 1, 'moons_noise': 0.2,
                   'gaussian_means': [[-1, -1], [1, 1]],
                   'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train, test, clf = evaluate.get_data_and_model(dataset, model, data_params=data_params)
        q = evaluate.query_points(train, test, 'between_class_means', 20)[index]
        rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
        X_s = rng.multivariate_normal(q, np.cov(test['X'].T), evaluate.SURROGATE_SAMPLES)
        yhat = np.asarray(clf.predict(X_s)).astype(np.int64)
        kernel = weighting.kernel_weights(q, X_s)
        panels = [('(b) locality kernel', kernel),
                  (r'(c) $\times$ class weights', kernel * weighting.class_weights_from_labels(yhat)[yhat]),
                  (r'(d) $\times$ density ratio', kernel * weighting.density_ratio_weights(X_s, train['X']))]

    fig, axes = plt.subplots(2, 2, figsize=style.figsize(1.0, 0.82), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = axes.ravel()
    proj = draw.Projection(test['X'])
    lims = draw.limits(test['X'], pad=1.2)
    sub = np.linspace(0, len(X_s) - 1, show).astype(int)

    ax = axes[0]
    draw.surface(ax, clf.predict_proba, proj, lims)
    ax.scatter(X_s[sub, 0], X_s[sub, 1], s=3, color=style.SAMPLE, alpha=0.2, edgecolors='none',
               zorder=1)
    draw.classes(ax, test['X'], test['y'], size=6)
    ax.scatter(*q, s=160, marker='*', color=style.QUERY, edgecolors=style.INK, linewidths=0.6,
               zorder=5)
    ax.set_title('(a) training sample and test set')

    for ax, (title, w) in zip(axes[1:], panels):
        ax.scatter(test['X'][:, 0], test['X'][:, 1], s=4, color=style.MUTED, alpha=0.35,
                   edgecolors='none', zorder=1)
        order = np.argsort(w[sub])
        mesh = ax.scatter(X_s[sub][order, 0], X_s[sub][order, 1], s=3,
                          c=w[sub][order] / w[sub].max(), cmap=style.WEIGHT_CMAP, vmin=0, vmax=1,
                          edgecolors='none', zorder=2)
        ax.set_title(title)
    for ax in axes:
        ax.set(xlim=lims[0], ylim=lims[1], aspect='equal')
    for ax in axes[2:]:
        ax.set_xlabel('Feature 1')
    for ax in axes[::2]:
        ax.set_ylabel('Feature 2')
    bar = fig.colorbar(mesh, ax=axes[[1, 3]], fraction=0.05, pad=0.02)
    bar.set_label('weight, relative to the largest', fontsize=8)
    style.save(fig, 'clime/extensions/fig_schematic')


def marginal(dataset='Breast Cancer', model='Random Forest'):
    '''Standard LIME scored on test data against its own sample, every scheme, and all configurations.'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results, _, _ = ca.load('results_marginal.json')
    entry = results[f'{dataset}|{model}']
    x = np.arange(entry['n_query_points'])
    boundary = ca.boundary_index(entry)

    fig, axes = plt.subplots(1, 3, figsize=style.figsize(1.0, 0.42), constrained_layout=True)
    ax = axes[0]
    for eval_data, colour, label in (('test data', style.SCHEME_COLOURS[NORMAL], 'on test data'),
                                     ('sample locally', style.INK2, 'on its own sample')):
        ax.plot(x, ca.series(entry, NORMAL, METRIC, eval_data), color=colour, marker='o', ms=2.5,
                label=label)
    ax.set(xlabel='Query point', ylabel='Local fidelity', title='(a) standard LIME')
    ax.legend(loc='lower right', fontsize=7)

    ax = axes[1]
    for scheme, colour in style.SCHEME_COLOURS.items():
        y = ca.series(entry, scheme, METRIC, 'test data')
        if not np.all(np.isnan(y)):
            ax.plot(x, y, color=colour, lw=1.2, label=style.SCHEME_LABELS[scheme])
    ax.set(xlabel='Query point', title='(b) every scheme, on test data')
    for a in axes[:2]:
        a.axvline(boundary, color=style.MUTED, ls=':', lw=1)
        a.set_xticks(np.arange(0, len(x), 4))

    ax = axes[2]
    test = np.array([ca.variation(e, NORMAL, METRIC, 'test data') for e in results.values()])
    own = np.array([ca.variation(e, NORMAL, METRIC, 'sample locally') for e in results.values()])
    ok = np.isfinite(test) & np.isfinite(own)
    ax.scatter(own[ok], test[ok], s=8, color=style.SCHEME_COLOURS[NORMAL], alpha=0.8,
               edgecolors='none')
    lim = [0, max(own[ok].max(), test[ok].max()) * 1.05]
    ax.plot(lim, lim, color=style.MUTED, ls='--', lw=1)
    ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel='Variation on own sample',
           ylabel='Variation on test data', title=f'(c) all {int(ok.sum())} configurations')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=7)
    style.save(fig, 'clime/extensions/fig_marginal')


def grid(dataset='Gaussian', models=('Logistic', 'Logistic (label noise 0.3)')):
    '''Standard LIME's fidelity over a grid of query points, and what class weighting adds.'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results, _, _ = ca.load('results_grid.json')
    fig, axes = plt.subplots(len(models), 3, figsize=style.figsize(1.0, 0.27 * len(models)),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    titles = ['standard LIME, on test data', 'gain from class weights,\non test data',
              'gain from class weights,\non its own sample']
    for row, model in zip(axes, models):
        entry = results[f'{dataset}|{model}']
        normal = ca.series(entry, NORMAL, METRIC, 'test data')
        panels = [(normal, None),
                  (ca.series(entry, CIKM, METRIC, 'test data') - normal, True),
                  (ca.series(entry, CIKM, METRIC, 'sample locally')
                   - ca.series(entry, NORMAL, METRIC, 'sample locally'), True)]
        points = np.array(entry['query_points'], dtype=np.float64)
        side = int(round(np.sqrt(len(points))))
        extent = [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()]
        for ax, (values, diverging) in zip(row, panels):
            image = values.reshape(side, side)
            kw = {'cmap': 'viridis'}
            if diverging:
                span = np.nanmax(np.abs(image)) or 1e-6
                kw = {'cmap': 'RdBu_r', 'vmin': -span, 'vmax': span}
            mesh = ax.imshow(image, origin='lower', extent=extent, aspect='equal', **kw)
            bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
            bar.ax.tick_params(labelsize=7)
        row[0].set_ylabel(f'{model}\nFeature 2', fontsize=8)
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Feature 1')
    style.save(fig, f'clime/extensions/fig_grid_{dataset}')


def make():
    schematic()
    marginal()
    grid()


if __name__ == '__main__':
    make()
