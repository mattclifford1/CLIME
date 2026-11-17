'''Explanatory diagrams, drawn from fitted models on a two-Gaussian toy. Writes figures/diagram_*.pdf.'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from thesis_figures import plot, roles
import clime
from clime.data.utils import costs
from clime.models import projection
from common import boxes, style, surrogates
from common.style import EXPLAINER_COLOURS as COLOUR, INK, INK2, MUTED, TEXTWIDTH

QUERY = np.array([-1.7, -1.7])
PANEL = '#f3f2ee'


def toy(seed=0, n=300):
    rng = np.random.default_rng(seed)

    def draw(m):
        X = np.vstack([rng.normal(-1, 1, (m, 2)), rng.normal(1, 1, (m, 2))])
        return {'X': X, 'y': np.r_[np.zeros(m, int), np.ones(m, int)]}

    train, test = draw(n), draw(n)
    clime.RANDOM_SEED = seed
    box = boxes.Cached(projection.projection_model(train, family='Logistic', seed=seed))
    built, lime, balanced = surrogates.build(box, QUERY, test, deltas=())
    return test, box, built, lime, balanced


def _line(ax, surrogate, cut, lims, **kwargs):
    '''where the surrogate's score equals `cut`'''
    xx, yy = np.meshgrid(np.linspace(*lims, 200), np.linspace(*lims, 200))
    g = surrogate.score(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, g, levels=[cut], **kwargs)


def _frame(ax, box, lims, title):
    xx, yy = np.meshgrid(np.linspace(*lims, 200), np.linspace(*lims, 200))
    p = box.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap=roles.PROBA_CMAP, alpha=0.35, zorder=0)
    ax.contour(xx, yy, p, levels=[0.5], colors=INK, linewidths=1.0, zorder=1)
    plot.query(ax, QUERY, size=30)
    ax.set(xlim=lims, ylim=lims, xticks=[], yticks=[], aspect='equal')
    ax.set_title(title, fontsize=8, loc='left')
    ax.grid(False)


def mismatch():
    '''the sample a surrogate is trained on, the data it is scored on, and the three fixes'''
    test, box, built, lime, balanced = toy()
    lims = (-4.6, 3.4)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH*0.36))

    sample = lime.sampled
    shown = np.random.default_rng(0).choice(len(sample['X']), 700, replace=False)
    Xs, ys = sample['X'][shown], np.asarray(sample['y'])[shown]
    ws = costs.weights_based_on_distance(QUERY, sample['X'])
    share = (ws*(np.asarray(sample['y']) == 1)).sum()/ws.sum()
    _frame(axes[0], box, lims, '(a) trained on')
    plot.classes(axes[0], Xs, ys, size=5, alpha=0.45)
    axes[0].text(0.03, 0.03, f'samples around $q$:\n{share:.0%} class 1 by kernel weight',
                 transform=axes[0].transAxes, fontsize=7, color=INK)

    w = costs.weights_based_on_distance(QUERY, test['X'])
    f_class = box.predict(test['X'])
    weighted = (w*(f_class == 1)).sum()/w.sum()
    _frame(axes[1], box, lims, '(b) scored on')
    for label in (0, 1):
        keep = f_class == label
        axes[1].scatter(*test['X'][keep].T, s=3 + 22*w[keep], color=roles.CLASS[label],
                        marker=roles.CLASS_MARKER[label], alpha=0.6, linewidths=0, zorder=2)
    axes[1].text(0.03, 0.03, f'test data near $q$:\n{weighted:.0%} class 1 by kernel weight',
                 transform=axes[1].transAxes, fontsize=7, color=INK)

    _frame(axes[2], box, lims, '(c) surrogate boundaries')
    plot.classes(axes[2], test['X'], f_class, size=4, alpha=0.25)
    drawn = [('LIME', '--'), ('CLIME', '-'), ('LIME + re-threshold', ':')]
    for name, ls in drawn:
        surrogate, cut = built[name]
        _line(axes[2], surrogate, cut, lims, colors=COLOUR[name], linewidths=1.6, linestyles=ls,
              zorder=3)
    axes[2].text(0.03, 0.03, "LIME's score stays below 0.5:\nits boundary is off the plot",
                 transform=axes[2].transAxes, fontsize=7, color=INK)
    handles = [plt.Line2D([], [], color=INK, lw=1.0, label='black box $f$')]
    handles += [plt.Line2D([], [], color=COLOUR[n], lw=1.6, ls=ls, label=style.EXPLAINER_LABELS[n])
                for n, ls in drawn]
    fig.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.07),
               fontsize=7, handlelength=2.2)
    fig.subplots_adjust(wspace=0.05)
    style.save(fig, 'diagram_mismatch')


def slice_():
    '''along the line from the query point across the boundary: the fits and where they cut'''
    test, box, built, lime, balanced = toy()
    direction = np.array([1.0, 1.0])/np.sqrt(2)
    t = np.linspace(-2.2, 5.2, 400)
    X = QUERY + t[:, None]*direction
    f = box.predict_proba(X)[:, 1]
    kernel = costs.weights_based_on_distance(QUERY, X)
    cut = built['LIME + re-threshold'][1]

    def crossing(values, level):
        i = np.flatnonzero(np.diff(np.sign(values - level)))
        return t[i[0]] if len(i) else np.nan

    fig, ax = plt.subplots(figsize=(TEXTWIDTH*0.78, TEXTWIDTH*0.36))
    ax.fill_between(t, -0.22, kernel*0.12 - 0.22, color=MUTED, alpha=0.25, lw=0)
    ax.text(-2.1, -0.15, 'locality kernel', fontsize=6.5, color=INK2, va='center')
    ax.plot(t, f, color=INK, lw=1.4, label='black box $f(x)$')
    g_lime, g_clime = lime.score(X), balanced.score(X)
    ax.plot(t, g_lime, color=COLOUR['LIME'], ls='--', label='LIME fit')
    ax.plot(t, g_clime, color=COLOUR['CLIME'], label='CLIME fit')
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.axhline(cut, color=COLOUR['LIME + re-threshold'], lw=0.9, ls=':')
    ax.text(t[-1], 0.5, ' 0.5', fontsize=7, va='center', color=INK2)
    ax.text(t[-1], cut, ' moved cut', fontsize=7, va='center',
            color=COLOUR['LIME + re-threshold'])
    marks = [(crossing(f, 0.5), 0.5, INK, 'o', "$f$'s boundary", (-6, 8), 'right'),
             (crossing(g_clime, 0.5), 0.5, COLOUR['CLIME'], 's', "CLIME's", (0, 8), 'center'),
             (crossing(g_lime, cut), cut, COLOUR['LIME + re-threshold'], 'D',
              "LIME's, cut moved", (8, -11), 'left')]
    for where, level, colour, marker, text, offset, align in marks:
        if np.isfinite(where):
            ax.plot([where, where], [-0.22, level], color=colour, lw=0.6, alpha=0.7)
            ax.plot([where], [level], marker=marker, color=colour, ms=5, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.5)
            ax.annotate(text, (where, level), xytext=offset, textcoords='offset points',
                        ha=align, fontsize=6.5, color=colour)
    ax.text(t[-1] - 0.05, g_lime[-1] + 0.07, 'LIME reaches 0.5 off the plot', fontsize=6.5,
            color=COLOUR['LIME'], ha='right', va='bottom')
    plot.query(ax, [[0, f[np.argmin(np.abs(t))]]], size=30)
    ax.annotate('$q$', (0, f[np.argmin(np.abs(t))]), xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=8)
    ax.set(xlabel='distance from $q$ towards the other class', ylabel='P(class 1)',
           ylim=(-0.23, 1.08), xlim=(t[0], t[-1]))
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(False)
    style.save(fig, 'diagram_slice')


def _box(ax, x, y, w, h, text, colour=PANEL, size=7, **kwargs):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle='round,pad=0.004,rounding_size=0.012', fc=colour,
                                ec=INK2, lw=0.7))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, **kwargs)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8, lw=0.8,
                                 color=INK, shrinkA=0, shrinkB=0))


def design():
    '''the pipeline, and the two places a class imbalance can be corrected'''
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH*0.50))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    top, w, h = 0.83, 0.158, 0.27
    stages = [
        (0.085, 'data', '22 datasets:\nown ratio,\n110:110, 200:20'),
        (0.262, 'black box $f$', '6 families:\nplain, weighted\nor SMOTE'),
        (0.439, 'threshold of $f$', '0.5, validation\nor 19 deltas\nmethods'),
        (0.616, 'query point $q$', 'line through\nclass means;\ntest points'),
        (0.793, 'surrogate $g$', 'LIME, CLIME,\nLIME with its\ncut moved'),
    ]
    for x, title, text in stages:
        _box(ax, x, top, w, h, '')
        ax.text(x, top + h/2 - 0.045, title, ha='center', va='center', fontsize=7.5, weight='bold')
        ax.text(x, top - 0.035, text, ha='center', va='center', fontsize=6.3, color=INK2)
    for (a, *_), (b, *_) in zip(stages[:-1], stages[1:]):
        _arrow(ax, (a + w/2, top), (b - w/2, top))
    _box(ax, 0.948, top, 0.094, h, '')
    ax.text(0.948, top + h/2 - 0.045, 'score', ha='center', va='center', fontsize=7.5,
            weight='bold')
    ax.text(0.948, top - 0.035, 'cut,\nranking,\ncalibration', ha='center',
            va='center', fontsize=6.3, color=INK2)
    _arrow(ax, (0.793 + w/2, top), (0.948 - 0.047, top))

    ax.text(0.0, 0.60, 'Where the classes are imbalanced, and the two ways to correct it',
            fontsize=7.5, weight='bold', va='center')
    cols = [(0.435, 'reweight or resample\n(the fitted model changes)'),
            (0.810, 'move the threshold\n(the fitted model is kept)')]
    rows = [(0.345, 'the black box,\ntrained on labels $y$'),
            (0.125, 'the surrogate, trained\non black-box labels $f(X_s)$')]
    for x, text in cols:
        ax.text(x, 0.485, text, ha='center', va='center', fontsize=6.5, color=INK2)
    cells = {(0, 0): ('class-weighted loss, SMOTE', '#ece5f7'),
             (0, 1): ('re-thresholding on validation\ndata, deltas', '#e1edfb'),
             (1, 0): ('CLIME: class-balanced\nsample weights $w_{xc}$', '#ece5f7'),
             (1, 1): ("LIME's surrogate with\nits 0.5 cut moved", '#e1edfb')}
    for i, (y, text) in enumerate(rows):
        ax.text(0.0, y, text, ha='left', va='center', fontsize=6.8)
        for j, (x, _) in enumerate(cols):
            label, colour = cells[(i, j)]
            _box(ax, x, y, 0.35, 0.17, label, colour=colour, size=7)
    style.save(fig, 'diagram_design')


def scores():
    '''operating point, ranking and calibration of one surrogate against the black box'''
    test, box, built, lime, balanced = toy()
    w = costs.weights_based_on_distance(QUERY, test['X'])
    f = box.predict_proba(test['X'])[:, 1]
    g = lime.score(test['X'])
    cut = built['LIME + re-threshold'][1]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH*0.62, TEXTWIDTH*0.40))
    for label in (0, 1):
        keep = (f > 0.5) == bool(label)
        ax.scatter(g[keep], f[keep], s=2 + 20*w[keep], color=roles.CLASS[label],
                   marker=roles.CLASS_MARKER[label], alpha=0.55, linewidths=0)
    lo, hi = min(g.min(), 0) - 0.02, max(g.max(), 1) + 0.02
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=':')
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.axvline(0.5, color=COLOUR['LIME'], lw=1.2, ls='--')
    ax.axvline(cut, color=COLOUR['LIME + re-threshold'], lw=1.2, ls=':')
    ax.text(0.5, 1.07, 'cut at 0.5', color=COLOUR['LIME'], fontsize=7, ha='center')
    ax.text(cut, 1.07, 'moved cut', color=COLOUR['LIME + re-threshold'], fontsize=7, ha='center')
    ax.set(xlabel="surrogate's score $g(x)$", ylabel='black box $f(x)$', xlim=(lo, hi),
           ylim=(-0.04, 1.04))
    notes = [(0.86, 'ranking: $g$ orders the points as $f$ does'),
             (0.72, 'operating point: where $g$ is cut'),
             (0.58, 'calibration: distance from the diagonal')]
    for y, text in notes:
        ax.text(0.33, y, text, fontsize=6.8, color=INK2, va='center',
                bbox=dict(fc='white', ec='none', pad=1.5))
    ax.grid(False)
    style.save(fig, 'diagram_scores')


if __name__ == '__main__':
    style.use()
    mismatch()
    slice_()
    design()
    scores()
