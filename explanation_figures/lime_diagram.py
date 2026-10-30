'''How LIME builds a surrogate g around a query point q (fig:literature-review:lime).'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.datasets import make_moons
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

import draw
import style

NAME = '02-literature-review/LIME'
WIDTH = 1.0
SEED = 0
Q = np.array([0.5, 0.25])
N_SAMPLES, SPREAD, KERNEL = 40, 0.45, 0.5
MARKER = {0: 's', 1: 'o'}


def make():
    X, y = make_moons(400, noise=0.2, random_state=SEED)
    f = SVC(gamma=2, probability=True, random_state=SEED).fit(X, y)
    rng = np.random.default_rng(SEED)
    X_s = rng.normal(Q, SPREAD, size=(N_SAMPLES, 2))
    w = np.exp(-np.sum((X_s - Q)**2, axis=1) / KERNEL**2)
    fx = f.predict(X_s)
    g = Ridge(alpha=1.0).fit(X_s, f.predict_proba(X_s)[:, 1], sample_weight=w)

    fig, (key, ax) = plt.subplots(1, 2, figsize=style.figsize(WIDTH, 0.38),
                                  gridspec_kw={'width_ratios': [1, 1.6]})
    proj = draw.Projection(X)
    lims = ((-1.4, 2.4), (-0.9, 1.4))
    draw.surface(ax, f.predict_proba, proj, lims, boundary=False)
    for c in (0, 1):
        m = fx == c
        ax.scatter(X_s[m, 0], X_s[m, 1], s=8 + 70 * w[m], marker=MARKER[c],
                   color=style.CLASS[c], edgecolors=style.INK, linewidths=0.6, zorder=2)
    # g's 0.5 level set: coef . x + intercept = 0.5
    xs = np.array(lims[0])
    ax.plot(xs, (0.5 - g.intercept_ - g.coef_[0] * xs) / g.coef_[1], color=style.INK,
            ls='--', lw=1.6, zorder=3)
    ax.scatter(*Q, s=110, color=style.QUERY, edgecolors=style.INK, linewidths=1.0, zorder=4)
    ax.set(xlim=lims[0], ylim=lims[1], xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(True)
    ax.text(0.97, 0.04, r'$g \approx f$ near $q$', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=9, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))

    def cls(c):
        return Line2D([], [], ls='', marker=MARKER[c], ms=6, color=style.CLASS[c],
                      markeredgecolor=style.INK)
    handles = [(Patch(color=style.CLASS[0], alpha=style.SURFACE_ALPHA),
                Patch(color=style.CLASS[1], alpha=style.SURFACE_ALPHA)),
               Line2D([], [], color=style.INK, ls='--', lw=1.6),
               Line2D([], [], ls='', marker='o', ms=9, color=style.QUERY,
                      markeredgecolor=style.INK),
               (cls(0), cls(1))]
    labels = [r'$f$  (black box)', r'$g$  (surrogate model)', r'$q$  (query point)',
              r'$X_s$  (sampled data)']
    key.axis('off')
    key.legend(handles, labels, loc='center', handler_map={tuple: HandlerTuple(ndivide=None)},
               handlelength=3, labelspacing=1.1, fontsize=9, title='Key')
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
