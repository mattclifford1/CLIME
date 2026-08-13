'''
Figure: what the experiment actually looks like.

The setup is described in prose - 20 query points along the line between class means,
10,000 points drawn per neighbourhood, weighted by an exponential kernel - and that is a
lot to hold at once. This grounds it on the two-dimensional Gaussian data, where the whole
arrangement is visible at once.

(a) the black box's probability surface, the training data, and where the query points sit
(b) one query point's neighbourhood: the sample the surrogate is actually fitted to, with
    the locality weights that decide how much each point counts

Deliberately not shown: the surrogate's own fit. Figure 1 already does that, as a transect
in probability and logit space, which is where the difference between the two surrogates
is legible.

usage:  python fig_setup.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

QUERY_INDEX = 11          # just off the decision boundary, as used by fig_mechanism
N_SHOWN = 1500            # subsample of the 10,000 drawn, so the file stays small

train, test = clime.data.AVAILABLE_DATASETS['Gaussian'](
    class_samples=[200, 200], gaussian_means=[[-1, -1], [1, 1]],
    gaussian_covs=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
norm = clime.data.normaliser(train)
train, test = norm(train), norm(test)
clf = clime.models.AVAILABLE_MODELS['Logistic'](data=train)

qs, _ = get_points_between_class_means(test)
qs = np.asarray(qs)
q = qs[QUERY_INDEX]

X = np.asarray(test['X'])
y = np.asarray(test['y'])

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1))

# ---- (a) the black box, the data, and the query points -------------------------------
ax = axs[0]
pad = 1.0
xs = np.linspace(X[:, 0].min()-pad, X[:, 0].max()+pad, 300)
ys = np.linspace(X[:, 1].min()-pad, X[:, 1].max()+pad, 300)
gx, gy = np.meshgrid(xs, ys)
probs = clf.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
# A light greyscale wash plus labelled contours, rather than a saturated fill: the class
# colours have to stay readable on top, and a diverging blue/red map fights them directly.
ax.contourf(gx, gy, probs, levels=np.linspace(0, 1, 21), cmap='Greys', alpha=0.13,
            zorder=0)
ax.contour(gx, gy, probs, levels=[0.5], colors=[INK], linewidths=1.0, zorder=2)
ax.annotate('$p = 0.5$', xy=(2.15, -2.75), fontsize=6.5, color=INK_2, rotation=-45)

for cls, colour, marker in [(0, BLUE, 'o'), (1, ORANGE, 's')]:
    ax.scatter(X[y == cls, 0], X[y == cls, 1], s=9, facecolor=colour, edgecolor='white',
               linewidth=0.3, alpha=0.65, zorder=3, label=f'class {cls}')

ax.plot(qs[:, 0], qs[:, 1], color=INK, lw=0.8, ls='-', zorder=4)
ax.scatter(qs[:, 0], qs[:, 1], s=17, facecolor='white', edgecolor=INK, linewidth=0.9,
           zorder=5, label='query points')
ax.scatter([q[0]], [q[1]], s=46, facecolor=AQUA, edgecolor=INK, linewidth=0.9, zorder=6)
ax.annotate('shown in (b)', xy=(q[0], q[1]), xytext=(26, -30), textcoords='offset points',
            fontsize=6.5, color=INK, ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none',
                      alpha=0.85),
            arrowprops=dict(arrowstyle='-', lw=0.6, color=INK_2))
ax.set_title('(a)  black box, data, query points', fontsize=8.5, color=INK, pad=6)
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.legend(loc='upper left', fontsize=6.5, handletextpad=0.3, borderpad=0.25,
          labelspacing=0.25, scatterpoints=1)

# ---- (b) one neighbourhood, as the surrogate sees it ---------------------------------
ax = axs[1]
rng = clime.utils.rng_from_point(q, salt='figure: setup illustration')
cov = np.cov(X.T)
sample = rng.multivariate_normal(q, cov, 10000)
weights = costs.weights_based_on_distance(q, sample)
keep = rng.choice(len(sample), N_SHOWN, replace=False)
sample, weights = sample[keep], weights[keep]

order = np.argsort(weights)          # heaviest points drawn last, so they stay visible
scat = ax.scatter(sample[order, 0], sample[order, 1], c=weights[order], s=8,
                  cmap='viridis', linewidth=0, alpha=0.85, zorder=2)
ax.contour(gx, gy, probs, levels=[0.5], colors=[INK], linewidths=1.0, zorder=3)
ax.scatter([q[0]], [q[1]], s=52, facecolor=AQUA, edgecolor=INK, linewidth=1.0, zorder=5)

cbar = fig.colorbar(scat, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('locality weight $w_x$', fontsize=7)
cbar.ax.tick_params(labelsize=6.5)
cbar.outline.set_visible(False)

ax.set_title('(b)  the neighbourhood fitted at one query point', fontsize=8.5, color=INK,
             pad=6)
ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_xlim(axs[0].get_xlim())
ax.set_ylim(axs[0].get_ylim())

fig.tight_layout(w_pad=1.4)
os.makedirs('figs', exist_ok=True)
fig.savefig(paths.fig('fig_setup.pdf'))
fig.savefig(paths.fig('fig_setup.png'))
print('fig_setup written')
