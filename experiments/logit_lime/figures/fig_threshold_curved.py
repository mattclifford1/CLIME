'''
A model with curved log-odds explained at three decision thresholds
(-> figs/fig_threshold_curved.pdf).

usage:  python figures/fig_threshold_curved.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, threshold, toy
from common.style import *

import json
import warnings
import numpy as np
from scipy.special import expit
import clime
from clime.data.utils import costs

warnings.filterwarnings('ignore')

QUERY_INDEX = 7                      # f(q) = 0.001 as trained
THRESHOLDS = (0.99, 0.5, 0.01)
SWEEP = expit(-np.linspace(-7, 7, 29))   # thresholds from 0.999 to 0.001
SURROGATES = {'standard': ('bLIMEy (normal)', BLUE, 'standard LIME'),
              'logit': ('bLIMEy (logit)', ORANGE, 'Logit-LIME')}
WASH = '#f2f1ed'
HALF = 2.6
ARROW = 1.25                         # length of the direction arrows, in sd


def explain(box):
    return {label: np.asarray(clime.explainer.AVAILABLE_EXPLAINERS[name](
                box, query_point=q, train_data=train, test_data=test).get_explanation(), dtype=float)
            for label, (name, _, _) in SURROGATES.items()}


def share(coef):
    '''x1's share of the explanation'''
    return float(abs(coef[1])/np.abs(coef).sum())


def cosine(a, b):
    return float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b)))


clf, train, test, qs = toy.build(**toy.CURVED)
q = qs[QUERY_INDEX]
K = costs.KERNEL_WIDTH_SCALE*np.sqrt(q.size)
truth = gradients.grad_logit(clf, toy.CURVED['model'], q)[0]
boxes = {t: threshold.at_threshold(clf, t) for t in THRESHOLDS}
fits = {t: explain(box) for t, box in boxes.items()}
f_q = {t: float(box.predict_proba(q[None, :])[0, 1]) for t, box in boxes.items()}
swept = [explain(threshold.at_threshold(clf, t)) for t in SWEEP]
shares = {label: np.array([share(c[label]) for c in swept]) for label in SURROGATES}

X, y = np.asarray(test['X']), np.asarray(test['y'])
gx, gy = np.meshgrid(np.linspace(q[0] - HALF, q[0] + HALF, 300),
                     np.linspace(q[1] - HALF, q[1] + HALF, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])
top = {'standard': 1.3*max(np.abs(fits[t]['standard']).max() for t in THRESHOLDS),
       'logit': 1.3*np.abs(truth).max()}

fig = plt.figure(figsize=(7.4, 4.3))
spec = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], width_ratios=[1, 1, 1, 1.15])
for column, t in enumerate(THRESHOLDS):
    ax = fig.add_subplot(spec[0, column])
    z = boxes[t].log_odds(grid).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=[0, np.inf], colors=[WASH], zorder=0)
    ax.contour(gx, gy, boxes[0.5].log_odds(grid).reshape(gx.shape), levels=[0], colors=[MUTED],
               linewidths=0.8, linestyles=[(0, (3, 2))], zorder=2)
    ax.contour(gx, gy, z, levels=[0], colors=[INK], linewidths=1.3, zorder=3)
    for cls, marker, face in ((0, 'o', 'white'), (1, 's', MUTED)):
        ax.scatter(X[y == cls, 0], X[y == cls, 1], s=5, marker=marker, facecolor=face,
                   edgecolor=MUTED, linewidth=0.4, zorder=1)
    ax.add_patch(plt.Circle(q, K*np.sqrt(-2*np.log(0.5)), fill=False, edgecolor=AQUA, lw=0.7,
                            ls=(0, (2, 2)), zorder=4))
    tip = q + 1.15*ARROW*truth/np.linalg.norm(truth)
    ax.plot([q[0], tip[0]], [q[1], tip[1]], color=INK, lw=1.0, ls=(0, (4, 1.6)), zorder=5)
    for label, (_, colour, _) in SURROGATES.items():
        coef = fits[t][label]
        ax.annotate('', xy=q + ARROW*coef/np.linalg.norm(coef), xytext=q, zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.3, shrinkA=0, shrinkB=0))
    ax.scatter([q[0]], [q[1]], s=28, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=7)
    ax.set_xlim(q[0] - HALF, q[0] + HALF)
    ax.set_ylim(q[1] - HALF, q[1] + HALF)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('$x_0$', fontsize=8)
    if column == 0:
        ax.set_ylabel('$x_1$', fontsize=8)
    ax.set_title(f'threshold {t}' + (' (as trained)' if t == 0.5 else ''), fontsize=8.5)

    ax = fig.add_subplot(spec[1, column])
    right = ax.twinx()
    for label, axis, offset in (('standard', ax, 0.0), ('logit', right, 2.6)):
        coef, colour = fits[t][label], SURROGATES[label][1]
        axis.bar(offset + np.arange(2), coef, 0.72, color=colour, zorder=3)
        for j, value in enumerate(coef):
            above = max(value, truth[j]) if label == 'logit' else value
            axis.annotate(f'{value:.2f}', xy=(offset + j, above), xytext=(0, 2),
                          textcoords='offset points', fontsize=6.4, color=colour, ha='center',
                          fontweight='bold' if j == int(np.argmax(np.abs(coef))) else 'normal')
        axis.set_ylim(0, top[label])
        axis.tick_params(axis='y', colors=colour, labelsize=6.5)
    right.hlines(truth, 2.6 + np.arange(2) - 0.42, 2.6 + np.arange(2) + 0.42, color=INK, lw=1.4,
                 zorder=5)
    right.grid(False)
    right.spines['top'].set_visible(False)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(['$x_0$', '$x_1$']*2)
    ax.grid(axis='x', visible=False)
    for at, label in ((0.5, 'standard'), (3.1, 'logit')):
        ax.annotate(SURROGATES[label][2].replace('standard ', ''), xy=(at, 0), xytext=(0, -22),
                    textcoords='offset points', fontsize=7, color=SURROGATES[label][1], ha='center', va='top',
                    annotation_clip=False)
    if column == 0:
        ax.set_ylabel('probability per sd', fontsize=7.5, color=BLUE)
    if column == len(THRESHOLDS) - 1:
        right.set_ylabel('log-odds per sd', fontsize=7.5, color=ORANGE)

ax = fig.add_subplot(spec[:, 3])
ax.axhline(0.5, color=MUTED, lw=0.7)
ax.annotate('$x_1$ ranked first', xy=(0.04, 0.5), xycoords=('axes fraction', 'data'), xytext=(0, 3),
            textcoords='offset points', fontsize=6.4, color=INK_2, ha='left')
ax.annotate('$x_0$ ranked first', xy=(0.04, 0.5), xycoords=('axes fraction', 'data'), xytext=(0, -3),
            textcoords='offset points', fontsize=6.4, color=INK_2, ha='left', va='top')
for label, (_, colour, name) in SURROGATES.items():
    ax.plot(SWEEP, shares[label], color=colour, marker='o', ms=2.2, label=name)
ax.axhline(share(truth), color=INK, lw=1.0, ls=(0, (4, 1.6)), label='the model\'s gradient at $q$')
for t in THRESHOLDS:
    ax.axvline(t, color=MUTED, lw=0.6, ls=(0, (1, 1.4)))
ax.set_xscale('logit')
ax.set_xlim(SWEEP[0], SWEEP[-1])       # the threshold falls from left to right
ax.set_xticks([0.99, 0.5, 0.01])
ax.set_xticklabels(['0.99', '0.5', '0.01'], fontsize=7)
ax.minorticks_off()
ax.set_ylim(0.2, 0.8)
ax.set_xlabel('decision threshold', fontsize=8)
ax.set_ylabel('share of the explanation on $x_1$', fontsize=8)
ax.set_title('every threshold', fontsize=8.5)

fig.tight_layout(w_pad=0.8, h_pad=1.0, rect=(0, 0.06, 1, 1))
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7,
           columnspacing=1.8)
fig.savefig(paths.fig('fig_threshold_curved.pdf'))
fig.savefig(paths.fig('fig_threshold_curved.png'))

flips = SWEEP[np.where(np.diff(np.sign(shares['standard'] - 0.5)) != 0)[0]]
readings = {
    'toy': toy.CURVED, 'query_index': QUERY_INDEX, 'gradient': truth.tolist(),
    'gradient_share': share(truth),
    'f_q': {str(t): f_q[t] for t in THRESHOLDS},
    'coef': {label: {str(t): fits[t][label].tolist() for t in THRESHOLDS} for label in SURROGATES},
    'share': {label: {str(t): share(fits[t][label]) for t in THRESHOLDS} for label in SURROGATES},
    'cosine_with_gradient': {label: {str(t): cosine(fits[t][label], truth) for t in THRESHOLDS}
                             for label in SURROGATES},
    'share_over_thresholds': {label: [float(shares[label].min()), float(shares[label].max())]
                              for label in SURROGATES},
    'standard_ranking_flips_below_threshold': [float(t) for t in flips],
}
json.dump(readings, open(paths.results('threshold_curved_readings.json'), 'w'), indent=1)
for key, value in readings.items():
    print(key, json.dumps(value))
