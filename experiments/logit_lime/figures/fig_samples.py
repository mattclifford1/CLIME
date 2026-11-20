'''
How many samples each surrogate needs to recover a linear black box's weights
(-> figs/fig_samples.pdf).  Panel (d) reads results_samples.json.

usage:  python figures/fig_samples.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, toy
from common.style import *

import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs

warnings.filterwarnings('ignore')

QUERY_INDEX = 2                      # the query point of fig_threshold.py
FEW = 30                             # samples in each neighbourhood of panels (a) and (b)
SHOWN = 20                           # neighbourhoods drawn there
DRAWS = 100                          # neighbourhoods behind each point of panel (c)
SAMPLES = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
SURROGATES = {'standard': ('bLIMEy (normal)', BLUE, 'standard LIME'),
              'logit': ('bLIMEy (logit)', ORANGE, 'Logit-LIME')}
ARROW = 1.9                          # length of the arrows of (a) and (b), in sd
HALF = 2.4


def explain(label, n, draw):
    e = clime.explainer.AVAILABLE_EXPLAINERS[SURROGATES[label][0]](
        clf, query_point=q, train_data=train, test_data=test, samples=n,
        sample_salt=f'samples figure, draw {draw}')
    return np.asarray(e.get_explanation(), dtype=float)


def angle(coef):
    '''degrees between an explanation and the model's weights'''
    c = coef @ truth/(np.linalg.norm(coef)*np.linalg.norm(truth))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


clf, train, test, qs = toy.build(**toy.LINEAR)
q = qs[QUERY_INDEX]
truth = gradients.grad_logit(clf, toy.LINEAR['model'], q)[0]
u = truth/np.linalg.norm(truth)
K = costs.KERNEL_WIDTH_SCALE*np.sqrt(q.size)

angles = {label: np.array([[angle(explain(label, n, draw)) for draw in range(DRAWS)]
                           for n in SAMPLES]) for label in SURROGATES}
few = {label: [explain(label, FEW, draw) for draw in range(SHOWN)] for label in SURROGATES}

swept = {k: v for k, v in json.load(open(paths.results('results_samples.json'))).items()
         if 'error' not in v}
sizes = sorted(int(n) for n in next(iter(swept.values()))['cos']['standard'])
grid = {label: np.array([[np.nanmean(v['cos'][label][str(n)]) for n in sizes]
                         for v in swept.values()]) for label in SURROGATES}

fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.25]))
gx, gy = np.meshgrid(np.linspace(q[0] - HALF, q[0] + HALF, 200),
                     np.linspace(q[1] - HALF, q[1] + HALF, 200))
surface = clf.predict_proba(np.column_stack([gx.ravel(), gy.ravel()]))[:, 1].reshape(gx.shape)
rng = clime.utils.rng_from_point(q, salt='samples figure, draw 0')
cloud = rng.multivariate_normal(q, np.cov(np.asarray(test['X']).T), FEW)
for ax, (label, (_, colour, name)), letter in zip(axs, SURROGATES.items(), 'ab'):
    ax.contour(gx, gy, surface, levels=[0.5], colors=[INK], linewidths=1.0, zorder=2)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=5, facecolor=MUTED, edgecolor='none', zorder=1)
    ax.add_patch(plt.Circle(q, K*np.sqrt(-2*np.log(0.5)), fill=False, edgecolor=AQUA, lw=0.7,
                            ls=(0, (2, 2)), zorder=3))
    for coef in few[label]:
        tip = q + ARROW*coef/np.linalg.norm(coef)
        ax.plot([q[0], tip[0]], [q[1], tip[1]], color=colour, lw=0.8, alpha=0.7, zorder=4)
    tip = q + 1.12*ARROW*u
    ax.plot([q[0], tip[0]], [q[1], tip[1]], color=INK, lw=1.0, ls=(0, (4, 1.6)), zorder=5)
    ax.scatter([q[0]], [q[1]], s=24, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=6)
    spread = np.median(angles[label][SAMPLES.index(FEW)])
    ax.annotate(f'median {spread:.1f}$^\\circ$ from the\nweights, {DRAWS} draws', xy=(0.04, 0.04),
                xycoords='axes fraction', fontsize=6.4, color=colour, ha='left', va='bottom')
    ax.set_xlim(q[0] - HALF, q[0] + HALF)
    ax.set_ylim(q[1] - HALF, q[1] + HALF)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$x_0$', fontsize=8)
    ax.set_title(f'({letter}) {name},\n{SHOWN} draws of {FEW} samples', fontsize=8)
axs[0].set_ylabel('$x_1$', fontsize=8)

ax = axs[2]
for label, (_, colour, name) in SURROGATES.items():
    low, mid, high = np.quantile(angles[label], [0.1, 0.5, 0.9], axis=1)
    ax.fill_between(SAMPLES, low, high, color=colour, alpha=0.2, lw=0)
    ax.plot(SAMPLES, mid, color=colour, marker='o', ms=2.4, label=name)
ax.axvline(10000, color=MUTED, lw=0.7, ls=(0, (1, 1.4)))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('samples in the neighbourhood', fontsize=7.5)
ax.set_ylabel('degrees from the weights', fontsize=7.5)
ax.set_title('(c) the toy, at $q$', fontsize=8)

ax = axs[3]
for label, (_, colour, name) in SURROGATES.items():
    ax.plot(sizes, grid[label].mean(axis=0), color=colour, marker='o', ms=2.4, label=name)
ax.axvline(10000, color=MUTED, lw=0.7, ls=(0, (1, 1.4)))
ax.set_xscale('log')
ax.set_xlabel('samples in the neighbourhood', fontsize=7.5)
ax.set_ylabel('cosine with the weights', fontsize=7.5)
ax.set_title(f'(d) {len(swept)} linear black boxes', fontsize=8)

fig.tight_layout(w_pad=0.9, rect=(0, 0.08, 1, 1))
handles, labels = axs[2].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 1.6))))
handles.append(plt.Line2D([], [], color=MUTED, lw=0.7, ls=(0, (1, 1.4))))
fig.legend(handles, [*labels, 'the model\'s weights', 'LIME\'s default, 10,000 samples'],
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=7, columnspacing=1.6)
fig.savefig(paths.fig('fig_samples.pdf'))
fig.savefig(paths.fig('fig_samples.png'))

readings = {
    'toy': toy.LINEAR, 'query_index': QUERY_INDEX, 'draws': DRAWS,
    'median_angle': {label: {str(n): float(np.median(a)) for n, a in zip(SAMPLES, angles[label])}
                     for label in SURROGATES},
    'grid_configs': len(swept),
    'grid_datasets': len({k.split('|')[0] for k in swept}),
    'grid_mean_cosine': {label: {str(n): float(c) for n, c in zip(sizes, grid[label].mean(axis=0))}
                         for label in SURROGATES},
    'grid_lowest_at_10000': {label: sorted(
        (float(np.nanmean(v['cos'][label]['10000'])), k) for k, v in swept.items())[:3]
        for label in SURROGATES},
    'grid_above_0.99': {label: {str(n): int((grid[label][:, i] > 0.99).sum())
                                for i, n in enumerate(sizes)} for label in SURROGATES},
}
json.dump(readings, open(paths.results('samples_readings.json'), 'w'), indent=1)
for key in ('median_angle', 'grid_mean_cosine', 'grid_above_0.99', 'grid_lowest_at_10000'):
    print(key, json.dumps(readings[key]))
