'''
One logistic regression explained at three decision thresholds
(-> figs/fig_threshold_idea.pdf, fig_threshold_toy.pdf, fig_threshold_sweep.pdf).

usage:  python figures/fig_threshold.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, threshold
from common.style import *

import json
import warnings
import numpy as np
from scipy.special import expit, logit
import clime
from clime.data.utils import costs
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

DATASET, MODEL = 'Gaussian', 'Logistic'
QUERY_INDEX = 4                      # a confident negative: f(q) = 0.0003 as trained
THRESHOLDS = (0.5, 0.1, 0.01)
STYLES = {0.5: '-', 0.1: (0, (4, 1.6)), 0.01: (0, (1, 1.4))}
SWEEP = expit(-np.linspace(-7, 11, 37))   # thresholds from 0.999 to 0.00002
SURROGATES = {'standard': 'bLIMEy (normal)', 'logit': 'bLIMEy (logit)'}
WASH = '#f2f1ed'


def explain(box, q):
    '''coefficients and value at q of each surrogate (a probability, and a log-odds)'''
    out = {}
    for label, name in SURROGATES.items():
        e = clime.explainer.AVAILABLE_EXPLAINERS[name](box, query_point=q, train_data=train,
                                                       test_data=test)
        m = e.surrogate_model
        at_q = (m.predict(q[None, :])[0, 1] if label == 'standard'
                else float(m.intercept_) + q @ np.atleast_2d(m.coef_)[-1, :])
        out[label] = (np.asarray(e.get_explanation(), dtype=float), float(at_q))
    return out


def cosine(a, b):
    return float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b)))


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, SURROGATES['standard'], METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
qs = np.asarray(get_points_between_class_means(test)[0], dtype=float)
q = qs[QUERY_INDEX]
K = costs.KERNEL_WIDTH_SCALE*np.sqrt(q.size)
truth = gradients.grad_logit(clf, MODEL, q)[0]
u = truth/np.linalg.norm(truth)
boxes = {t: threshold.at_threshold(clf, t) for t in THRESHOLDS}
fits = {t: explain(box, q) for t, box in boxes.items()}
f_q = {t: float(box.predict_proba(q[None, :])[0, 1]) for t, box in boxes.items()}
s_q = float(threshold.Shifted(clf, 0).log_odds(q[None, :])[0]/np.linalg.norm(truth))

norms = {label: np.array([[np.linalg.norm(explain(threshold.at_threshold(clf, t), point)[label][0])
                           for t in SWEEP] for point in qs]) for label in SURROGATES}
mid = int(np.argmin(np.abs(SWEEP - 0.5)))
fold = {label: n/n[:, [mid]] for label, n in norms.items()}


def along(s):
    '''points at signed distance s from the trained boundary, on the line through q along the weights'''
    return q[None, :] + np.outer(np.asarray(s) - s_q, u)


# ------------------------------------------------------------ figure 1: the idea, in one dimension
s = np.linspace(-4.2, 1.6, 400)
near = np.linspace(s_q - 1.5, s_q + 1.5, 50)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.75))

ax = axs[0]
moved = 0.1
cross = float(logit(moved)/np.linalg.norm(truth))
ax.plot(s, boxes[0.5].predict_proba(along(s))[:, 1], color=INK, lw=1.8)
ax.plot(s, boxes[moved].predict_proba(along(s))[:, 1], color=INK, lw=1.8, ls=STYLES[moved])
for level, at in ((0.5, 0.0), (moved, cross)):
    ax.axhline(level, color=MUTED, lw=0.6)
    ax.annotate(f'$t={level}$', xy=(1.55, level), xytext=(0, 2), textcoords='offset points',
                fontsize=6.5, color=INK_2, ha='right')
ax.plot([cross, cross], [moved, 0.5], color=AQUA, lw=1.2)
ax.scatter([cross, cross], [moved, 0.5], s=16, facecolor='white', edgecolor=AQUA, linewidth=1.1,
           zorder=5)
ax.scatter([0], [0.5], s=16, facecolor='white', edgecolor=INK, linewidth=1.1, zorder=5)
ax.annotate('the trained model\ncut at 0.1 ...', xy=(cross, moved), xytext=(-2.0, 0.25),
            fontsize=6.4, color=INK_2, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkB=4))
ax.annotate('... is this model\ncut at 0.5', xy=(cross, 0.5), xytext=(-2.4, 0.78),
            fontsize=6.4, color=INK_2, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkB=4))
ax.set_ylabel('probability of class 1')
ax.set_title('(a) a threshold move shifts the curve', fontsize=8.5)

ax = axs[1]
weight = costs.weights_based_on_distance(q, along(s))
ax.fill_between(s, -0.22, -0.22 + 0.11*weight/weight.max(), color='#dfe8f4', lw=0)
ax.annotate('locality weight', xy=(s_q + 1.25, -0.215), fontsize=6.2, color='#5b7fa8', ha='left',
            va='bottom')
for t in THRESHOLDS:
    coef, at_q = fits[t]['standard']
    ax.plot(s, boxes[t].predict_proba(along(s))[:, 1], color=INK, lw=1.2, ls=STYLES[t])
    line = at_q + (coef @ u)*(near - s_q)
    ax.plot(near, line, color=BLUE, lw=1.8, ls=STYLES[t])
    ax.annotate(f'{coef @ u:.2f}', xy=(near[-1], line[-1]), xytext=(3, 0),
                textcoords='offset points', fontsize=6.6, color=BLUE, va='center')
ax.annotate('slope of\nLIME\'s fit,\nper sd', xy=(0.04, 0.72), xycoords='axes fraction',
            fontsize=6.4, color=BLUE, ha='left', va='center')
ax.set_ylim(-0.22, 1.04)
ax.set_title('(b) probability: three slopes', fontsize=8.5)

ax = axs[2]
for t in THRESHOLDS:
    coef, at_q = fits[t]['logit']
    ax.plot(near, at_q + (coef @ u)*(near - s_q), color=ORANGE, lw=4.0, alpha=0.55,
            solid_capstyle='butt')
    ax.plot(s, boxes[t].log_odds(along(s)), color=INK, lw=1.2, ls=STYLES[t])
ax.annotate(f'intercepts {threshold.shift_for(0.1):.1f} and\n{threshold.shift_for(0.01):.1f} above the first',
            xy=(0.04, 0.95), xycoords='axes fraction', fontsize=6.4, color=INK_2, ha='left',
            va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.0), zorder=8)
ax.annotate(f'slope of Logit-LIME\'s fit:\n{fits[0.5]["logit"][0] @ u:.2f}, '
            f'{fits[0.1]["logit"][0] @ u:.2f} and {fits[0.01]["logit"][0] @ u:.2f}',
            xy=(0.97, 0.06), xycoords='axes fraction', fontsize=6.4, color=ORANGE, ha='right',
            va='bottom')
ax.set_ylabel('log-odds of class 1')
ax.set_title('(c) log-odds: one slope', fontsize=8.5)

for ax in axs:
    ax.set_xlabel('distance from the trained boundary (sd)', fontsize=7.5)
    ax.set_xlim(s[0], s[-1])
for ax in axs[1:]:
    ax.axvline(s_q, color=AQUA, lw=0.9, zorder=1)
axs[1].annotate('$q$', xy=(s_q, 1.0), xytext=(3, 0), textcoords='offset points', fontsize=8,
                color=INK, va='top')

fig.tight_layout(w_pad=1.2, rect=(0, 0.09, 1, 1))
handles = [plt.Line2D([], [], color=INK, lw=1.2, ls=STYLES[t], label=f'black box at threshold {t}')
           for t in THRESHOLDS]
handles += [plt.Line2D([], [], color=BLUE, lw=1.8, label='standard LIME'),
            plt.Line2D([], [], color=ORANGE, lw=4.0, alpha=0.55, label='Logit-LIME')]
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02),
           handlelength=2.2, columnspacing=1.2, fontsize=7)
fig.savefig(paths.fig('fig_threshold_idea.pdf'))
fig.savefig(paths.fig('fig_threshold_idea.png'))

# ------------------------------------------------------------ figure 2: the model and what the user is shown
X, y = np.asarray(test['X']), np.asarray(test['y'])
HALF = 3.6
gx, gy = np.meshgrid(np.linspace(-HALF, HALF, 300), np.linspace(-HALF, HALF, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])
names = ['$x_0$', '$x_1$']
top = {'standard': 1.25*max(np.abs(fits[t]['standard'][0]).max() for t in THRESHOLDS),
       'logit': 1.25*np.abs(truth).max()}

fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.5), gridspec_kw=dict(height_ratios=[1.55, 1]))
for column, t in enumerate(THRESHOLDS):
    ax = axs[0, column]
    z = boxes[t].log_odds(grid).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=[0, np.inf], colors=[WASH], zorder=0)
    ax.contour(gx, gy, boxes[0.5].log_odds(grid).reshape(gx.shape), levels=[0], colors=[MUTED],
               linewidths=0.8, linestyles=[(0, (3, 2))], zorder=2)
    ax.contour(gx, gy, z, levels=[0], colors=[INK], linewidths=1.3, zorder=3)
    for cls, marker, face in ((0, 'o', 'white'), (1, 's', MUTED)):
        ax.scatter(X[y == cls, 0], X[y == cls, 1], s=5, marker=marker, facecolor=face,
                   edgecolor=MUTED, linewidth=0.4, zorder=1)
    for level in (0.5, 0.1):
        ax.add_patch(plt.Circle(q, K*np.sqrt(-2*np.log(level)), fill=False, edgecolor=AQUA,
                                lw=0.7, ls=(0, (2, 2)), zorder=4))
    ax.scatter([q[0]], [q[1]], s=30, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.annotate(f'$f(q) = {f_q[t]:.4f}$' if f_q[t] < 0.01 else f'$f(q) = {f_q[t]:.3f}$',
                xy=q, xytext=(0, -26), textcoords='offset points', fontsize=6.6, color=INK,
                ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0),
                zorder=7)
    ax.annotate('predicted\nclass 1', xy=(0.96, 0.95), xycoords='axes fraction', fontsize=6.6,
                color=INK_2, ha='right', va='top')
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('$x_0$', fontsize=8)
    ax.set_title(f'threshold {t}' + (' (as trained)' if t == 0.5 else ''), fontsize=8.5)

    ax = axs[1, column]
    right = ax.twinx()
    for label, axis, colour, offset in (('standard', ax, BLUE, 0.0), ('logit', right, ORANGE, 2.6)):
        coef = fits[t][label][0]
        axis.bar(offset + np.arange(2), coef, 0.72, color=colour, zorder=3)
        for j, value in enumerate(coef):
            axis.annotate(f'{value:.3f}' if label == 'standard' else f'{value:.2f}',
                          xy=(offset + j, value), xytext=(0, 2), textcoords='offset points',
                          fontsize=6.4, color=colour, ha='center')
        axis.set_ylim(0, top[label])
        axis.tick_params(axis='y', colors=colour, labelsize=6.5)
    right.hlines(truth, 2.6 + np.arange(2) - 0.42, 2.6 + np.arange(2) + 0.42, color=INK, lw=1.4,
                 zorder=5)
    right.grid(False)
    right.spines['top'].set_visible(False)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(names*2)
    ax.grid(axis='x', visible=False)
    for at, text, colour in ((0.5, 'standard LIME', BLUE), (3.1, 'Logit-LIME', ORANGE)):
        ax.annotate(text, xy=(at, 0), xytext=(0, -22), textcoords='offset points', fontsize=7,
                    color=colour, ha='center', va='top', annotation_clip=False)
    if column == 0:
        ax.legend(handles=[plt.Line2D([], [], color=INK, lw=1.4, label='the model\'s weights')],
                  loc='upper left', fontsize=6.4, handlelength=1.2, borderpad=0.1)
        ax.set_ylabel('probability per sd', fontsize=7.5, color=BLUE)
        axs[0, column].set_ylabel('$x_1$', fontsize=8)
    if column == len(THRESHOLDS) - 1:
        right.set_ylabel('log-odds per sd', fontsize=7.5, color=ORANGE)
fig.tight_layout(w_pad=1.0, h_pad=1.2)
fig.savefig(paths.fig('fig_threshold_toy.pdf'))
fig.savefig(paths.fig('fig_threshold_toy.png'))

# ------------------------------------------------------------ figure 3: every threshold, every query point
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax = axs[0]
ax.plot(SWEEP, norms['standard'][QUERY_INDEX], color=BLUE, marker='o', ms=2.4)
ax.axvline(f_q[0.5], color=AQUA, lw=0.9)
ax.annotate('boundary\nthrough $q$', xy=(f_q[0.5], 0.03), xytext=(-4, 0), textcoords='offset points',
            fontsize=6.4, color=INK_2, va='bottom', ha='right')
ax.set_ylabel('probability per sd', fontsize=8, color=BLUE)
ax.tick_params(axis='y', colors=BLUE)
ax.set_ylim(0, 0.5)
right = ax.twinx()
right.plot(SWEEP, norms['logit'][QUERY_INDEX], color=ORANGE, marker='o', ms=2.4)
right.axhline(np.linalg.norm(truth), color=INK, lw=1.2, ls=(0, (4, 1.6)))
right.set_ylabel('log-odds per sd', fontsize=8, color=ORANGE)
right.tick_params(axis='y', colors=ORANGE)
right.set_ylim(0, 5.5)
right.grid(False)
right.spines['top'].set_visible(False)
right.annotate('Logit-LIME, on the truth', xy=(0.5, np.linalg.norm(truth)), xytext=(0, 4),
               textcoords='offset points', fontsize=6.6, color=ORANGE, ha='center')
ax.annotate('standard LIME', xy=(0.9, 0.06), fontsize=6.6, color=BLUE, ha='center')
ax.set_title('(a) size of the explanation at $q$', fontsize=8.5)

ax = axs[1]
for i in range(len(qs)):
    ax.plot(SWEEP, fold['standard'][i], color=BLUE, lw=0.7, alpha=0.55)
    ax.plot(SWEEP, fold['logit'][i], color=ORANGE, lw=0.7, alpha=0.55)
ax.plot(SWEEP, fold['standard'][QUERY_INDEX], color=BLUE, lw=1.8)
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e4)
ax.set_ylabel('size relative to threshold 0.5', fontsize=8)
ax.annotate('standard LIME,\n20 query points', xy=(0.04, 0.95), xycoords='axes fraction',
            fontsize=6.6, color=BLUE, ha='left', va='top')
ax.annotate('Logit-LIME', xy=(0.5, 1), xytext=(0, 8), textcoords='offset points', fontsize=6.6,
            color=ORANGE, ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.0), zorder=8)
ax.set_title('(b) every query point', fontsize=8.5)

for ax in axs:
    ax.set_xscale('logit')
    ax.set_xlim(SWEEP[0], SWEEP[-1])       # the threshold falls from left to right
    ax.set_xticks([0.999, 0.99, 0.9, 0.5, 0.1, 0.01, 0.001, 0.0001])
    ax.set_xticklabels(['0.999', '0.99', '0.9', '0.5', '0.1', '0.01', '0.001', '0.0001'], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel('decision threshold', fontsize=8)
fig.tight_layout(w_pad=1.6)
fig.savefig(paths.fig('fig_threshold_sweep.pdf'))
fig.savefig(paths.fig('fig_threshold_sweep.png'))

# ------------------------------------------------------------ the numbers the text quotes
at = {t: int(np.argmin(np.abs(SWEEP - t))) for t in THRESHOLDS}
readings = {
    'dataset': DATASET, 'model': MODEL, 'query_index': QUERY_INDEX, 'truth': truth.tolist(),
    'distance_to_boundary': {str(t): abs(s_q - float(logit(t)/np.linalg.norm(truth)))
                             for t in THRESHOLDS},
    'kernel_width': K,
    'f_q': {str(t): f_q[t] for t in THRESHOLDS},
    'coef': {label: {str(t): fits[t][label][0].tolist() for t in THRESHOLDS} for label in SURROGATES},
    'norm': {label: {str(t): float(np.linalg.norm(fits[t][label][0])) for t in THRESHOLDS}
             for label in SURROGATES},
    'cosine_with_truth': {label: {str(t): cosine(fits[t][label][0], truth) for t in THRESHOLDS}
                          for label in SURROGATES},
    'standard_claims_at_q': {str(t): fits[t]['standard'][1] for t in THRESHOLDS},
    'fold_at_q': {label: {str(t): float(fold[label][QUERY_INDEX, at[t]]) for t in THRESHOLDS}
                  for label in SURROGATES},
    'fold_over_points': {label: {str(t): [float(fold[label][:, at[t]].min()),
                                          float(fold[label][:, at[t]].max())]
                                 for t in THRESHOLDS} for label in SURROGATES},
    'span_over_thresholds_at_q': {label: float(norms[label][QUERY_INDEX].max()
                                               / norms[label][QUERY_INDEX].min())
                                  for label in SURROGATES},
}
json.dump(readings, open(paths.results('threshold_readings.json'), 'w'), indent=1)
for key in ('f_q', 'norm', 'fold_at_q', 'fold_over_points', 'span_over_thresholds_at_q',
            'cosine_with_truth', 'standard_claims_at_q', 'distance_to_boundary'):
    print(key, json.dumps(readings[key]))
