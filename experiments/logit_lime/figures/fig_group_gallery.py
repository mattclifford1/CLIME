'''
One black box per registered group, and what each surrogate can do with it
(-> figs/fig_group_gallery.pdf).

Figure 1 makes the mechanism argument on four black boxes chosen to separate smoothness
from linearity. This one is the taxonomy: a representative of every group A-E, in the
order they were registered, so a reader can see the geometry the grouping is about and
what it costs the surrogate - and, in the bottom row, whether a local ground truth for
the explanation exists at all.

That last row is the point of the figure. Groups A, B and C have a gradient, so the true
local importances are defined and a surrogate's explanation can be scored. Groups D and E
do not: the log-odds are a step function, flat almost everywhere and undefined on the
splits, so there is nothing for a linear explanation to be a good approximation of. The
usual reading of LIME - a local linear approximation of the black box - has no referent
there, which is a stronger statement than "LIME does badly on trees".

Moons rather than the Gaussian data of Figure 1: two dimensions so the geometry is
visible, but a genuinely nonlinear problem, so a black box's group is not handed to it by
the data.

usage:  python figures/fig_group_gallery.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients
from common.style import *

import warnings
import numpy as np
import clime
from matplotlib.colors import LinearSegmentedColormap
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASET = 'Moons'
QUERY_INDEX = 10
T_LIM = 2.2
WASH = LinearSegmentedColormap.from_list('wash', ['#ffffff', '#dbdad6'])

# one per registered group, in registration order
COLUMNS = [('A', 'exactly linear', 'Logistic', 'Logistic regression'),
           ('B', 'quadratic', 'QDA', 'QDA'),
           ('C', 'smooth', 'MLP', 'Neural network'),
           ('D', 'piecewise constant', 'Decision Tree', 'Decision tree'),
           ('E', 'calibrated forest', 'Random Forest (Platt calibrated)',
            'Forest + Platt')]


def logit(p, lim=9):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.clip(np.log(p/(1 - p)), -lim, lim)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


def fmt_ratio(v):
    if v >= 1000:
        e = int(np.floor(np.log10(v)))
        return f'${v/10**e:.1f}\\!\\times\\!10^{{{e}}}$'
    if v >= 100:
        return f'${v:.0f}\\times$'
    return f'${v:.2f}\\times$'


brier = clime.evaluation.AVAILABLE_EVALUATION_METRICS['Brier score (local)']

fig, axs = plt.subplots(3, len(COLUMNS), figsize=(7.1, 4.9), sharex='col',
                        gridspec_kw=dict(height_ratios=[0.62, 1, 1]))

summary = []
for col, (letter, geometry, model, nice) in enumerate(COLUMNS):
    r = clime.pipeline.run_pipeline(opts(DATASET, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train, test = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test)
    q = np.asarray(qs[QUERY_INDEX], dtype=float)

    direction = np.asarray(qs[-1]) - np.asarray(qs[0])
    direction = direction/np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    t = np.linspace(-T_LIM, T_LIM, 400)
    line = q[None, :] + t[:, None]*direction[None, :]

    e_std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
        clf, query_point=q, train_data=train, test_data=test)
    e_log = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (logit)'](
        clf, query_point=q, train_data=train, test_data=test)
    p_bb = clf.predict_proba(line)[:, 1]
    raw_std = e_std.surrogate_model.predict(line)[:, 1]
    p_log = e_log.predict_proba(line)[:, 1]

    # the annotations are configuration means over all 20 query points, as everywhere
    # else in the study - a single point's ratio is not representative of the column.
    # Only the curves are drawn at QUERY_INDEX.
    has_truth = gradients.has_gradient(model)
    truth = gradients.grad_logit(clf, model, np.asarray(qs, dtype=float)) if has_truth \
        else None
    b_all, cos_all = {'standard': [], 'logit': []}, {'standard': [], 'logit': []}
    for i, qi in enumerate(qs):
        qi = np.asarray(qi, dtype=float)
        eval_i = get_local_points(test, qi)
        for label, name in (('standard', 'bLIMEy (normal)'), ('logit', 'bLIMEy (logit)')):
            e = clime.explainer.AVAILABLE_EXPLAINERS[name](
                clf, query_point=qi, train_data=train, test_data=test)
            b_all[label].append(brier(e, black_box_model=clf, data=eval_i, query_point=qi))
            if has_truth and np.linalg.norm(truth[i]) > 0:
                cos_all[label].append(
                    cosine(np.asarray(e.get_explanation(), dtype=float), truth[i]))
    b_std, b_log = np.mean(b_all['standard']), np.mean(b_all['logit'])

    # ---- feature space -------------------------------------------------------------
    ax = axs[0, col]
    X, y = np.asarray(test['X']), np.asarray(test['y'])
    u, v = (X - q) @ direction, (X - q) @ perp
    # drawn past the axis limits so no edge of the wash is visible inside the panel
    gu, gv = np.meshgrid(np.linspace(-1.4*T_LIM, 1.4*T_LIM, 240),
                         np.linspace(-1.4*T_LIM, 1.4*T_LIM, 160))
    grid = q[None, :] + gu.ravel()[:, None]*direction[None, :] \
                      + gv.ravel()[:, None]*perp[None, :]
    surf = clf.predict_proba(grid)[:, 1].reshape(gu.shape)
    ax.pcolormesh(gu, gv, surf, cmap=WASH, vmin=0, vmax=1, shading='gouraud',
                  zorder=0, rasterized=True)
    ax.contour(gu, gv, surf, levels=[0.5], colors=[INK], linewidths=0.8, zorder=2)
    for cls, colour in [(0, BLUE), (1, ORANGE)]:
        m = y == cls
        ax.scatter(u[m], v[m], s=5, facecolor=colour, edgecolor='white', linewidth=0.2,
                   alpha=0.7, zorder=3)
    ax.plot([-T_LIM, T_LIM], [0, 0], color=INK, lw=0.7, zorder=4)
    ax.scatter([0], [0], s=24, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f'$\\bf{{{letter}}}$  {nice}', fontsize=8.2, color=INK, pad=13)
    ax.annotate(f'log-odds {geometry}', xy=(0.5, 1.03), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=6.3, color=INK_2,
                annotation_clip=False)
    if col == 0:
        ax.set_ylabel('feature\nspace', fontsize=7.5, color=INK_2)

    # ---- probability space ---------------------------------------------------------
    ax = axs[1, col]
    ax.axhspan(-0.35, 0, color='#f2f1ed', zorder=0)
    ax.axhspan(1, 1.35, color='#f2f1ed', zorder=0)
    ax.plot(t, p_bb, color=INK, lw=3.2, label='black box $f$', zorder=3)
    ax.plot(t, raw_std, color=BLUE, ls=':', lw=1.2, zorder=2)
    ax.plot(t, np.clip(raw_std, 0, 1), color=BLUE, lw=1.5, label='standard LIME',
            zorder=4)
    ax.plot(t, p_log, color=ORANGE, lw=1.5, label='Logit-LIME', zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_ylim(-0.35, 1.35)
    ax.annotate(f'Brier {fmt_ratio(b_std/b_log)} better' if b_log > 0 else 'Brier —',
                xy=(0.04, 0.96), xycoords='axes fraction', fontsize=6.3, color=INK_2,
                va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                                    pad=1.2))
    if col == 0:
        ax.set_ylabel('probability', fontsize=8.5)

    # ---- logit space, and whether a ground truth exists ----------------------------
    ax = axs[2, col]
    ax.plot(t, logit(p_bb), color=INK, lw=3.2, zorder=3)
    ax.plot(t, logit(np.clip(raw_std, 1e-12, 1 - 1e-12)), color=BLUE, lw=1.5, zorder=4)
    ax.plot(t, logit(p_log), color=ORANGE, lw=1.5, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_ylim(-11.5, 15.5)
    if has_truth:
        c_std, c_log = np.mean(cos_all['standard']), np.mean(cos_all['logit'])
        note = f'ground truth exists\ncos  {c_std:.2f} / {c_log:.2f}'
        colour = INK_2
        summary.append((nice, b_std/b_log, c_std, c_log))
    else:
        note = 'no gradient:\nno ground truth'
        colour = '#a3453b'
        summary.append((nice, b_std/b_log, np.nan, np.nan))
    ax.annotate(note, xy=(0.04, 0.97), xycoords='axes fraction', fontsize=6.3,
                color=colour, va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.4))
    if col == 0:
        ax.set_ylabel('log-odds', fontsize=8.5)

axs[0, 0].set_xlim(-T_LIM, T_LIM)
fig.align_ylabels()
fig.tight_layout(w_pad=0.9, h_pad=0.5, rect=(0, 0.075, 1, 1))
fig.supxlabel('distance from $q$ along the transect', y=0.062, fontsize=9, color=INK)
handles, labels = axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.8)

fig.canvas.draw()
for col in range(len(COLUMNS)):
    ax = axs[0, col]
    bb = ax.get_window_extent()
    half = T_LIM*(bb.height/bb.width)
    ax.set_ylim(-half, half)

fig.savefig(paths.fig('fig_group_gallery.pdf'))
fig.savefig(paths.fig('fig_group_gallery.png'))
print(f"{'black box':<18s} {'Brier ratio':>12s} {'cos std':>9s} {'cos logit':>10s}")
for nice, ratio, cs, cl in summary:
    print(f'{nice:<18s} {ratio:>12,.2f} {cs:>9.3f} {cl:>10.3f}')
print('fig_group_gallery written')
