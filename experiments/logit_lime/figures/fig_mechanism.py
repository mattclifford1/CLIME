'''
Figure 1 - the mechanism.

A transect through the black box's decision surface, shown in probability space and in
logit space, with both surrogates overlaid.

Four black boxes, chosen to separate two properties a reader is likely to conflate:

  Logistic regression   log-odds exactly linear in x   -> Logit-LIME recovers f
  MLP                   log-odds learned, near linear  -> large benefit, not exact
  SVM (RBF)             smooth, but log-odds curve     -> no benefit
  Random forest         piecewise constant             -> no benefit, nothing fits

The two middle columns do the work. The MLP shows the benefit is not an artefact of the
black box literally being the surrogate's hypothesis class - a learned, nonlinear model
whose log-odds merely happen to be close to linear still gains an order of magnitude. The
SVM is every bit as smooth as the logistic regression, and the standard surrogate leaves
[0,1] there just as badly, yet fitting in logit space buys almost nothing. Smoothness is
not the property that matters; linearity of the log-odds is.

The top row is a locator strip. Its horizontal axis is the same transect coordinate as
the two rows below, so the reader can see what 'distance from q along the transect'
means in the feature space, and what each black box's surface looks like along it.

usage:  python fig_mechanism.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

from common.style import *
import numpy as np, warnings, clime
from matplotlib.colors import LinearSegmentedColormap
from clime.evaluation.key_points import get_points_between_class_means
warnings.filterwarnings('ignore')

T_LIM = 2.6                       # extent of the transect, in standardised units
QUERY_INDEX = 11                  # just off the decision boundary

train, test = clime.data.AVAILABLE_DATASETS['Gaussian'](
    class_samples=[200, 200], gaussian_means=[[-1, -1], [1, 1]],
    gaussian_covs=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
norm = clime.data.normaliser(train); train, test = norm(train), norm(test)
qs, _ = get_points_between_class_means(test)
qs = np.asarray(qs)
q = qs[QUERY_INDEX]

# the transect is the query-point line of the experiments, running class 0 -> class 1,
# re-parameterised as signed distance from q
direction = qs[-1] - qs[0]
direction = direction/np.linalg.norm(direction)
perp = np.array([-direction[1], direction[0]])
t = np.linspace(-T_LIM, T_LIM, 400)
line = q[None, :] + t[:, None]*direction[None, :]

X, y = np.asarray(test['X']), np.asarray(test['y'])
u_data = (X - q) @ direction                 # data in transect coordinates
v_data = (X - q) @ perp
u_qs = (qs - q) @ direction


# a very light wash: the class colours have to stay legible on top of it, and a
# perceptually strong map would dominate a panel this small
WASH = LinearSegmentedColormap.from_list('wash', ['#ffffff', '#dbdad6'])


def logit(p, lim=8):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.clip(np.log(p/(1-p)), -lim, lim)


MODELS = [('Logistic regression', 'Logistic'),
          ('Neural network (MLP)', 'MLP'),
          ('SVM (RBF)', 'SVM'),
          ('Random forest', 'Random Forest')]

# what the standard surrogate's Brier score is divided by, on this dataset, from the
# registered sweep (results_taxonomy.json, Gaussian).
BENEFIT = {'Logistic': '$12{,}600\\times$', 'MLP': '$9.7\\times$',
           'SVM': '$1.4\\times$', 'Random Forest': '$1.6\\times$'}

NOTE = {'Logistic': 'log-odds linear in $x$:\nLogit-LIME recovers $f$',
        'MLP': 'learned, but log-odds\nnearly linear',
        'SVM': 'smooth, but the log-odds\ncurve and turn back',
        'Random Forest': 'log-odds are a step\nfunction: neither fits'}

NCOL = len(MODELS)
fig, axs = plt.subplots(3, NCOL, figsize=(6.9, 4.85), sharex=True,
                        gridspec_kw=dict(height_ratios=[0.5, 1, 1]))

for col, (nice, model) in enumerate(MODELS):
    clf = clime.models.AVAILABLE_MODELS[model](train)
    p_bb = clf.predict_proba(line)[:, 1]
    e_std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
        clf, query_point=q, train_data=train, test_data=test)
    e_log = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (logit)'](
        clf, query_point=q, train_data=train, test_data=test)
    raw_std = e_std.surrogate_model.predict(line)[:, 1]   # unclipped: shows it leave [0,1]
    p_std = np.clip(raw_std, 0, 1)
    p_log = e_log.predict_proba(line)[:, 1]

    # ---- locator strip: the feature space, rotated so the transect is horizontal ------
    ax = axs[0, col]
    gu, gv = np.meshgrid(np.linspace(-T_LIM, T_LIM, 260), np.linspace(-1.6, 1.6, 130))
    grid = q[None, :] + gu.ravel()[:, None]*direction[None, :] \
                      + gv.ravel()[:, None]*perp[None, :]
    surf = clf.predict_proba(grid)[:, 1].reshape(gu.shape)
    ax.pcolormesh(gu, gv, surf, cmap=WASH, vmin=0, vmax=1, shading='gouraud',
                  zorder=0, rasterized=True)
    ax.contour(gu, gv, surf, levels=[0.5], colors=[INK], linewidths=0.8, zorder=2)
    for cls, colour in [(0, BLUE), (1, ORANGE)]:
        m = y == cls
        ax.scatter(u_data[m], v_data[m], s=6, facecolor=colour, edgecolor='white',
                   linewidth=0.25, alpha=0.7, zorder=3)
    ax.plot([-T_LIM, T_LIM], [0, 0], color=INK, lw=0.8, zorder=4)
    inside = np.abs(u_qs) <= T_LIM
    ax.scatter(u_qs[inside], np.zeros(inside.sum()), s=7, facecolor='white',
               edgecolor=INK, linewidth=0.6, zorder=5)
    ax.scatter([0], [0], s=28, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.set_yticks([])
    ax.grid(False)
    # the benefit sits as a column subtitle rather than inside a panel: all three panels
    # of a column are already carrying annotation, and it describes the column, not a panel
    ax.set_title(nice, color=INK, fontsize=8.5, pad=14)
    ax.annotate(f'Brier {BENEFIT[model]} better', xy=(0.5, 1.03),
                xycoords='axes fraction', ha='center', va='bottom', fontsize=6.5,
                color=INK_2, annotation_clip=False)
    if col == 0:
        ax.set_ylabel('feature\nspace', fontsize=7.5, color=INK_2)
        ax.annotate('$q$', xy=(0.53, 0.60), xycoords='axes fraction', fontsize=8,
                    color=INK)
        tag = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.0)
        ax.annotate('class 0', xy=(0.02, 0.05), xycoords='axes fraction', fontsize=6,
                    color=BLUE, bbox=tag)
        ax.annotate('class 1', xy=(0.98, 0.05), xycoords='axes fraction', fontsize=6,
                    color=ORANGE, ha='right', bbox=tag)

    # ---- probability space -----------------------------------------------------------
    ax = axs[1, col]
    ax.axhspan(-0.35, 0, color='#f2f1ed', zorder=0)
    ax.axhspan(1, 1.35, color='#f2f1ed', zorder=0)
    ax.plot(t, p_bb, color=INK, lw=3.4, label='black box $f$', zorder=3)
    ax.plot(t, raw_std, color=BLUE, ls=':', lw=1.3, zorder=2)
    ax.plot(t, p_std, color=BLUE, lw=1.6, label='standard LIME', zorder=4)
    ax.plot(t, p_log, color=ORANGE, lw=1.6, label='Logit-LIME', zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_ylim(-0.35, 1.35)
    if col == 0:
        ax.set_ylabel('probability  $p(y{=}1\\,|\\,x)$', fontsize=8.5)
        ax.annotate('outside $[0,1]$', xy=(-2.45, -0.30), fontsize=6.5, color=INK_2,
                    bbox=dict(facecolor='#f2f1ed', edgecolor='none', alpha=0.9, pad=1.2))

    # ---- logit space -----------------------------------------------------------------
    ax = axs[2, col]
    ax.plot(t, logit(p_bb), color=INK, lw=3.4, zorder=3)
    ax.plot(t, logit(p_std), color=BLUE, lw=1.6, zorder=4)
    ax.plot(t, logit(p_log), color=ORANGE, lw=1.6, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_ylim(-9.6, 13.6)          # headroom above the clipping level for the note
    ax.annotate(NOTE[model], xy=(0.03, 0.97), xycoords='axes fraction', fontsize=6.5,
                color=INK_2, va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.6))
    if col == 0:
        ax.set_ylabel('log-odds  $\\mathrm{logit}\\,p$', fontsize=8.5)

axs[0, 0].set_xlim(-T_LIM, T_LIM)
fig.align_ylabels()
# leave a strip at the bottom for the shared x label and the legend: with four narrow
# columns there is no longer room for a legend inside a panel without covering a curve
fig.tight_layout(w_pad=1.1, h_pad=0.55, rect=(0, 0.075, 1, 1))
fig.supxlabel('distance from $q$ along the transect', y=0.062, fontsize=9, color=INK)
handles, labels = axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.8)

# Make the locator strips true to scale: the transect coordinate and the perpendicular
# offset are the same units, so an unequal aspect would draw the decision boundary at the
# wrong angle. Forcing aspect='equal' would resize the axes and break the alignment with
# the panels below, so instead set the y range from the box the layout actually gave us.
fig.canvas.draw()
for col in range(NCOL):
    ax = axs[0, col]
    bb = ax.get_window_extent()
    half = T_LIM*(bb.height/bb.width)
    ax.set_ylim(-half, half)

fig.savefig(paths.fig('fig1_mechanism.pdf')); fig.savefig(paths.fig('fig1_mechanism.png'))
print('fig1 written')
