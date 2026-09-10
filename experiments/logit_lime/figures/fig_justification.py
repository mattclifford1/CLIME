'''
Why a fidelity score is worth reading at all: one case where the chain is visible end to
end (-> figs/fig_justification.pdf).

The study measures Brier and KL - how well the surrogate reproduces the black box's
probabilities. A reader is entitled to ask why that should matter, since nobody consumes
a surrogate's probabilities: they consume its feature importances. The link is an
assumption unless it is checked, and it can only be checked where the true importances
are known.

Breast Cancer with a logistic black box is that case. Its log-odds are exactly linear, so
its own coefficients ARE the local feature importances, and every step of the chain can
be drawn:

  (a) in probability space the standard surrogate fits a straight line to a sigmoid, and
      leaves [0,1] on both sides of it
  (b) in logit space the black box is a straight line, which the logit surrogate lies on
      and the standard surrogate does not
  (c) the coefficients that come out: the logit surrogate recovers the black box's own
      ranking, the standard one does not

The same three panels also show why the failure is invisible to a threshold based
instrument: both surrogates put the boundary in nearly the same place. It is the
probabilities, and the weights that produce them, that differ.

usage:  python figures/fig_justification.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients
from common.style import *

import warnings
import numpy as np
import clime
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASET, MODEL = 'Breast Cancer', 'Logistic'
QUERY_INDEX = 10
N_FEATURES = 8          # rows in the importance panel
T_LIM = 2.6             # transect half length, in standardised units


def logit(p, lim=12):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.clip(np.log(p/(1 - p)), -lim, lim)


def unit_max(v):
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
names = list(test['feature_names'])
qs, _ = get_points_between_class_means(test)
q = np.asarray(qs[QUERY_INDEX], dtype=float)

truth = gradients.grad_logit(clf, MODEL, q)[0]

e_std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
    clf, query_point=q, train_data=train, test_data=test)
e_log = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (logit)'](
    clf, query_point=q, train_data=train, test_data=test)
c_std = np.asarray(e_std.get_explanation(), dtype=float)
c_log = np.asarray(e_log.get_explanation(), dtype=float)

# the transect runs along the query-point line, re-parameterised as distance from q
direction = np.asarray(qs[-1]) - np.asarray(qs[0])
direction = direction/np.linalg.norm(direction)
t = np.linspace(-T_LIM, T_LIM, 400)
line = q[None, :] + t[:, None]*direction[None, :]

p_bb = clf.predict_proba(line)[:, 1]
raw_std = e_std.surrogate_model.predict(line)[:, 1]      # unclipped
p_log = e_log.predict_proba(line)[:, 1]

# the fidelity numbers quoted in the caption, from the same surrogate objects
eval_data = get_local_points(test, q)
brier = clime.evaluation.AVAILABLE_EVALUATION_METRICS['Brier score (local)']
kl = clime.evaluation.AVAILABLE_EVALUATION_METRICS['KL divergence (local)']
scores = {label: (brier(e, black_box_model=clf, data=eval_data, query_point=q),
                  kl(e, black_box_model=clf, data=eval_data, query_point=q))
          for label, e in (('standard', e_std), ('logit', e_log))}


def cosine(a, b):
    return float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b)))


fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6),
                        gridspec_kw=dict(width_ratios=[1, 1, 1.5]))

# the locality kernel, so the reader can see where the fit is actually being asked to be
# good - a mismatch out in the tail would not matter, and this one is not out in the tail
kernel = clime.data.utils.costs.weights_based_on_distance(q, line)
kernel = kernel/kernel.max()

# ---- (a) probability space ----------------------------------------------------------
ax = axs[0]
ax.axhspan(-0.4, 0, color='#f2f1ed', zorder=0)
ax.axhspan(1, 1.4, color='#f2f1ed', zorder=0)
ax.fill_between(t, -0.4, -0.4 + 0.32*kernel, color='#dfe8f4', lw=0, zorder=1)
ax.plot(t, p_bb, color=INK, lw=3.2, label='black box $f$', zorder=3)
ax.plot(t, raw_std, color=BLUE, ls=':', lw=1.2, zorder=2)
ax.plot(t, np.clip(raw_std, 0, 1), color=BLUE, lw=1.6, label='standard LIME', zorder=4)
ax.plot(t, p_log, color=ORANGE, lw=1.6, label='Logit-LIME', zorder=4)
ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
ax.set_ylim(-0.4, 1.4)
ax.set_xlim(-T_LIM, T_LIM)
ax.set_ylabel('probability  $p(y{=}1\\,|\\,x)$')
ax.set_xlabel('distance from $q$')
ax.set_title('(a) what is fitted', fontsize=8.5, color=INK)
ax.annotate('locality weight', xy=(0.5, 0.015), xycoords='axes fraction', fontsize=6,
            color='#5b7fa8', ha='center')

# ---- (b) logit space ----------------------------------------------------------------
ax = axs[1]
ax.plot(t, logit(p_bb), color=INK, lw=3.2, zorder=3)
ax.plot(t, logit(np.clip(raw_std, 1e-12, 1 - 1e-12)), color=BLUE, lw=1.6, zorder=4)
ax.plot(t, logit(p_log), color=ORANGE, lw=1.6, zorder=4)
ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
ax.set_xlim(-T_LIM, T_LIM)
ax.set_ylabel('log-odds  $\\mathrm{logit}\\,p$')
ax.set_xlabel('distance from $q$')
ax.set_title('(b) where the black box is straight', fontsize=8.5, color=INK)
ax.annotate(f"Brier  {scores['standard'][0]/scores['logit'][0]:,.0f}$\\times$ better\n"
            f"KL     {scores['standard'][1]/scores['logit'][1]:,.0f}$\\times$ better",
            xy=(0.97, 0.03), xycoords='axes fraction', fontsize=6.2, color=INK_2,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.6))

# ---- (c) the explanation that comes out ---------------------------------------------
# same rows as Table 3: the truly most important features, plus whatever a surrogate puts
# first - without that row the figure would not show the actual error, which is a feature
# the black box ranks ninth being promoted to the top
ax = axs[2]
order = list(np.argsort(-np.abs(truth))[:N_FEATURES])
extra = [int(np.argmax(np.abs(c))) for c in (c_std, c_log)]
order += [i for i in dict.fromkeys(extra) if i not in order]
ypos = np.arange(len(order))[::-1]
h = 0.26
ax.barh(ypos + h, unit_max(truth)[order], height=h, color=INK, label='black box (truth)')
ax.barh(ypos, unit_max(c_std)[order], height=h, color=BLUE, label='standard LIME')
ax.barh(ypos - h, unit_max(c_log)[order], height=h, color=ORANGE, label='Logit-LIME')
ax.set_yticks(ypos)
ax.set_yticklabels([names[i] for i in order], fontsize=6.2)
for tick, i in zip(ax.get_yticklabels(), order):
    if i in extra and i not in list(np.argsort(-np.abs(truth))[:N_FEATURES]):
        tick.set_color(BLUE)          # the feature standard LIME wrongly promotes
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlim(-1.18, 0.62)
ax.set_xlabel('weight, scaled to unit maximum')
ax.set_title('(c) the explanation that comes out', fontsize=8.5, color=INK)
ax.grid(axis='y', visible=False)
ax.annotate(f'cosine to truth\nstandard  {cosine(c_std, truth):.2f}\n'
            f'Logit-LIME  {cosine(c_log, truth):.2f}',
            xy=(0.985, 0.05), xycoords='axes fraction', fontsize=6.5, color=INK_2,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.6))

fig.tight_layout(w_pad=1.4, rect=(0, 0.09, 1, 1))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.8)

fig.savefig(paths.fig('fig_justification.pdf'))
fig.savefig(paths.fig('fig_justification.png'))
print(f'fig_justification written   '
      f'cos standard {cosine(c_std, truth):.4f}  logit {cosine(c_log, truth):.4f}')
print(f"  Brier standard {scores['standard'][0]:.3e}  logit {scores['logit'][0]:.3e}"
      f"   ratio {scores['standard'][0]/scores['logit'][0]:.1f}x")
print(f"  KL    standard {scores['standard'][1]:.3e}  logit {scores['logit'][1]:.3e}"
      f"   ratio {scores['standard'][1]/scores['logit'][1]:.1f}x")
