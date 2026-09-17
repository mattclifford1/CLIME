'''
The same result, rendered as an image (-> figs/fig_digits.pdf).

Every other explanation figure here is a bar chart, which forces the reader to compare
ranked lists. On a dataset whose features have a spatial layout the comparison can be made
directly: the black box's true coefficients, standard LIME's and Logit-LIME's are all
8x8 images.

Digits 3 vs 8 with a logistic black box, so the log-odds are exactly linear and coef_ IS
the local importance vector (the group A argument of sweep_ground_truth.py). 64 features
puts this in the regime where the paper reports both surrogates degrading, which is the
point: the gap between them is widest where the problem is hardest.

The bottom row is the reason the figure is laid out this way. Side by side the two
explanations look similar - a cosine of 0.91 is not a visibly wrong picture - so the
figure also draws each surrogate's signed error against the truth on a shared scale.
That is the quantity actually being claimed, rather than a difference the reader is asked
to perform by eye.

The query point shown is the one whose standard-LIME cosine is closest to the mean over
all 20 query points, chosen that way rather than by hand so the figure is representative
of the configuration and not of a picked case. The mean over all 20 is printed and
annotated.

usage:  python figures/fig_digits.py
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
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

DATASET, MODEL = 'Digits 3 vs 8', 'Logistic'
SIDE = 8                 # the images are 8x8


def unit_max(v):
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
qs, _ = get_points_between_class_means(test)

# every query point first, so the point that gets drawn can be chosen as a representative
# one rather than a flattering one
rows = []
for q in qs:
    q = np.asarray(q, dtype=float)
    truth = gradients.grad_logit(clf, MODEL, q)[0]
    e_std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
        clf, query_point=q, train_data=train, test_data=test)
    e_log = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (logit)'](
        clf, query_point=q, train_data=train, test_data=test)
    c_std = np.asarray(e_std.get_explanation(), dtype=float)
    c_log = np.asarray(e_log.get_explanation(), dtype=float)
    rows.append((q, truth, c_std, c_log, cosine(c_std, truth), cosine(c_log, truth)))

mean_std = float(np.nanmean([x[4] for x in rows]))
mean_log = float(np.nanmean([x[5] for x in rows]))
pick = int(np.argmin([abs(x[4] - mean_std) for x in rows]))
q, truth, c_std, c_log, cos_std, cos_log = rows[pick]

# everything is compared after scaling to unit maximum: cosine is scale invariant and the
# two surrogates report in different units (probability against log-odds per unit feature),
# so an unscaled difference would measure the change of units rather than the error
t_img, s_img, l_img = unit_max(truth), unit_max(c_std), unit_max(c_log)
err_std, err_log = s_img - t_img, l_img - t_img
emax = float(max(np.abs(err_std).max(), np.abs(err_log).max()))

# ---- the figure ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(5.4, 3.9))

panels = [
    (axs[0, 0], '(a) black box (truth)', t_img, 1.0, 'RdBu_r', None),
    (axs[0, 1], '(b) standard LIME', s_img, 1.0, 'RdBu_r', f'cosine to truth  {cos_std:.3f}'),
    (axs[0, 2], '(c) Logit-LIME', l_img, 1.0, 'RdBu_r', f'cosine to truth  {cos_log:.3f}'),
    (axs[1, 1], '(e) error, standard', err_std, emax, 'PuOr_r',
     f'largest  {np.abs(err_std).max():.2f}'),
    (axs[1, 2], '(f) error, Logit-LIME', err_log, emax, 'PuOr_r',
     f'largest  {np.abs(err_log).max():.2f}'),
]

for ax, title, img, lim, cmap, foot in panels:
    ax.imshow(img.reshape(SIDE, SIDE), cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    if foot:
        ax.set_xlabel(foot, fontsize=6.3, labelpad=2)

# the query point, in the standardised units the black box actually sees. It lies on the
# line between the class means, so it is not one of the data digits and is not drawn as
# though it were
ax = axs[1, 0]
ax.imshow(q.reshape(SIDE, SIDE), cmap='gray_r')
ax.set_title('(d) query point $q$', fontsize=8, color=INK, pad=3)
ax.set_xlabel('standardised pixels', fontsize=6.3, labelpad=2)

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.6)

fig.tight_layout(w_pad=0.7, h_pad=1.1)

# as a footer rather than beside a panel: the 2x3 grid has no empty slot to put it in, and
# every position inside one crowds a neighbouring title
fig.text(0.5, -0.015,
         f'mean cosine over {len(rows)} query points   '
         f'standard {mean_std:.3f}    Logit-LIME {mean_log:.3f}',
         ha='center', va='top', fontsize=6.4, color=INK_2)
fig.savefig(paths.fig('fig_digits.pdf'))
fig.savefig(paths.fig('fig_digits.png'))

print(f'fig_digits written   query point {pick} of {len(rows)} '
      f'(closest to the mean, not picked)')
print(f'  this point   cos standard {cos_std:.4f}   logit {cos_log:.4f}')
print(f'  all points   cos standard {mean_std:.4f}   logit {mean_log:.4f}')
print(f'  logit better at {sum(1 for x in rows if x[5] > x[4])}/{len(rows)} query points')
print(f'  largest error (unit-max scaled)  standard {np.abs(err_std).max():.3f}   '
      f'logit {np.abs(err_log).max():.3f}')
print(f'  mean |error|                     standard {np.abs(err_std).mean():.3f}   '
      f'logit {np.abs(err_log).mean():.3f}')
