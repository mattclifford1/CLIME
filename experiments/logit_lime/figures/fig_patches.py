'''
The same question in the interpretable domain (-> figs/fig_patches.pdf).

LIME on an image does not show the surrogate pixels. It cuts the image into patches and
shows it one binary "patch present / patch absent" indicator per patch, which is the
transform the paper otherwise avoids in order to isolate the effect it studies. This figure
is that transform made visible, and the check that the effect survives it.

The panels walk the whole pipeline once:

  (a) the query image, cut into 2x2 patches - the 16 things the surrogate can see
  (b) one sampled x(z): the patches that are off are replaced by the baseline, which in
      the standardised space the pipeline hands over is the dataset mean image
  (c) the truth. For a black box with linear log-odds this is exact, not estimated:
      gamma_j = sum_{i in patch j} beta_i (q_i - b_i)   (common/patches.py)
  (d) standard LIME's patch weights, (e) Logit-LIME's
  (f) the signed error of each against (c), per patch

Logistic regression is the black box because it is the paper's running group A example and
the least selective choice, not because it flatters the result: it is the configuration
with the SMALLEST cosine gap of the three linear families (Nearest Class Mean's is far
wider). What it does show at full strength is the fidelity gap, which is a factor of ~10^5
in KL over the hypercube.

usage:  python figures/fig_patches.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, patches as P
from common.style import *

import warnings
import numpy as np
import clime
from sweeps.sweep import opts, METRICS
from sweeps.sweep_patches import query_images, brier, kl, cosine

warnings.filterwarnings('ignore')

DATASET, MODEL = 'Digits 3 vs 8', 'Logistic'
SIDE, PATCH = P.SIDE, P.PATCH
PSIDE = SIDE//PATCH          # 4 patches per side


def unit_max(v):
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, test = r['clf'], r['test_data']
grid = P.patch_indices()
Q, _ = query_images(test)

# every query image first, so the one drawn is representative rather than chosen
rows = []
for q in Q:
    q = np.asarray(q, dtype=float)
    baseline = np.zeros_like(q)
    beta = np.asarray(gradients.grad_logit(clf, MODEL, q[None, :])[0], float)
    truth = P.true_patch_coefficients(beta, q, baseline, grid)
    e_std = P.PatchLIME(clf, q, grid, baseline=baseline, train_logits=False)
    e_log = P.PatchLIME(clf, q, grid, baseline=baseline, train_logits=True)
    c_std = np.asarray(e_std.get_explanation(), float)
    c_log = np.asarray(e_log.get_explanation(), float)

    Ze = P.sample_Z(len(grid), q, salt=P.EVAL_SALT)
    we = P.kernel_weights(Ze)
    fe = clf.predict_proba(P.compose(Ze, q, baseline, grid))[:, 1]
    kls = (kl(e_std.predict_proba(Ze)[:, 1], fe, we),
           kl(e_log.predict_proba(Ze)[:, 1], fe, we))
    rows.append((q, truth, c_std, c_log, cosine(c_std, truth), cosine(c_log, truth), kls))

mean_std = float(np.nanmean([x[4] for x in rows]))
mean_log = float(np.nanmean([x[5] for x in rows]))
pick = int(np.argmin([abs(x[4] - mean_std) for x in rows]))
q, truth, c_std, c_log, cos_std, cos_log, kls = rows[pick]

t_img, s_img, l_img = unit_max(truth), unit_max(c_std), unit_max(c_log)
err_std, err_log = s_img - t_img, l_img - t_img

# one sampled masked image, for panel (b)
z_demo = P.sample_Z(len(grid), q)[3]
x_demo = P.compose(z_demo[None, :], q, np.zeros_like(q), grid)[0]

# ---- the figure ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(5.8, 4.1))


def grid_lines(ax):
    for k in range(PATCH, SIDE, PATCH):
        ax.axhline(k - 0.5, color=ORANGE, lw=0.7)
        ax.axvline(k - 0.5, color=ORANGE, lw=0.7)


# (a) and (b) must share a grey scale: (b) is (a) with patches replaced by the baseline,
# and on separate scales the same surviving pixel renders as a different grey in each
glim = (float(min(q.min(), x_demo.min())), float(max(q.max(), x_demo.max())))

ax = axs[0, 0]
ax.imshow(q.reshape(SIDE, SIDE), cmap='gray_r', vmin=glim[0], vmax=glim[1])
grid_lines(ax)
ax.set_title('(a) query image, in patches', fontsize=8, color=INK, pad=3)
ax.set_xlabel(f'{len(grid)} binary indicators', fontsize=6.3, labelpad=2)

ax = axs[0, 1]
ax.imshow(x_demo.reshape(SIDE, SIDE), cmap='gray_r', vmin=glim[0], vmax=glim[1])
grid_lines(ax)
ax.set_title('(b) one sample $x(z)$', fontsize=8, color=INK, pad=3)
ax.set_xlabel(f'{int(z_demo.sum())} of {len(grid)} patches on', fontsize=6.3, labelpad=2)

heat = [(axs[0, 2], '(c) truth $\\gamma$', t_img, None),
        (axs[1, 0], '(d) standard LIME', s_img, f'cosine  {cos_std:.3f}'),
        (axs[1, 1], '(e) Logit-LIME', l_img, f'cosine  {cos_log:.3f}')]
for ax, title, img, foot in heat:
    ax.imshow(img.reshape(PSIDE, PSIDE), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    if foot:
        ax.set_xlabel(foot, fontsize=6.3, labelpad=2)

for ax in list(axs[0, :]) + [axs[1, 0], axs[1, 1]]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.6)

# (f) the error per patch: at a cosine of 1.000 the map in (e) is the map in (c), and the
# only way to show "no error" honestly is to plot the error
ax = axs[1, 2]
x = np.arange(len(grid))
ax.bar(x - 0.2, err_std, width=0.4, color=BLUE, label='standard')
ax.bar(x + 0.2, err_log, width=0.4, color=ORANGE, label='Logit-LIME')
ax.axhline(0, color=MUTED, lw=0.6)
# headroom above the bars, so the annotation has somewhere to sit that is not on top of
# the data it is describing
lim = float(np.max(np.abs(np.concatenate([err_std, err_log]))))
ax.set_ylim(-1.15*lim, 1.45*lim)
ax.set_title('(f) error against the truth', fontsize=8, color=INK, pad=3)
ax.set_xlabel('patch', fontsize=6.3, labelpad=2)
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=6)
ax.legend(fontsize=5.8, loc='lower left', handlelength=1.0, borderpad=0.2)
# without this the orange bars read as a plotting failure rather than as the result:
# Logit-LIME's error is not missing from the panel, it is too small to draw
ax.annotate(f'largest $|$error$|$\nstandard  {np.abs(err_std).max():.3f}\n'
            f'Logit-LIME  {np.abs(err_log).max():.0e}',
            xy=(0.97, 0.97), xycoords='axes fraction', fontsize=5.8, color=INK_2,
            ha='right', va='top')

fig.tight_layout(w_pad=0.8, h_pad=1.2)
fig.text(0.5, -0.015,
         f'mean cosine over {len(rows)} query images   '
         f'standard {mean_std:.3f}    Logit-LIME {mean_log:.3f}',
         ha='center', va='top', fontsize=6.4, color=INK_2)

fig.savefig(paths.fig('fig_patches.pdf'))
fig.savefig(paths.fig('fig_patches.png'))

print(f'fig_patches written   image {pick} of {len(rows)} (closest to the mean)')
print(f'  this image   cos standard {cos_std:.4f}   logit {cos_log:.4f}')
print(f'  all images   cos standard {mean_std:.4f}   logit {mean_log:.4f}')
print(f'  KL           standard {kls[0]:.3e}   logit {kls[1]:.3e}   '
      f'ratio {kls[0]/max(kls[1], 1e-30):,.0f}x')
print(f'  largest patch error   standard {np.abs(err_std).max():.3f}   '
      f'logit {np.abs(err_log).max():.3f}')
