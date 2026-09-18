'''
The explanation as an image, with the data it is about (-> figs/fig_digits.pdf).

Every other explanation figure here is a bar chart, which forces the reader to compare
ranked lists. On a dataset whose features have a spatial layout the comparison can be made
directly: the black box's true coefficients, standard LIME's and Logit-LIME's are all 8x8
images.

The first row is context, and it is not decoration. A coefficient map means nothing on its
own - "red pushes towards 8" is unreadable unless the reader can see what an 8 looks like in
this data and how it differs from a 3. So the row shows real examples of both classes before
the maps that claim to explain them, and the query point alongside, which is a point on the
line between the class means (Section 5) and so is deliberately NOT one of the data digits:
it sits between them and looks it.

THE IMAGES ARE DRAWN IN RAW PIXEL VALUES, THE COEFFICIENT MAPS IN STANDARDISED UNITS, and
the reason is worth recording because two earlier versions of this figure got it wrong.
Standardising divides each pixel by its own standard deviation. The border pixels that are
nearly always blank have a tiny one, so a stray mark there becomes a value of +12 while the
strokes that actually draw the digit sit between -1 and +3; worse, dividing each pixel
separately removes exactly the shared stroke structure that makes a 3 look like a 3. On a
min/max colour scale the outliers own the colourmap and every digit renders as flat grey
noise; on a percentile scale the contrast comes back but the shapes do not, because the
information is gone from the values rather than from the colour range. Raw pixels are the
only form in which a digit is legible, so the context row is mapped back into them - the
pipeline's standardisation is reconstructed from the training split and inverted, which
reproduces its own arrays exactly (checked: max difference 0). The coefficient panels stay
in the space the surrogate was actually fitted in.

The bottom row's error panels are the other reason the figure is laid out this way. Side by
side the two explanations look similar - a cosine of 0.91 is not a visibly wrong picture -
so the figure draws each surrogate's signed error against the truth on a shared scale. That
is the quantity actually being claimed, rather than a difference the reader is asked to
perform by eye.

The query point shown is the one whose standard-LIME cosine is closest to the mean over all
20 query points, chosen that way rather than by hand so the figure is representative of the
configuration and not of a picked case.

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
N_TILE = 4               # examples shown per class, as a 2x2 tile
PIXEL_MAX = 16.0         # the native range of sklearn's digits


def unit_max(v):
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


def tile(images):
    '''four 8x8 digits into one 16x16 block, so a panel can show real data compactly'''
    ims = [im.reshape(SIDE, SIDE) for im in images[:N_TILE]]
    return np.block([[ims[0], ims[1]], [ims[2], ims[3]]])


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
qs, _ = get_points_between_class_means(test)

# the same dataset before the pipeline standardised it. The loader is deterministic, so the
# rows line up with the pipeline's own splits (labels and order verified identical), and the
# training mean and standard deviation are exactly the transform the pipeline applied
train_raw, test_raw = clime.data.AVAILABLE_DATASETS[DATASET]()
mu = train_raw['X'].mean(axis=0)
sd = train_raw['X'].std(axis=0)
sd = np.where(sd == 0, 1.0, sd)          # StandardScaler's zero-variance convention

# real examples of each class, in pixel values. class 1 is the digit 8 (see
# clime/data/loaders/sklearn_toy.py), class 0 the digit 3
y = np.asarray(test_raw['y'])
ex_3 = tile(test_raw['X'][y == 0])
ex_8 = tile(test_raw['X'][y == 1])

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
q_pixels = q*sd + mu                      # back into the units a digit is legible in

# everything is compared after scaling to unit maximum: cosine is scale invariant and the
# two surrogates report in different units (probability against log-odds per unit feature),
# so an unscaled difference would measure the change of units rather than the error
t_img, s_img, l_img = unit_max(truth), unit_max(c_std), unit_max(c_log)
err_std, err_log = s_img - t_img, l_img - t_img
emax = float(max(np.abs(err_std).max(), np.abs(err_log).max()))

# ---- the figure ---------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0))

grey = [(axs[0, 0], '(a) examples: digit 3', ex_3, 'class 0, pixel values'),
        (axs[0, 1], '(b) examples: digit 8', ex_8, 'class 1, pixel values'),
        (axs[0, 2], '(c) query point $q$', q_pixels.reshape(SIDE, SIDE),
         'between the class means')]
for ax, title, img, foot in grey:
    ax.imshow(img, cmap='gray_r', vmin=0.0, vmax=PIXEL_MAX)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    ax.set_xlabel(foot, fontsize=6.3, labelpad=2)
# separators, so the tiles read as four digits rather than one texture
for ax in (axs[0, 0], axs[0, 1]):
    ax.axhline(SIDE - 0.5, color=ORANGE, lw=0.8)
    ax.axvline(SIDE - 0.5, color=ORANGE, lw=0.8)

panels = [(axs[0, 3], '(d) black box (truth)', t_img, 1.0, 'RdBu_r', None),
          (axs[1, 0], '(e) standard LIME', s_img, 1.0, 'RdBu_r',
           f'cosine to truth  {cos_std:.3f}'),
          (axs[1, 1], '(f) Logit-LIME', l_img, 1.0, 'RdBu_r',
           f'cosine to truth  {cos_log:.3f}'),
          (axs[1, 2], '(g) error, standard', err_std, emax, 'PuOr_r',
           f'largest  {np.abs(err_std).max():.2f}'),
          (axs[1, 3], '(h) error, Logit-LIME', err_log, emax, 'PuOr_r',
           f'largest  {np.abs(err_log).max():.2f}')]
for ax, title, img, lim, cmap, foot in panels:
    ax.imshow(img.reshape(SIDE, SIDE), cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    if foot:
        ax.set_xlabel(foot, fontsize=6.3, labelpad=2)

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.6)

fig.tight_layout(w_pad=0.7, h_pad=1.1)
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
print(f'  query point in pixel units: [{q_pixels.min():.1f}, {q_pixels.max():.1f}] '
      f'of 0-{PIXEL_MAX:.0f}')
