'''
One query point where the threshold cannot tell four surrogates apart
(-> figs/fig_blind.pdf).

The point is chosen by the protocol in analysis/select_example.py, not by hunting: it is
the only query point of the 1,650 with a gradient ground truth at which both fidelity
cells agree to within 0.01 while the Brier score differs by more than tenfold and the two
explanations differ by more than 0.4 in cosine.  That rarity is the honest framing and
belongs in the caption - the common failure is the tie, not this dramatic version of it.

Three panels, left to right, following the surrogate from what it predicts to what it is
read for:

  (a) the black box and the four surrogates along the transect through q.  They cross
      p = 1/2 within a hair of each other, which is the whole of what a thresholded
      instrument looks at, and they disagree about everything else.
  (b) the explanation that comes out.  Standard LIME's is at 52 degrees to the truth;
      Logit-LIME's and the hard-label surrogate's are nearly on it.
  (c) what each instrument says.  Fidelity gives all four the same number to three
      decimal places; Brier and KL span an order of magnitude.

The fourth surrogate is the null explainer of sweeps/sweep_null.py, which has no
explanation at all.  It is drawn in the neutral grey rather than a fourth categorical
colour because it is a baseline, not another method.

usage:  python figures/fig_blind.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients, surrogates
from common.style import *

import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASET, MODEL, QUERY_INDEX = 'Breast Cancer', 'MLP', 12
N_FEATURES = 7
KERNEL_MASS = 0.1              # transect extent, as in analysis/select_example.py

SERIES = [('standard LIME', 'bLIMEy (normal)', BLUE),
          ('Logit-LIME', 'bLIMEy (logit)', ORANGE),
          ('hard-label LIME', 'bLIMEy (logistic regression)', AQUA),
          ('null explainer', None, MUTED)]
INSTRUMENTS = [('fidelity\ntest data', 'fidelity (local)', 'test', False),
               ('fidelity\nlocal sample', 'fidelity (local)', 'local', False),
               ('Brier', 'Brier score (local)', 'local', True),
               ('KL', 'KL divergence (local)', 'local', True),
               ('cosine to\nthe truth', None, None, False)]


def logit(p, lim=12):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.clip(np.log(p/(1 - p)), -lim, lim)


def unit_max(v):
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else float('nan')


r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
names = list(test['feature_names'])
qs, _ = get_points_between_class_means(test)
q = np.asarray(qs[QUERY_INDEX], dtype=float)
truth = gradients.grad_logit(clf, MODEL, q[None, :])[0]

expls, coefs = {}, {}
for label, key, _ in [(a, b, c) for a, b, c in SERIES]:
    expls[label] = (surrogates.null_explainer(clf, q, test) if key is None
                    else clime.explainer.AVAILABLE_EXPLAINERS[key](
                        clf, q, train_data=train, test_data=test))
    coefs[label] = np.asarray(expls[label].get_explanation(), dtype=float)

# the readings, from the same surrogate objects the picture is drawn from
metrics = clime.evaluation.AVAILABLE_EVALUATION_METRICS
eval_data = {'local': get_local_points(test, q), 'test': test}
table = {}
for label, _, _ in [(a, b, c) for a, b, c in SERIES]:
    row = {}
    for head, key, where, _ in INSTRUMENTS:
        row[head] = (cosine(coefs[label], truth) if key is None
                     else float(metrics[key](expls[label], black_box_model=clf,
                                             data=eval_data[where], query_point=q)))
    table[label] = row

# the transect: the query-point line through q, out to where the locality weight has
# fallen to a tenth, so the picture covers what the metrics actually weight
direction = np.asarray(qs[-1]) - np.asarray(qs[0])
direction = direction/np.linalg.norm(direction)
lim = np.sqrt(len(q))*costs.KERNEL_WIDTH_SCALE*np.sqrt(-2.0*np.log(KERNEL_MASS))
t = np.linspace(-lim, lim, 500)
line = q[None, :] + t[:, None]*direction[None, :]
p_bb = np.asarray(clf.predict_proba(line))[:, 1]
kernel = costs.weights_based_on_distance(q, line)
kernel = kernel/kernel.max()


def crossing(p):
    '''where the curve passes 1/2, in transect units; nan if it never does'''
    s = np.sign(p - 0.5)
    i = np.nonzero(np.diff(s) != 0)[0]
    return float(t[i[0]]) if len(i) else float('nan')


fig = plt.figure(figsize=(7.2, 4.1))
gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0], width_ratios=[1.0, 1.2],
                      hspace=0.62, wspace=0.28)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
       fig.add_subplot(gs[1, :])]

# ---- (a) what the threshold looks at -------------------------------------------------
ax = axs[0]
ax.fill_between(t, -0.08, -0.08 + 0.22*kernel, color='#dfe8f4', lw=0, zorder=1)
ax.axhline(0.5, color=MUTED, lw=0.7, ls='--', zorder=2)
ax.plot(t, p_bb, color=INK, lw=3.0, label='black box $f$', zorder=3)
crossings = {'black box': crossing(p_bb)}
for label, _, colour in SERIES:
    p = np.asarray(expls[label].predict_proba(line))[:, 1]
    ax.plot(t, p, color=colour, lw=1.5, label=label, zorder=4)
    crossings[label] = crossing(p)
ax.axvline(0, color=MUTED, lw=0.6, ls=':', zorder=1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-0.08, 1.06)
ax.set_xlabel('distance from $q$')
ax.set_ylabel('probability  $p(y{=}1\\,|\\,x)$')
ax.set_title('(a) what the threshold sees', fontsize=8.5, color=INK)
finite = [v for k, v in crossings.items() if np.isfinite(v)]
if len(finite) > 1:
    spread = max(finite) - min(finite)
    ax.annotate(f'every boundary within {spread:.2f}\nof the black box\'s own',
                xy=(0.04, 0.72), xycoords='axes fraction',
                fontsize=6.2, color=INK_2, ha='left', va='center')
ax.annotate('the null explainer has no\nboundary to cross at all',
            xy=(0.97, 0.40), xycoords='axes fraction', fontsize=6.2, color=MUTED,
            ha='right', va='center')

# ---- (b) the explanation that comes out ----------------------------------------------
ax = axs[1]
order = list(np.argsort(-np.abs(truth))[:N_FEATURES])
extra = [int(np.argmax(np.abs(c))) for c in coefs.values() if np.any(c)]
order += [i for i in dict.fromkeys(extra) if i not in order]
ypos = np.arange(len(order))[::-1]
h = 0.19
ax.barh(ypos + 1.5*h, unit_max(truth)[order], height=h, color=INK,
        label='black box (truth)')
for j, (label, _, colour) in enumerate(SERIES[:3]):
    ax.barh(ypos + (0.5 - j)*h, unit_max(coefs[label])[order], height=h, color=colour,
            label=label)
ax.set_yticks(ypos)
ax.set_yticklabels([names[i] for i in order], fontsize=6.0)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel('weight, scaled to unit maximum')
ax.set_title('(b) the explanation that comes out', fontsize=8.5, color=INK)
ax.grid(axis='y', visible=False)

# ---- (c) what each instrument says ---------------------------------------------------
# a table rather than a plot: five quantities in four different units cannot share an
# axis, and the point is the numbers themselves
ax = axs[2]
ax.axis('off')
heads = [h for h, _, _, _ in INSTRUMENTS]
n_cols, n_rows = len(SERIES), len(heads)
LEFT = 1.35                  # room for the instrument names, in column widths
for j, (label, _, colour) in enumerate(SERIES):
    ax.text(LEFT + j + 0.5, n_rows + 0.15, label, ha='center', va='bottom', fontsize=6.8,
            color=colour)
for i, (head, _, _, lower) in enumerate(INSTRUMENTS):
    y = n_rows - 1 - i
    ax.text(LEFT - 0.12, y + 0.5, head.replace('\n', ' '), ha='right', va='center',
            fontsize=6.6, color=INK_2)
    col = [table[s][head] for s, _, _ in SERIES]
    finite_col = [c for c in col if np.isfinite(c)]
    for j, (label, _, _) in enumerate(SERIES):
        v = table[label][head]
        if not np.isfinite(v):
            txt = '--'
        elif head.startswith('Brier') or head.startswith('KL'):
            txt = f'{v:.1e}'.replace('e-0', r'e$-$')
        else:
            txt = f'{v:.3f}'
        # shade by rank within the row, so "all the same" looks like "all the same"
        rank = (sorted(finite_col, reverse=not lower).index(v)
                if np.isfinite(v) and finite_col else len(finite_col))
        ax.add_patch(plt.Rectangle((LEFT + j, y), 1, 1, facecolor=BLUE,
                                   alpha=0.22*(len(finite_col) - rank)/max(len(finite_col), 1),
                                   lw=0))
        ax.text(LEFT + j + 0.5, y + 0.5, txt, ha='center', va='center', fontsize=7.0,
                color=INK)
ax.set_xlim(0, LEFT + n_cols + 0.05)
ax.set_ylim(-0.05, n_rows + 0.9)
ax.set_title('(c) what each instrument says about them', fontsize=8.5, color=INK,
             loc='left', x=0.0, pad=12)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.035), handlelength=1.6, columnspacing=1.5)
fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.10)
fig.savefig(paths.fig('fig_blind.pdf'))
fig.savefig(paths.fig('fig_blind.png'))

print(f'fig_blind written   {DATASET} | {MODEL} point {QUERY_INDEX}, '
      f'f(q) = {np.asarray(clf.predict_proba(q[None, :]))[0, 1]:.4f}')
print(f"{'':18s} " + ' '.join(f'{h.replace(chr(10), " "):>18s}' for h in heads))
for label, _, _ in SERIES:
    print(f'{label:<18s} ' + ' '.join(f'{table[label][h]:>18.4g}' for h in heads))
print('  boundary crossings: ' + ', '.join(f'{k} {v:+.3f}' for k, v in crossings.items()))
json.dump({'config': f'{DATASET}|{MODEL}', 'index': QUERY_INDEX,
           'readings': table, 'crossings': crossings},
          open(paths.results('example_readings.json'), 'w'), indent=1)
