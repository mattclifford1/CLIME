'''
Does a fidelity score predict a correct explanation? (-> figs/fig_fidelity_explanation.pdf)

The study measures Brier and KL. A user reads feature importances. The link between them
is an assumption, and results_gradient_truth.json is the only place it can be tested,
because there both quantities are recorded from the same surrogate at the same query
point and a true importance vector exists to score against.

The answer is a qualified yes, and the two panels separate the parts that hold from the
part that does not.

  (a) level - across surrogates and query points, a lower divergence goes with an
      explanation closer to the truth. This is the claim the study needs.

  (b) magnitude - the size of the fidelity gain does not predict the size of the
      explanation gain. Almost every configuration sits in the quadrant where both
      favour Logit-LIME, but within that quadrant the relationship is slightly negative:
      the configurations where logit space wins by orders of magnitude on Brier are the
      exactly-linear ones, where the standard surrogate's explanation was already close.

So fidelity is a valid proxy for ranking two surrogates and a poor one for predicting how
much better the explanation will be. Where a ground truth exists, score the explanation.

usage:  python figures/fig_fidelity_explanation.py [results_gradient_truth.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *

import numpy as np
from scipy.stats import spearmanr
from analysis.analyse_gradient_truth import load, BRIER, KL

METRIC = KL          # the better of the two proxies; Brier is reported in the text
NICE = 'KL divergence (local)'

GROUP_COLOUR = {'A linear': BLUE, 'B quadratic': ORANGE, 'C smooth': AQUA,
                'unassigned': MUTED}
GROUP_NICE = {'A linear': 'A exactly linear', 'B quadratic': 'B quadratic',
              'C smooth': 'C smooth', 'unassigned': 'unassigned'}

rows = load(sys.argv[1] if len(sys.argv) > 1 else None)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) level: every surrogate at every query point --------------------------------
ax = axs[0]
xs, ys = [], []
for label, colour, nice in (('standard', BLUE, 'standard LIME'),
                            ('logit', ORANGE, 'Logit-LIME')):
    x = np.array([p[label][METRIC] for r in rows for p in r['points']])
    y = np.array([p[label]['cos'] for r in rows for p in r['points']])
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = np.log10(x[ok]), y[ok]
    xs.append(x)
    ys.append(y)
    ax.scatter(x, y, s=3, facecolor=colour, alpha=0.22, linewidth=0, label=nice,
               rasterized=True)
rho = spearmanr(np.concatenate(xs), np.concatenate(ys))

# The cosine distribution piles up against 1, so a cloud of points shows nothing and a
# running median sits on the ceiling. Decile bins of the fidelity score, with the median
# and the interquartile range of the cosine in each, show the trend and how wide it is.
allx, ally = np.concatenate(xs), np.concatenate(ys)
edges = np.quantile(allx, np.linspace(0, 1, 11))
mid, med, lo, hi = [], [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (allx >= a) & (allx <= b)
    if np.sum(m) < 10:
        continue
    mid.append(np.median(allx[m]))
    med.append(np.median(ally[m]))
    lo.append(np.quantile(ally[m], 0.25))
    hi.append(np.quantile(ally[m], 0.75))
ax.fill_between(mid, lo, hi, color='#d8d7d3', lw=0, zorder=4, label='interquartile range')
ax.plot(mid, med, color=INK, lw=1.8, marker='o', markersize=3.2, zorder=5,
        label='median per decile')
ax.set_xlabel(f'$\\log_{{10}}$ {NICE}   (lower is a better fit)')
ax.set_ylabel('cosine to the true local importances')
ax.set_title('(a) better fit, better explanation', fontsize=8.5, color=INK)
ax.set_ylim(-0.35, 1.08)
ax.annotate(f'Spearman $\\rho = {rho.statistic:+.2f}$\n$n = {len(allx):,}$',
            xy=(0.03, 0.05), xycoords='axes fraction', fontsize=7, color=INK_2,
            va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                                   pad=1.6))
ax.legend(loc='lower right', fontsize=6.5, markerscale=1.6, handletextpad=0.4,
          borderpad=0.3)

# ---- (b) magnitude: paired, per configuration ---------------------------------------
ax = axs[1]
d_fid, d_cos, colours = [], [], []
for r in rows:
    fa, fb = r[f'{METRIC}_standard'], r[f'{METRIC}_logit']
    ca, cb = r['cos_standard'], r['cos_logit']
    if min(fa, fb) <= 0 or not all(np.isfinite(v) for v in (fa, fb, ca, cb)):
        continue
    d_fid.append(np.log10(fa) - np.log10(fb))
    d_cos.append(cb - ca)
    colours.append(GROUP_COLOUR.get(r['group'], MUTED))
d_fid, d_cos = np.array(d_fid), np.array(d_cos)
rho_p = spearmanr(d_fid, d_cos)

ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
ax.scatter(d_fid, d_cos, s=13, facecolor=colours, edgecolor='white', linewidth=0.3,
           alpha=0.9, zorder=3)
both = int(np.sum((d_fid > 0) & (d_cos > 0)))
ax.annotate(f'both favour\nLogit-LIME\n{both} of {len(d_fid)}', xy=(0.97, 0.95),
            xycoords='axes fraction', ha='right', va='top', fontsize=6.8, color=INK_2)
ax.annotate(f'Spearman $\\rho = {rho_p.statistic:+.2f}$', xy=(0.03, 0.05),
            xycoords='axes fraction', fontsize=7, color=INK_2, va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.6))
ax.set_xlabel('fidelity gain   $\\log_{10}$ ratio of divergences')
ax.set_ylabel('explanation gain   $\\Delta$ cosine')
ax.set_title('(b) but the sizes do not track', fontsize=8.5, color=INK)
handles = [plt.Line2D([], [], marker='o', ls='', markersize=4,
                      markerfacecolor=GROUP_COLOUR[g], markeredgecolor='white',
                      label=GROUP_NICE[g]) for g in GROUP_COLOUR]
ax.legend(handles=handles, loc='lower right', fontsize=6.3, handletextpad=0.3,
          borderpad=0.3)

fig.tight_layout(w_pad=1.6)
fig.savefig(paths.fig('fig_fidelity_explanation.pdf'))
fig.savefig(paths.fig('fig_fidelity_explanation.png'))
print(f'pooled rho {rho.statistic:+.3f} (n={len(allx)})   '
      f'paired rho {rho_p.statistic:+.3f} (n={len(d_fid)})   '
      f'both-favour quadrant {both}/{len(d_fid)}')
print('fig_fidelity_explanation written')
