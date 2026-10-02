'''
Logit-LIME's model class fitted two ways (-> figs/fig_softlabel.pdf), sixth registration.

(a) Each configuration's advantage over standard LIME (ratio of local Brier scores) for
    Logit-LIME, least squares on a clipped logit, against the soft-label surrogate,
    cross-entropy against the probabilities. Above the diagonal the loss, not the model
    class, decides the gain.
(b) The soft-label surrogate's KL advantage over Logit-LIME against the fraction of
    saturated probabilities in the neighbourhood: what the clip costs, where it bites.

Every configuration of the full grid (71 datasets x 16 black boxes) is drawn, coloured by
registered group; the registered/blind split is the analysis script's job, not the
figure's.

usage:  python figures/fig_softlabel.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *

import numpy as np
from scipy.stats import spearmanr
from analysis.analyse_soft_logistic import rows, GROUP_ORDER
from sweeps import full_grid

# the ordinal group ramp of fig_diagnostic, fig_groups and fig_kernel
GROUP_COLOUR = {'A linear': '#104281', 'B quadratic': '#256abf',
                'C smooth': '#3987e5', 'D piecewise constant': '#86b6ef',
                'E calibrated forest': ORANGE, 'unassigned': MUTED}
GROUP_LEGEND = {'A linear': 'A  linear', 'B quadratic': 'B  quadratic',
                'C smooth': 'C  smooth', 'D piecewise constant': 'D  piecewise const.',
                'E calibrated forest': 'E  calibrated forest',
                'unassigned': 'unassigned'}
LO, HI = 1e-1, 1e7

all_rows, _ = rows()
data = [r for r in all_rows if r['model'] in full_grid.MODELS
        and all(np.isfinite(r.get(k, np.nan)) for k in ('std_brier', 'logit_brier', 'sat'))]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))

ax = axs[0]
ax.plot([LO, HI], [LO, HI], color=MUTED, lw=0.8, ls='--', zorder=1)
ax.axhline(1, color=MUTED, lw=0.6, ls=':', zorder=1)
ax.axvline(1, color=MUTED, lw=0.6, ls=':', zorder=1)
for g in GROUP_ORDER:
    sub = [r for r in data if r['group'] == g]
    if not sub:
        continue
    x = np.clip([r['std_brier']/max(r['logit_brier'], 1e-30) for r in sub], LO, HI)
    y = np.clip([r['std_brier']/max(r['soft_brier'], 1e-30) for r in sub], LO, HI)
    ax.scatter(x, y, s=11, facecolor=GROUP_COLOUR[g], edgecolor='white', linewidth=0.3,
               alpha=0.85, zorder=3, label=GROUP_LEGEND[g])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_xlabel('Logit-LIME: advantage over standard LIME')
ax.set_ylabel('soft-label: advantage over standard LIME')
above = np.mean([r['soft_brier'] < r['logit_brier'] for r in data])
ax.set_title(f'(a) the same model class, two losses', fontsize=8.5, color=INK)
ax.annotate(f'soft-label better\nat {above:.0%}', xy=(0.04, 0.96), xycoords='axes fraction',
            va='top', fontsize=6.8, color=INK_2)
ax.annotate('Logit-LIME better', xy=(0.96, 0.04), xycoords='axes fraction', ha='right',
            fontsize=6.8, color=INK_2)

ax = axs[1]
ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
sat = np.array([r['sat'] for r in data])
ratio = np.array([r['logit_kl']/max(r['soft_kl'], 1e-30) for r in data])
for g in GROUP_ORDER:
    m = np.array([r['group'] == g for r in data])
    if m.any():
        ax.scatter(sat[m], np.clip(ratio[m], 1e-3, 1e6), s=11, facecolor=GROUP_COLOUR[g],
                   edgecolor='white', linewidth=0.3, alpha=0.85, zorder=3)
rho = spearmanr(sat, ratio)[0]
ax.set_yscale('log')
ax.set_ylim(1e-3, 1e6)
ax.set_xlabel('fraction of saturated probabilities in the neighbourhood')
ax.set_ylabel("Logit-LIME's local KL / soft-label's")
ax.set_title('(b) what the clip costs', fontsize=8.5, color=INK)
ax.annotate(f'Spearman $\\rho = {rho:+.2f}$\n$n = {len(data):,}$', xy=(0.04, 0.96),
            xycoords='axes fraction', va='top', fontsize=6.8, color=INK_2,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.6))

# the legend goes under both panels: inside (a) it would cover the group A cloud
fig.tight_layout(w_pad=1.6, rect=(0, 0.08, 1, 1))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=7,
           handletextpad=0.3, columnspacing=1.2, markerscale=1.3, bbox_to_anchor=(0.5, -0.01))
fig.savefig(paths.fig('fig_softlabel.pdf'))
fig.savefig(paths.fig('fig_softlabel.png'))
print(f'fig_softlabel written: n = {len(data)}, soft better than Logit-LIME on Brier at '
      f'{above:.1%}, rho(saturation, KL ratio) {rho:+.3f}')
