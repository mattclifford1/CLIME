'''
Figure: the pre-registered test.

Logit-LIME's benefit against the a priori geometry of the black box's log-odds. The
grouping was registered before the sweep was run (PREREGISTRATION.md); the ordering
A > B > C > D is the prediction, not a post-hoc fit.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import numpy as np
from analysis.analyse import load, GROUP_ORDER

# ordinal ramp: darker = larger predicted effect. Steps 650/500/400/250 of the blue
# sequential ramp - the lightest clears the 2:1 ordinal floor on a light surface.
GROUP_COLOUR = {'A linear': '#104281', 'B quadratic': '#256abf',
                'C smooth': '#3987e5', 'D piecewise constant': '#86b6ef',
                # E is a registered group, so it gets its own hue rather than the grey
                # reserved for the one black box left deliberately unassigned
                'E calibrated forest': ORANGE, 'unassigned': MUTED}
# single letters on the axis; the caption spells the groups out, which keeps six
# categories legible in a two panel figure
SHORT = {'A linear': 'A', 'B quadratic': 'B', 'C smooth': 'C',
         'D piecewise constant': 'D', 'E calibrated forest': 'E', 'unassigned': 'GB'}

rows, _ = load(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json'))
groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in rows)]

# Both quantities have long tails driven by near-degenerate neighbourhoods, where the
# black box is almost constant and R^2_logit is set by where the probabilities were
# clipped rather than by any geometry. Those points are plotted at the axis edge and
# counted in the caption instead of being allowed to set the scale for everything else.
ADV_LIM = (10**-0.6, 10**7)
GAP_LIM = (-0.35, 0.78)


def clamp(vals, lim):
    v = np.asarray(vals, dtype=float)
    out = np.clip(v, *lim)
    return out, int(np.sum(v < lim[0]) + np.sum(v > lim[1]) + np.sum(~np.isfinite(v)))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.2),
                        gridspec_kw={'width_ratios': [1.25, 1]})

# (a) benefit by group
ax = axs[0]
ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
rng = np.random.default_rng(0)
n_out_adv = 0
for i, g in enumerate(groups):
    vals = np.array([r['adv'] for r in rows if r['group'] == g])
    shown, n_out = clamp(vals, ADV_LIM)
    n_out_adv += n_out
    jitter = rng.uniform(-0.17, 0.17, len(vals))
    ax.scatter(i + jitter, shown, s=22, facecolor=GROUP_COLOUR[g], edgecolor='white',
               linewidth=0.6, alpha=0.9, zorder=3)
    med = np.nanmedian(vals)
    ax.plot([i-0.30, i+0.30], [med, med], color=INK, lw=2.0, zorder=4,
            solid_capstyle='butt')
ax.set_yscale('log')
ax.set_ylim(*ADV_LIM)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([SHORT[g] for g in groups], fontsize=8)
ax.set_ylabel('Logit-LIME advantage\n(Brier ratio, $>1$ is better)')
ax.set_xlabel('predicted log-odds geometry of the black box')
ax.set_xlim(-0.6, len(groups)-0.4)
ax.annotate('break-even', xy=(0.99, 1.7), xycoords=('axes fraction', 'data'),
            fontsize=7, color=MUTED, ha='right')
ax.text(-0.01, 1.04, '(a)', transform=ax.transAxes, fontsize=9, color=INK)

# (b) the diagnostic separates the same groups
ax = axs[1]
n_out_gap = 0
for i, g in enumerate(groups):
    vals = np.array([r['gap'] for r in rows if r['group'] == g])
    shown, n_out = clamp(vals, GAP_LIM)
    n_out_gap += n_out
    jitter = rng.uniform(-0.17, 0.17, len(vals))
    ax.scatter(i + jitter, shown, s=22, facecolor=GROUP_COLOUR[g], edgecolor='white',
               linewidth=0.6, alpha=0.9, zorder=3)
    med = np.nanmedian(vals)
    ax.plot([i-0.30, i+0.30], [med, med], color=INK, lw=2.0, zorder=4,
            solid_capstyle='butt')
ax.axhline(0, color=MUTED, lw=0.8, ls='--', zorder=1)
ax.set_ylim(*GAP_LIM)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([SHORT[g] for g in groups], fontsize=8)
ax.set_ylabel('log-odds linearity gap  $R^2_{\\mathrm{logit}} - R^2_{p}$')
ax.set_xlabel('predicted log-odds geometry')
ax.set_xlim(-0.6, len(groups)-0.4)
ax.text(-0.01, 1.04, '(b)', transform=ax.transAxes, fontsize=9, color=INK)

print(f'clamped to axis edge: {n_out_adv} advantage, {n_out_gap} gap')
fig.tight_layout(w_pad=1.4)
os.makedirs('figs', exist_ok=True)
fig.savefig(paths.fig('fig_groups.pdf'))
fig.savefig(paths.fig('fig_groups.png'))
print('fig_groups written')
