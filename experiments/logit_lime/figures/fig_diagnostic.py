'''
Figure 2 - the diagnostic, and the negative control.
(a) log-odds linearity gap vs Logit-LIME's advantage: strong relationship.
(b) probability saturation vs the same advantage: none. Calibrating a random forest
    removes saturation entirely without moving the gap or the advantage.

Points are coloured by the log-odds group registered in PREREGISTRATION.md, so the same
colours carry across to fig_groups.

usage:  python fig_diagnostic.py [results_taxonomy.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import numpy as np
from scipy.stats import spearmanr
from analysis.analyse import load, degenerate, GROUP_ORDER

# same ordinal ramp as fig_groups and fig_kernel
GROUP_COLOUR = {'A linear': '#104281', 'B quadratic': '#256abf',
                'C smooth': '#3987e5', 'D piecewise constant': '#86b6ef',
                'E calibrated forest': ORANGE, 'unassigned': MUTED}
GROUP_LEGEND = {'A linear': 'A  linear', 'B quadratic': 'B  quadratic',
                'C smooth': 'C  smooth', 'D piecewise constant': 'D  piecewise const.',
                'E calibrated forest': 'E  calibrated forest',
                'unassigned': 'gradient boosting'}
SHORT = {'Logistic': 'Logistic', 'LDA': 'LDA', 'QDA': 'QDA',
         'Gaussian Naive Bayes': 'NB', 'MLP': 'MLP', 'SVM': 'SVM',
         'Decision Tree': 'Tree', 'Random Forest': 'RF', 'k Nearest Neighbours': 'kNN',
         'Gradient Boosting': 'GBoost',
         'Random Forest (Platt calibrated)': 'RF+Platt',
         'Random Forest (isotonic calibrated)': 'RF+iso'}

rows, _ = load(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json'))
groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in rows)]

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1))
for ax, tag, xkey, xlabel in [
        (axs[0], '(a)', 'gap', 'log-odds linearity gap   $R^2_{\\mathrm{logit}} - R^2_{p}$'),
        (axs[1], '(b)', 'sat', 'fraction of saturated probabilities')]:
    ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
    # Degenerate configurations have no meaningful gap, so panel (a) drops them - from the
    # scatter as well as from rho, since a gap of -108 would flatten the x axis and the
    # plotted points must be the points the statistic is computed on. Panel (b) keeps
    # them: saturation is well defined for them, and they are exactly the heavily
    # saturated points that panel is about.
    ok = rows if xkey == 'sat' else [r for r in rows if r not in degenerate(rows)]
    for g in groups:
        sub = [r for r in ok if r['group'] == g]
        ax.scatter([r[xkey] for r in sub], [r['adv'] for r in sub], s=17,
                   facecolor=GROUP_COLOUR[g], edgecolor='white', linewidth=0.4,
                   alpha=0.85, zorder=3, label=GROUP_LEGEND[g])
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    rho, pv = spearmanr([r[xkey] for r in ok], [r['adv'] for r in ok])
    # panel letter folded into the title: as a separate text it collides with it
    ax.set_title(f"{tag}   Spearman $\\rho$ = {rho:.2f}   "
                 f"($p$ = {pv:.1g},  $n$ = {len(ok)})", color=INK, pad=6, fontsize=8.5)

axs[0].set_ylabel('Logit-LIME advantage\n(Brier ratio, $>1$ is better)')
axs[1].tick_params(labelleft=False)
# Clamp to the informative range. Degenerate configurations reach an advantage of 1.5e11
# (and 0, which a log axis cannot show at all) purely because both surrogates are fitting
# a constant; left unclamped they compress every real result into a band. Points outside
# are drawn at the edge and counted in the caption rather than dropped.
ADV_LIM = (10**-1.2, 10**7)
n_clamped = int(sum(not (ADV_LIM[0] <= r['adv'] <= ADV_LIM[1]) for r in rows))
for ax in axs:
    ax.set_ylim(*ADV_LIM)
for ax, xkey in [(axs[0], 'gap'), (axs[1], 'sat')]:
    shown = rows if xkey == 'sat' else [r for r in rows if r not in degenerate(rows)]
    outside = [r for r in shown if not (ADV_LIM[0] <= r['adv'] <= ADV_LIM[1])]
    ax.scatter([r[xkey] for r in outside],
               [np.clip(r['adv'], *ADV_LIM) for r in outside],
               s=17, facecolor=[GROUP_COLOUR[r['group']] for r in outside],
               edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
print(f'clamped to axis edge: {n_clamped}')
axs[0].annotate('break-even', xy=(0.99, 1.6), xycoords=('axes fraction', 'data'),
                fontsize=7, color=MUTED, ha='right')

# the calibration triple is the negative control - call it out on panel (b). One dataset
# only: the whole point is the horizontal move at fixed advantage, which a cloud hides.
tri = {r['model']: r for r in rows if r['dataset'] == 'Gaussian'
       and r['model'].startswith('Random Forest')}
offs = {'Random Forest': (6, 3), 'Random Forest (Platt calibrated)': (7, 4),
        'Random Forest (isotonic calibrated)': (6, -9)}
for model, r in tri.items():
    axs[1].annotate(SHORT[model], (r['sat'], r['adv']), fontsize=6.5, color=INK_2,
                    xytext=offs.get(model, (6, 3)), textcoords='offset points', va='center')
if 'Random Forest' in tri and 'Random Forest (Platt calibrated)' in tri:
    axs[1].annotate(f"Platt calibration moves the random forest\n"
                    f"from {tri['Random Forest']['sat']:.0%} saturation to "
                    f"{tri['Random Forest (Platt calibrated)']['sat']:.0%} without\n"
                    f"changing its benefit",
                    xy=(0.97, 0.72), xycoords='axes fraction', fontsize=6.5, color=INK_2,
                    va='top', ha='right')

axs[0].legend(loc='upper left', fontsize=6.5, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25, scatterpoints=1)
fig.tight_layout(w_pad=1.2)
os.makedirs('figs', exist_ok=True)
fig.savefig(paths.fig('fig2_diagnostic.pdf'))
fig.savefig(paths.fig('fig2_diagnostic.png'))
print(f'fig2 written  (n = {len(rows)})')
