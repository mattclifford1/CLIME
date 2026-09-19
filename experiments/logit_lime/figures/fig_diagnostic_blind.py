'''
The diagnostic out of sample: Figure 2(a) repeated on the 42 datasets added by the fifth
registration, which the choice of R²_logit had not seen, next to the one measure that needs
no fit at all.

(a) 1 - R²_logit (guarded, sweep.guarded_r2) against Logit-LIME's advantage, new datasets
    only, coloured by registered group as in Figure 2
(b) the locality-weighted relative dispersion of grad logit f over the neighbourhood
    (check C3, sweep_diagnostic_checks.py) against the same advantage, over the whole full
    grid, for the black boxes with an analytic gradient. Group A sits at exactly zero and is
    drawn at the floor

usage:  python figures/fig_diagnostic_blind.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import json
import numpy as np
from scipy.stats import spearmanr
from analysis import diagnostic_stats as ds
from sweeps import full_grid

GROUP_COLOUR = {'A linear': '#104281', 'B quadratic': '#256abf',
                'C smooth': '#3987e5', 'D piecewise constant': '#86b6ef',
                'E calibrated forest': ORANGE, 'unassigned': MUTED}
GROUP_LEGEND = {'A linear': 'A  linear', 'B quadratic': 'B  quadratic',
                'C smooth': 'C  smooth', 'D piecewise constant': 'D  piecewise const.',
                'E calibrated forest': 'E  calibrated forest', 'unassigned': 'unassigned'}
ADV_LIM = (10**-1.2, 10**7)
FLOOR = 1e-5

full = ds.load(paths.results('results_full.json'))
new, _ = ds.usable([r for r in full if r['dataset'] in full_grid.NEW], 'guarded')
adv = {(r['dataset'], r['model']): r['adv'] for r in full}

ck = json.load(open(paths.results('results_diagnostic_checks.json')))
disp = []
for k, v in ck.items():
    if k.startswith('_') or 'error' in v or not np.isfinite(v.get('grad_dispersion', np.nan)):
        continue
    # the same exclusion as panel (a): f constant to rounding over most neighbourhoods
    if v.get('n_defined_r2_logit', 0) < ds.MIN_DEFINED:
        continue
    dset, model = k.split('|')
    if (dset, model) in adv:
        disp.append(dict(group=v['group'], x=max(v['grad_dispersion'], FLOOR),
                         adv=adv[(dset, model)]))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
panels = [(axs[0], '(a)', [dict(group=r['group'], x=max(1 - r['r2_logit'], FLOOR), adv=r['adv'])
                            for r in new],
           'unexplained log-odds variance  $1 - R^2_{\\mathrm{logit}}$', '42 new datasets'),
          (axs[1], '(b)', disp, 'dispersion of $\\nabla\\,\\mathrm{logit}\\,f$ over the neighbourhood',
           'differentiable, all 71')]
for ax, tag, pts, xlabel, what in panels:
    ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
    for g in GROUP_COLOUR:
        sub = [p for p in pts if p['group'] == g]
        if not sub:
            continue
        ax.scatter([p['x'] for p in sub], [np.clip(p['adv'], *ADV_LIM) for p in sub], s=14,
                   facecolor=GROUP_COLOUR[g], edgecolor='white', linewidth=0.4, alpha=0.85,
                   zorder=3, label=GROUP_LEGEND[g])
    rho = spearmanr([p['x'] for p in pts], [p['adv'] for p in pts])[0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(*ADV_LIM)
    ax.set_xlabel(xlabel)
    ax.set_title(f'{tag}  {what}:  $\\rho$ = {rho:.2f},  $n$ = {len(pts)}', color=INK,
                 pad=6, fontsize=8)
axs[0].axvline(0.05, color=MUTED, lw=0.8, ls=':', zorder=1)
axs[0].set_ylabel('Logit-LIME advantage\n(Brier ratio, $>1$ is better)')
axs[0].legend(loc='lower left', fontsize=6.5, handletextpad=0.3, borderpad=0.2,
              labelspacing=0.25, scatterpoints=1)
fig.tight_layout(w_pad=1.2)
fig.savefig(paths.fig('fig_diagnostic_blind.pdf'))
fig.savefig(paths.fig('fig_diagnostic_blind.png'))
print(f'written fig_diagnostic_blind (a: {len(panels[0][2])}, b: {len(disp)})')
