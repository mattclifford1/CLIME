'''
The diagnostic across slices of the full grid: Spearman ρ with Logit-LIME's advantage, for
R²_logit and for the registered gap Δ, with dataset-level bootstrap intervals. Reads
results/analysis_robustness.json (analysis/analyse_robustness.py).

usage:  python figures/fig_robustness.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import json
import numpy as np

slices = json.load(open(paths.results('analysis_robustness.json')))
slices = [s for s in slices if s['n'] > 2]

# one row per slice, a gap between axes
ys, labels, axis_heads, y = [], [], [], 0
last = None
for s in slices:
    if s['axis'] != last:
        if last is not None:
            y += 0.6
        axis_heads.append((y, s['axis']))
        last = s['axis']
    ys.append(y)
    labels.append(f"{s['label']}  (n={s['n']})")
    y += 1
ys = np.array(ys)

fig, ax = plt.subplots(figsize=(6.9, 0.19*len(ys) + 1.0))
OFF = 0.17
for key, colour, dy, name in [('r2_logit', BLUE, -OFF, '$R^2_{\\mathrm{logit}}$'),
                              ('gap', ORANGE, +OFF, '$\\Delta = R^2_{\\mathrm{logit}} - R^2_{p}$')]:
    rho = np.array([s[f'rho_{key}'] for s in slices], float)
    lo = np.array([s[f'ci_{key}'][0] for s in slices], float)
    hi = np.array([s[f'ci_{key}'][1] for s in slices], float)
    ax.hlines(ys + dy, lo, hi, color=colour, lw=1.4, zorder=2)
    ax.scatter(rho, ys + dy, s=26, color=colour, edgecolor='white', linewidth=0.8,
               zorder=3, label=name)

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=7)
for yh, head in axis_heads:
    # the axis name sits at the figure's left edge, clear of the tick labels
    ax.annotate(head, xy=(0.01, yh), xycoords=('figure fraction', 'data'), ha='left',
                va='center', fontsize=7, color=INK, fontweight='bold',
                annotation_clip=False)
ax.invert_yaxis()
ax.set_xlim(0.2, 1.0)
ax.set_xlabel('Spearman $\\rho$ with Logit-LIME\'s advantage  (95% interval over datasets)')
ax.legend(loc='lower left', fontsize=7, frameon=True, framealpha=0.95, borderpad=0.3)
ax.grid(axis='x', color='#e5e4e0', lw=0.6)
fig.tight_layout()
fig.subplots_adjust(left=0.36)
fig.savefig(paths.fig('fig_robustness.pdf'))
fig.savefig(paths.fig('fig_robustness.png'))
print(f'written fig_robustness ({len(ys)} slices)')
