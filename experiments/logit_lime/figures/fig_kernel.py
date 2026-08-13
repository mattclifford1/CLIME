'''
Figure: is the effect an artefact of LIME's default kernel width?

The locality kernel width k = scale * sqrt(n_features) sits on both sides of the
experiment - it weights the surrogate's training points and it defines the neighbourhood
the metric scores on. This plots the advantage and the diagnostic against that scale.

usage:  python fig_kernel.py [results_kernel.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *
import json
import numpy as np
from analysis.analyse import GROUP_ORDER

# same ordinal ramp as fig_groups, so the groups read consistently across figures
GROUP_COLOUR = {'A linear': '#104281', 'B quadratic': '#256abf',
                'C smooth': '#3987e5', 'D piecewise constant': '#86b6ef',
                'E calibrated forest': MUTED, 'unassigned': MUTED}
GROUP_OF = {'Logistic': 'A linear', 'LDA': 'A linear', 'MLP': 'C smooth',
            'Random Forest': 'D piecewise constant', 'Gradient Boosting': 'unassigned'}
DEFAULT_SCALE = 0.75


def load(path):
    d = json.load(open(path))
    rows = []
    for key, v in d.items():
        if key.startswith('_') or 'error' in v:
            continue
        scale, dataset, model = key.split('|')
        std, logit = 'bLIMEy (normal)', 'bLIMEy (logit)'
        rows.append(dict(scale=float(scale), dataset=dataset, model=model,
                         group=GROUP_OF.get(model, 'unassigned'),
                         gap=v['diagnostic']['gap'],
                         adv=v[std]['mean']/max(v[logit]['mean'], 1e-30)))
    return rows


def median_by_scale(rows, model, field):
    '''median across datasets at each scale'''
    scales = sorted({r['scale'] for r in rows})
    xs, ys = [], []
    for s in scales:
        vals = [r[field] for r in rows if r['model'] == model and r['scale'] == s]
        if vals:
            xs.append(s)
            ys.append(np.median(vals))
    return np.array(xs), np.array(ys)


rows = load(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_kernel.json'))
models = [m for m in GROUP_OF if any(r['model'] == m for r in rows)]

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.0))

for ax, field, ylabel in [
        (axs[0], 'adv', 'Logit-LIME advantage\n(Brier ratio, $>1$ is better)'),
        (axs[1], 'gap', 'log-odds linearity gap\n$R^2_{\\mathrm{logit}} - R^2_{p}$')]:
    ax.axvline(DEFAULT_SCALE, color=MUTED, lw=0.8, ls=':', zorder=1)
    for m in models:
        xs, ys = median_by_scale(rows, m, field)
        ax.plot(xs, ys, color=GROUP_COLOUR[GROUP_OF[m]], marker='o', ms=3.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3, label=m)
    ax.set_xscale('log')
    # label the swept scales themselves; the default decade ticks show only 10^0 here
    scales = sorted({r['scale'] for r in rows})
    ax.set_xticks(scales)
    ax.set_xticklabels([f'{s:g}' for s in scales], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel('locality kernel width scale')
    ax.set_ylabel(ylabel)

axs[0].axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
axs[0].set_yscale('log')
axs[0].annotate('break-even', xy=(0.99, 1.3), xycoords=('axes fraction', 'data'),
                fontsize=7, color=MUTED, ha='right')
axs[0].annotate('LIME default', xy=(DEFAULT_SCALE, 1.02), xycoords=('data', 'axes fraction'),
                fontsize=7, color=MUTED, ha='center')
axs[1].axhline(0, color=MUTED, lw=0.8, ls='--', zorder=1)
axs[0].text(-0.01, 1.04, '(a)', transform=axs[0].transAxes, fontsize=9, color=INK)
axs[1].text(-0.01, 1.04, '(b)', transform=axs[1].transAxes, fontsize=9, color=INK)
axs[1].legend(loc='best', fontsize=7)

fig.tight_layout(w_pad=1.4)
os.makedirs('figs', exist_ok=True)
fig.savefig(paths.fig('fig_kernel.pdf'))
fig.savefig(paths.fig('fig_kernel.png'))
print('fig_kernel written')
