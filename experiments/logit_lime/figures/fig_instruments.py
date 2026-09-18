'''
What each instrument can see, with no fit in the loop (-> figs/fig_instruments.pdf).

Reads results_instruments.json.  Rows are the three ways a surrogate can be wrong -
direction, slope, confidence - and columns are query points from the decision boundary out
to f(q) = 0.99.

Six instruments in six different units cannot share an axis, so each curve is scaled to
put 1 at the best reading that instrument gives anywhere in its ROW and 0 at the worst.
Scaling per row rather than per panel is what makes the first row readable: an instrument
whose excursion shrinks from left to right is one that stops being able to see a wrong
direction as the query point leaves the decision boundary, and normalising each panel
separately would have drawn all three as the same curve.  A flat line at the top is an
instrument that cannot tell the truth from any of the errors on that axis at all.

The two rows to read together are the last two.  `slope` and `sharpen` multiply the
surrogate's coefficients by the same factor; the difference is that `slope` leaves
g(q) = f(q) - which slides the surrogate's class boundary - while `sharpen` multiplies the
whole log-odds and leaves the boundary where it is.  Fidelity responds to the first and is
exactly constant under the second, which is the point: it is not measuring the slope at
all, only where the boundary fell.

usage:  python figures/fig_instruments.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *

import json
import numpy as np

CONFIG = 'Gaussian|Logistic'          # the drawable one; the table carries the 30-D case

# grouped by family: the two threshold-at-0.5 cells, the two rank based instruments, the
# two proper scoring rules
STYLE = {
    'fidelity | local sample': dict(color=BLUE, ls='-'),
    'fidelity | test data':    dict(color=BLUE, ls=(0, (4, 1.6))),
    'fidelity at f(q)':        dict(color=AQUA, ls='-'),
    'Spearman':                dict(color=AQUA, ls=(0, (4, 1.6))),
    'Brier':                   dict(color=ORANGE, ls='-'),
    'KL':                      dict(color=ORANGE, ls=(0, (4, 1.6))),
}
LABEL = {'fidelity at f(q)': 'fidelity at $f(q)$'}     # legend text, where it differs
AXIS_LABEL = {
    'theta': ('rotation of the explanation  $\\theta$ (degrees)',
              'direction:  $\\cos\\theta$ is the cosine to the truth'),
    'scale': ('slope multiplier  $s$', 'slope only:  $g(q)=f(q)$ kept, boundary moves'),
    'sharpen': ('log-odds multiplier  $s$',
                'confidence:  boundary fixed, sharpness changes'),
}
FLAT_TOL = 0.0       # "flat" means exactly flat here, and it is


def goodness(values, lower_is_better, reference):
    '''
    the curve on a common 1 = best, 0 = worst scale, with the scale taken from
    `reference` - every value that instrument produces anywhere in the row.

    An instrument that does not move at all has no range to normalise by.  That is not a
    degenerate case to be hidden - it is the result - so it is drawn as a flat line at 1:
    every member of the family, truth and errors alike, is the best reading it has.
    '''
    sign = -1.0 if lower_is_better else 1.0
    v = sign*np.asarray(values, dtype=float)
    ref = sign*np.asarray(reference, dtype=float)
    lo, hi = np.nanmin(ref), np.nanmax(ref)
    flat = not np.isfinite(lo) or hi - lo <= FLAT_TOL
    if flat:
        return np.ones_like(v), True
    return (v - lo)/(hi - lo), False


d = json.load(open(paths.results('results_instruments.json')))
meta = d['_meta']
cfg = d[CONFIG]
theta = np.array(meta['thetas_deg'])
scales = np.array(meta['scales'])
lower = set(meta['lower_is_better'])
axes_order = ['theta', 'scale', 'sharpen']
x_of = {'theta': theta, 'scale': scales, 'sharpen': scales}

fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.4), sharey=True)

for row, axis in enumerate(axes_order):
    # the common scale for this row: everything that instrument reads in it
    reference = {name: np.concatenate([p['curves'][name][axis] for p in cfg['points']])
                 for name in STYLE}
    for col, point in enumerate(cfg['points']):
        ax = axs[row, col]
        flat_here = []
        for name in STYLE:
            y, flat = goodness(point['curves'][name][axis], name in lower,
                               reference[name])
            ax.plot(x_of[axis], y, label=LABEL.get(name, name),
                    lw=2.0 if flat else 1.5, **STYLE[name])
            if flat:
                flat_here.append(name)
        # the truth sits at theta = 0 / s = 1 on every axis
        ax.axvline(0 if axis == 'theta' else 1, color=MUTED, lw=0.7, ls=':')
        if axis != 'theta':
            ax.set_xscale('log')
            ax.set_xticks([0.1, 1, 10])
            ax.set_xticklabels(['0.1', '1', '10'])
        else:
            ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_ylim(-0.06, 1.16)
        if flat_here:
            ax.annotate('no change' if len(flat_here) == 2 else 'no change',
                        xy=(0.5, 1.03), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=6.4,
                        color=BLUE if len(flat_here) == 2 else AQUA)
        if row == 0:
            ax.set_title(f"$f(q) = {point['f_q']:.2f}$", fontsize=8.5, color=INK, pad=4)
        if col == 1:
            ax.set_xlabel(AXIS_LABEL[axis][0])
        if col == 0:
            ax.set_ylabel('reading  (1 = best)')
            ax.annotate(AXIS_LABEL[axis][1], xy=(0.0, 1.34), xycoords='axes fraction',
                        fontsize=8, color=INK_2, ha='left', va='center',
                        annotation_clip=False)
fig.tight_layout(rect=(0, 0.055, 1, 0.955), h_pad=2.6)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.012), handlelength=2.0, columnspacing=1.5)

fig.savefig(paths.fig('fig_instruments.pdf'))
fig.savefig(paths.fig('fig_instruments.png'))
print('fig_instruments written')
for axis in axes_order:
    for point in cfg['points']:
        flat = [n for n in STYLE
                if np.ptp(np.asarray(point['curves'][n][axis], dtype=float)) == 0.0]
        print(f"  {axis:8s} f(q)={point['f_q']:.3f}  exactly flat: "
              f"{', '.join(flat) if flat else '-'}")
