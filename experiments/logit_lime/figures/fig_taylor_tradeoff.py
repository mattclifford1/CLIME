'''
The price of an explanation-optimal surrogate (-> figs/fig_taylor_tradeoff.pdf).

The first-order Taylor expansion of the black box's log-odds at q has the true local
importances as its coefficients, by construction. Panels (a) and (b) put the two
objectives side by side, so what it costs can be read across them: the Taylor surrogate is
best on explanation everywhere, and that is free only where the log-odds are linear. In
groups B and C it buys the explanation with a worse fit, which is what the second
pre-registration predicted.

Fidelity is shown relative to Logit-LIME within each group rather than absolutely. The
groups differ by sixteen orders of magnitude in absolute KL - in group A every surrogate
reproduces the black box almost exactly - so on a shared absolute axis the trade, which
happens within a group, is invisible.

Panel (c) is the part a practitioner can use: the same gradient estimated from 2d
black-box queries, needing no more access than LIME, recovers almost all of the
explanation accuracy except where the black box saturates and a difference quotient of the
log-odds has nothing to measure.

usage:  python figures/fig_taylor_tradeoff.py [results_taylor.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths
from common.style import *

import numpy as np
from analysis.analyse_taylor import load, floored, KL, METHODS

GROUPS = [('A linear', 'A\nlinear'), ('B quadratic', 'B\nquadratic'),
          ('C smooth', 'C\nsmooth')]
SERIES = [('standard', BLUE, 'standard LIME'),
          ('logit', ORANGE, 'Logit-LIME'),
          ('taylor_fd', AQUA, 'Taylor (finite diff.)'),
          ('taylor', INK, 'Taylor (analytic)')]

rows = load(sys.argv[1] if len(sys.argv) > 1 else None)
subs = [(nice, [r for r in rows if r['group'] == g]) for g, nice in GROUPS]
x = np.arange(len(subs))
w = 0.2

fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75),
                        gridspec_kw=dict(width_ratios=[1, 1, 0.85]))

# ---- (a) explanation accuracy --------------------------------------------------------
ax = axs[0]
for i, (key, colour, label) in enumerate(SERIES):
    vals = [np.nanmean([r[f'cos_{key}'] for r in sub]) for _, sub in subs]
    ax.bar(x + (i - 1.5)*w, vals, width=w, color=colour, label=label)
ax.set_xticks(x)
ax.set_xticklabels([n for n, _ in subs], fontsize=7)
ax.set_ylim(0.7, 1.04)
ax.set_ylabel('cosine to the true\nlocal importances')
ax.set_title('(a) explanation: higher is better', fontsize=8.5, color=INK)
ax.grid(axis='x', visible=False)

# ---- (b) fidelity, relative to Logit-LIME within each group --------------------------
ax = axs[1]


def kl_ratio(sub, key):
    '''median over configurations of KL(key)/KL(Logit-LIME), paired within each'''
    a = floored([r[f'{KL}_{key}'] for r in sub])
    b = floored([r[f'{KL}_logit'] for r in sub])
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.median(a[ok]/b[ok]))


for i, (key, colour, label) in enumerate(SERIES):
    vals = [kl_ratio(sub, key) for _, sub in subs]
    ax.bar(x + (i - 1.5)*w, vals, width=w, color=colour, label=label)
ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=3)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([n for n, _ in subs], fontsize=7)
ax.set_ylabel('local KL divergence,\nrelative to Logit-LIME')
ax.set_title('(b) fidelity: lower is better', fontsize=8.5, color=INK)
ax.grid(axis='x', visible=False)
ax.annotate('no trade: the expansion\nis the black box', xy=(0.62, 1e-9),
            fontsize=6.0, color=INK_2, ha='left', va='center')
# the trade in B and C is a factor of ~1.5 on an axis spanning fifteen decades, so label it
for gi, (_, sub) in enumerate(subs):
    if gi == 0:
        continue
    v = kl_ratio(sub, 'taylor')
    ax.annotate(f'{v:.1f}$\\times$', xy=(gi + 1.5*w, v), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=6.2, color=INK_2)

# ---- (c) the practical version, and where it fails -----------------------------------
ax = axs[2]
sat, unsat, degen = [], [], []
for r in rows:
    for p in r['points']:
        (sat if p['saturated'] else unsat).append(p['taylor_fd']['cos'])
        degen.append(p['taylor_fd']['degenerate'])
sat, unsat = np.array(sat, dtype=float), np.array(unsat, dtype=float)
n_q = np.mean([p['n_queries_fd'] for r in rows for p in r['points']])

bars = [('std.\nLIME', np.nanmean([r['cos_standard'] for r in rows]), BLUE),
        ('logit\nLIME', np.nanmean([r['cos_logit'] for r in rows]), ORANGE),
        ('Taylor\nunsat.', np.nanmean(unsat), AQUA),
        ('Taylor\nsat.', np.nanmean(sat), '#a3453b')]
xs = np.arange(len(bars))
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.66)
for xi, (_, v, _) in zip(xs, bars):
    ax.annotate(f'{v:.3f}', xy=(xi, v), xytext=(0, 3), textcoords='offset points',
                ha='center', fontsize=6.6, color=INK_2)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=6.5)
ax.set_ylim(0.7, 1.06)
ax.set_ylabel('cosine to the true\nlocal importances')
ax.set_title('(c) black-box access only', fontsize=8.5, color=INK)
ax.grid(axis='x', visible=False)

fig.tight_layout(w_pad=1.5, rect=(0, 0.09, 1, 1))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.6)

fig.savefig(paths.fig('fig_taylor_tradeoff.pdf'))
fig.savefig(paths.fig('fig_taylor_tradeoff.png'))
print(f'unsaturated {np.nanmean(unsat):.4f}  saturated {np.nanmean(sat):.4f}  '
      f'degenerate {np.mean(degen):.1%}  queries {n_q:.0f}')
print('fig_taylor_tradeoff written')
