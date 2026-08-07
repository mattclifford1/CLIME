'''
Figure 2 - the diagnostic, and the negative control.
(a) log-odds linearity gap vs Logit-LIME's advantage: strong relationship.
(b) probability saturation vs the same advantage: none. Calibrating a random forest
    removes saturation entirely without moving the gap or the advantage.
'''
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import json, numpy as np
from scipy.stats import spearmanr

d = json.load(open('results.json'))
DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes']
MARKERS = {'Gaussian': 'o', 'Breast Cancer': 's',
           'Banknote Authentication': '^', 'Pima Indian Diabetes': 'D'}
SHORT = {'Logistic': 'Logistic', 'MLP': 'MLP', 'SVM': 'SVM',
         'Gradient Boosting': 'GBoost', 'Random Forest': 'RF',
         'Random Forest (Platt calibrated)': 'RF+Platt',
         'Random Forest (isotonic calibrated)': 'RF+iso'}
LABEL_ME = {'Logistic', 'MLP', 'Random Forest',
            'Random Forest (Platt calibrated)', 'Random Forest (isotonic calibrated)'}

rows = []
for key, e in d.items():
    dataset, model = key.split('|')
    m = e['metrics']['Brier score (local)']
    rows.append(dict(dataset=dataset, model=model,
                     gap=e['diagnostic']['gap'], sat=e['diagnostic']['saturation'],
                     adv=m['bLIMEy (normal)']['mean']/max(m['bLIMEy (logit)']['mean'], 1e-12)))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.1))
for ax, xkey, xlabel in [(axs[0], 'gap', 'log-odds linearity gap   $R^2_{\\mathrm{logit}} - R^2_{p}$'),
                         (axs[1], 'sat', 'fraction of saturated probabilities')]:
    ax.axhline(1, color=MUTED, lw=0.8, ls='--', zorder=1)
    for r in rows:
        ax.scatter(r[xkey], r['adv'], marker=MARKERS[r['dataset']], s=30,
                   facecolor=BLUE, edgecolor='white', linewidth=0.7, alpha=0.9, zorder=3)
    ax.set_yscale('log'); ax.set_xlabel(xlabel)
    rho, pv = spearmanr([r[xkey] for r in rows], [r['adv'] for r in rows])
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f}   ($p$ = {pv:.1g})", color=INK, pad=6)

axs[0].set_ylabel('Logit-LIME advantage\n(Brier ratio, $>1$ is better)')
axs[1].tick_params(labelleft=False)
lo = min(r['adv'] for r in rows)*0.30; hi = max(r['adv'] for r in rows)*3
for ax in axs:
    ax.set_ylim(lo, hi)
axs[0].annotate('break-even', xy=(0.99, 1.35), xycoords=('axes fraction', 'data'),
                fontsize=7, color=MUTED, ha='right')

# direct labels only where they can be read - the cluster at the origin is left unlabelled
# and described in the caption instead
for r in rows:
    if r['gap'] > 0.35 or r['adv'] > 100:
        axs[0].annotate(SHORT[r['model']], (r['gap'], r['adv']), fontsize=6.5, color=INK_2,
                        xytext=(5, -1), textcoords='offset points', va='center')
# the calibration triple is the negative control - call it out on panel (b)
tri = {r['model']: r for r in rows if r['dataset'] == 'Gaussian'
       and r['model'].startswith('Random Forest')}
offs = {'Random Forest': (6, 3), 'Random Forest (Platt calibrated)': (7, 4),
        'Random Forest (isotonic calibrated)': (6, -9)}
for model, r in tri.items():
    axs[1].annotate(SHORT[model], (r['sat'], r['adv']), fontsize=6.5, color=INK_2,
                    xytext=offs[model], textcoords='offset points', va='center')
axs[1].annotate('Platt calibration moves the random forest\n'
                'from 66% saturation to 0% without\nchanging its benefit',
                xy=(0.97, 0.72), xycoords='axes fraction', fontsize=6.5, color=INK_2,
                va='top', ha='right')

handles = [plt.Line2D([], [], marker=MARKERS[ds], ls='', color=BLUE, markersize=5,
                      markeredgecolor='white', label=ds) for ds in DATASETS]
axs[0].legend(handles=handles, loc='upper left', fontsize=6.5, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)
axs[0].text(-0.01, 1.06, '(a)', transform=axs[0].transAxes, fontsize=9, color=INK)
axs[1].text(-0.01, 1.06, '(b)', transform=axs[1].transAxes, fontsize=9, color=INK)
fig.tight_layout(w_pad=1.2)
fig.savefig('figs/fig2_diagnostic.pdf'); fig.savefig('figs/fig2_diagnostic.png')
print('fig2 written')
