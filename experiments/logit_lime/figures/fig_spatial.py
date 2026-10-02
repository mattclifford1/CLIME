'''
Figure 3 - where along the decision surface the advantage lives.
Local Brier score at each query point on the line between the class means.

The soft-label logistic surrogate (sixth registration) is read from
results_soft_logistic_full.json, which scores it on the same evaluation points; dashed,
because on a linear black box it lies on top of Logit-LIME.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

from common.style import *
import json, numpy as np

import sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json')))
soft = json.load(open(paths.results('results_soft_logistic_full.json')))
PANELS = [('Gaussian|Logistic', 'Logistic regression'),
          ('Gaussian|MLP', 'MLP'),
          ('Gaussian|Random Forest', 'Random forest')]

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5), sharex=True)
for ax, (key, nice) in zip(axs, PANELS):
    m = d[key]['metrics']['Brier score (local)']
    n = np.array(m['bLIMEy (normal)']['scores'])
    l = np.array(m['bLIMEy (logit)']['scores'])
    x = np.arange(len(n))
    ax.plot(x, n, color=BLUE, marker='o', markersize=3, markeredgecolor='white',
            markeredgewidth=0.5, label='standard LIME')
    ax.plot(x, l, color=ORANGE, marker='o', markersize=3, markeredgecolor='white',
            markeredgewidth=0.5, label='Logit-LIME')
    # refuse a results file from another seed or split: the soft-label line would be drawn
    # beside neighbourhoods it was not fitted on
    assert abs(n[0] - soft[key]['check_standard_brier_q0']) <= 1e-9*max(abs(n[0]), 1e-12), \
        f'{key}: results_soft_logistic_full.json does not join to this file'
    s_ = np.array([pt['Brier | local sample'] for pt in soft[key]['points']])
    ax.plot(x, s_, color=AQUA, ls='--', marker='o', markersize=2.4, markeredgecolor='white',
            markeredgewidth=0.4, label='soft-label logistic', zorder=5)
    ax.set_title(nice, color=INK)
    ax.set_xlabel('query point along the line')
    ax.set_xticks([0, 5, 10, 15, 19])
axs[0].set_ylabel('local Brier score\n(lower is better)')
axs[0].legend(loc='upper left', handlelength=1.4, borderpad=0.2)
fig.tight_layout(w_pad=1.2)
fig.savefig(paths.fig('fig3_spatial.pdf')); fig.savefig(paths.fig('fig3_spatial.png'))
print('fig3 written')
