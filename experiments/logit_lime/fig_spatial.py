'''
Figure 3 - where along the decision surface the advantage lives.
Local Brier score at each query point on the line between the class means.
'''
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import json, numpy as np

d = json.load(open('results.json'))
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
    ax.set_title(nice, color=INK)
    ax.set_xlabel('query point along the line')
    ax.set_xticks([0, 5, 10, 15, 19])
axs[0].set_ylabel('local Brier score\n(lower is better)')
axs[0].legend(loc='upper left', handlelength=1.4, borderpad=0.2)
fig.tight_layout(w_pad=1.2)
fig.savefig('figs/fig3_spatial.pdf'); fig.savefig('figs/fig3_spatial.png')
print('fig3 written')
