'''
Figure 1 - the mechanism.
A transect through the black box's decision surface, shown in probability space and in
logit space, with both surrogates overlaid. Two black boxes: one whose log-odds are
linear in x (logistic regression) and one whose are not (random forest).
'''
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np, warnings, clime
from clime.evaluation.key_points import get_points_between_class_means
warnings.filterwarnings('ignore')

train, test = clime.data.AVAILABLE_DATASETS['Gaussian'](
    class_samples=[200, 200], gaussian_means=[[-1, -1], [1, 1]],
    gaussian_covs=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
norm = clime.data.normaliser(train); train, test = norm(train), norm(test)
qs, _ = get_points_between_class_means(test)
q = np.array(qs[11])                      # just off the decision boundary

direction = (np.array(qs[-1]) - np.array(qs[0]))
direction = direction/np.linalg.norm(direction)
t = np.linspace(-2.6, 2.6, 400)
line = q[None, :] + t[:, None]*direction[None, :]

def logit(p, lim=8):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.clip(np.log(p/(1-p)), -lim, lim)

MODELS = [('Logistic regression', 'Logistic'), ('Random forest', 'Random Forest')]
fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.4), sharex=True)

for col, (nice, model) in enumerate(MODELS):
    clf = clime.models.AVAILABLE_MODELS[model](train)
    p_bb = clf.predict_proba(line)[:, 1]
    e_std = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (normal)'](
        clf, query_point=q, train_data=train, test_data=test)
    e_log = clime.explainer.AVAILABLE_EXPLAINERS['bLIMEy (logit)'](
        clf, query_point=q, train_data=train, test_data=test)
    raw_std = e_std.surrogate_model.predict(line)[:, 1]     # unclipped: shows it leaving [0,1]
    p_std = np.clip(raw_std, 0, 1)
    p_log = e_log.predict_proba(line)[:, 1]

    ax = axs[0, col]
    ax.axhspan(-0.35, 0, color='#f2f1ed', zorder=0); ax.axhspan(1, 1.35, color='#f2f1ed', zorder=0)
    ax.plot(t, p_bb, color=INK, lw=3.4, label='black box $f$', zorder=3)
    ax.plot(t, raw_std, color=BLUE, ls=':', lw=1.3, zorder=2)
    ax.plot(t, p_std, color=BLUE, lw=1.6, label='standard LIME', zorder=4)
    ax.plot(t, p_log, color=ORANGE, lw=1.6, label='Logit-LIME', zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_ylim(-0.35, 1.35); ax.set_title(nice, color=INK)
    ax.text(0.06, 1.20, '$q$', fontsize=8, color=MUTED)
    if col == 0:
        ax.set_ylabel('probability  $p(y{=}1\\,|\\,x)$')
        ax.legend(loc='upper left', handlelength=1.4, borderpad=0.2)
        ax.annotate('unbounded: outside $[0,1]$', xy=(-2.45, -0.28), fontsize=7, color=INK_2,
                    bbox=dict(facecolor='#f2f1ed', edgecolor='none', alpha=0.9, pad=1.2))

    ax = axs[1, col]
    ax.plot(t, logit(p_bb), color=INK, lw=3.4, zorder=3)
    ax.plot(t, logit(p_std), color=BLUE, lw=1.6, zorder=4)
    ax.plot(t, logit(p_log), color=ORANGE, lw=1.6, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.set_xlabel('position along the transect through $q$')
    ax.set_ylim(-9.6, 9.6)
    note = ('log-odds are linear in $x$:\nLogit-LIME recovers $f$ exactly'
            if col == 0 else 'log-odds are a step function:\nneither surrogate fits')
    ax.annotate(note, xy=(0.03, 0.80), xycoords='axes fraction', fontsize=7, color=INK_2,
                va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.6))
    if col == 0:
        ax.set_ylabel('log-odds  $\\mathrm{logit}\\,p$')

fig.align_ylabels()
fig.tight_layout(w_pad=1.6, h_pad=0.8)
fig.savefig('figs/fig1_mechanism.pdf'); fig.savefig('figs/fig1_mechanism.png')
print('fig1 written')
