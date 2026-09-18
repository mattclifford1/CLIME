'''
What a coefficient claims, and how far it can be carried (-> figs/fig_reading.pdf).

The rest of the paper measures how well a surrogate reproduces the black box.  This figure
is about the number the user is actually handed: one coefficient per feature, presented as
"how much this feature matters".  Standard LIME's is a probability per unit feature, and
that reading has three defects which are visible without any appeal to fit quality.

  (a) It expires inside its own neighbourhood.  A linear model of a probability is a
      probability only in a slab of width 1/||beta|| about its own boundary.  Walk out of
      that slab - a distance shorter than the locality kernel's width, here - and the
      surrogate claims a number that is not a probability.  The logit surrogate has no
      such edge to fall off: its p = 0.99 isoline sits at about the same distance where
      the standard one hits 1, and it still has infinitely far left to go.

  (b) It does not start where the black box is, and it is a chord rather than a tangent,
      so "the probability rises by beta_j per unit" has neither an agreed base value nor a
      step size over which it is true.

  (c) It conflates two things a user needs separately.  Along the line of query points the
      black box's log-odds slope is constant - the features matter exactly as much at
      every point - yet the standard coefficient falls by two orders of magnitude as the
      prediction saturates.  A reader cannot tell "this feature does not matter" from "the
      prediction is already certain".  In logit space the slope carries the first and the
      intercept the second.

This is an argument about the reported number, not about fidelity.  Repairing the range is
NOT where the fidelity benefit comes from - the SVM column of Figure 1 makes that point -
and the two must not be conflated.

usage:  python figures/fig_reading.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients
from common.style import *

import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

LOGIT = r'\mathrm{logit}'          # main.tex's \logit macro is not mathtext

DATASET, MODEL = 'Gaussian', 'Logistic'
QUERY_INDEX = 11          # the same neighbourhood as Figure 1; f(q) ~ 0.85
HALF = 2.6                # half width of panel (a), in standardised units
WALK = 3.0                # panel (b) walks this far along one feature, in sd
SAT_EPS = 1e-9            # the squash bound of logit_ridge: p outside this is saturated
SAT_SHADE = 0.10          # shade panel (c) where this much of the kernel mass saturates


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p/(1 - p))


def surrogate_logodds(expl, X):
    '''logit g for the logit surrogate, read off the ridge before the sigmoid'''
    m = expl.surrogate_model
    return float(m.intercept_) + np.atleast_2d(X) @ np.atleast_2d(m.coef_)[-1, :]


def build(clf, train, test, q):
    E = clime.explainer.AVAILABLE_EXPLAINERS
    return (E['bLIMEy (normal)'](clf, query_point=q, train_data=train, test_data=test),
            E['bLIMEy (logit)'](clf, query_point=q, train_data=train, test_data=test))


def neighbourhood(test, q):
    '''
    the surrogate's OWN training sample, regenerated.

    Deliberately the same draw site as bLIMEy._sample_locally - same salt - so the mass
    reported below is mass the surrogate was actually fitted on, not a fresh sample that
    merely resembles it.
    '''
    rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
    X = rng.multivariate_normal(q, np.cov(test['X'].T), 10000)
    return X, costs.weights_based_on_distance(q, X)


# ------------------------------------------------------------------------ the numbers

r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                parallel_eval=False)
clf, train, test = r['clf'], r['train_data'], r['test_data']
names = list(test['feature_names'])
qs, _ = get_points_between_class_means(test)
qs = np.asarray(qs, dtype=float)
q = qs[QUERY_INDEX]
d = q.size
K = costs.KERNEL_WIDTH_SCALE*np.sqrt(d)          # the locality kernel width, Eq. (kernel)

truth = gradients.grad_logit(clf, MODEL, q)[0]
J = int(np.argmax(np.abs(truth)))                # the feature panel (b) walks along

e_std, e_log = build(clf, train, test, q)
b_std = np.asarray(e_std.get_explanation(), dtype=float)
b_log = np.asarray(e_log.get_explanation(), dtype=float)
f_q = float(clf.predict_proba(q[None, :])[0, 1])
g_q = float(e_std.surrogate_model.predict(q[None, :])[0, 1])      # unclipped
l_q = float(surrogate_logodds(e_log, q[None, :])[0])

n_std = np.linalg.norm(b_std)
u_std = b_std/n_std                              # direction the coefficient points
r_up = (1 - g_q)/n_std                           # distance until the claim exceeds 1
r_down = g_q/n_std                               # distance until it falls below 0

Xn, wn = neighbourhood(test, q)
g_n = e_std.surrogate_model.predict(Xn)[:, 1]
mass_hi = float(np.sum(wn*(g_n > 1))/np.sum(wn))
mass_lo = float(np.sum(wn*(g_n < 0))/np.sum(wn))

assert abs(f_q - 0.85) < 0.05, f'query point {QUERY_INDEX} moved: f(q)={f_q:.3f}'

# ------------------------------------------------------- panel (c): along the query line
line_std, line_log, line_truth, line_tangent, line_sat = [], [], [], [], []
for qi in qs:
    ei_std, ei_log = build(clf, train, test, qi)
    ti = gradients.grad_logit(clf, MODEL, qi)[0]
    fi = float(clf.predict_proba(qi[None, :])[0, 1])
    line_std.append(np.linalg.norm(np.asarray(ei_std.get_explanation(), dtype=float)))
    line_log.append(np.linalg.norm(np.asarray(ei_log.get_explanation(), dtype=float)))
    line_truth.append(np.linalg.norm(ti))
    line_tangent.append(fi*(1 - fi)*np.linalg.norm(ti))
    Xi, wi = neighbourhood(test, qi)
    pi = np.asarray(clf.predict_proba(Xi))[:, 1]
    line_sat.append(float(np.sum(wi*((pi < SAT_EPS) | (pi > 1 - SAT_EPS)))/np.sum(wi)))
line_std, line_log = np.array(line_std), np.array(line_log)
line_truth, line_tangent = np.array(line_truth), np.array(line_tangent)
line_sat = np.array(line_sat)

# position along the query line, measured from where the black box crosses p = 0.5, so the
# horizontal axis is a distance in the feature space rather than an index
f_line = np.asarray(clf.predict_proba(qs))[:, 1]
step = qs - qs[0]
s_line = np.linalg.norm(step, axis=1)*np.sign(np.sum(step*(qs[-1] - qs[0]), axis=1))
cross = np.interp(0.0, logit(f_line), s_line)
s_line = s_line - cross

# ------------------------------------------------------------------------- the figure

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.75),
                        gridspec_kw=dict(width_ratios=[1, 1.05, 1.2]))

# ---- (a) where the surrogate is a probability at all ---------------------------------
ax = axs[0]
gx, gy = np.meshgrid(np.linspace(q[0] - HALF, q[0] + HALF, 300),
                     np.linspace(q[1] - HALF, q[1] + HALF, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])
surf = clf.predict_proba(grid)[:, 1].reshape(gx.shape)
g_grid = e_std.surrogate_model.predict(grid)[:, 1].reshape(gx.shape)
l_grid = surrogate_logodds(e_log, grid).reshape(gx.shape)

# grey is "not a probability" here and in panel (b), so the two panels can be read
# together: everything outside the surrogate's own slab 0 <= g <= 1 is shaded, and the
# white band is the whole region in which its output can be read as a probability at all
ax.set_facecolor('#f2f1ed')
ax.contourf(gx, gy, g_grid, levels=[0.0, 1.0], colors=['white'], zorder=0)
ax.contour(gx, gy, surf, levels=[0.5], colors=[INK], linewidths=0.9, zorder=2)
ax.contour(gx, gy, g_grid, levels=[0, 1], colors=[BLUE], linewidths=1.2, zorder=3)
ax.contour(gx, gy, l_grid, levels=[logit(0.01), logit(0.99)], colors=[ORANGE],
           linewidths=1.2, linestyles=[(0, (4, 1.6))], zorder=3)
# the locality kernel the surrogate was fitted with: iso-weight rings at w = 0.5 and 0.1
for w_level in (0.5, 0.1):
    radius = K*np.sqrt(-2*np.log(w_level))
    ax.add_patch(plt.Circle(q, radius, fill=False, edgecolor=MUTED, lw=0.7,
                            ls=(0, (2, 2)), zorder=4))
ax.annotate('', xy=q + r_up*u_std, xytext=q, zorder=6,
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2, shrinkA=0, shrinkB=0))
ax.scatter([q[0]], [q[1]], s=30, facecolor=AQUA, edgecolor=INK, linewidth=0.8, zorder=7)

tag = dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.0)
ax.annotate(f'$g>1$:  {mass_hi:.0%}\nof the weight', xy=(0.96, 0.94),
            xycoords='axes fraction', fontsize=6.3, color=BLUE, ha='right', va='top',
            bbox=tag, zorder=8)
ax.annotate(f'$g<0$:  {mass_lo:.0%}', xy=(0.04, 0.06), xycoords='axes fraction',
            fontsize=6.3, color=BLUE, ha='left', va='bottom', bbox=tag, zorder=8)
# Logit-LIME's p = 0.99 isoline sits almost exactly on the standard surrogate's g = 1 -
# the two orange dashes track the two blue lines - so the comparison goes in the caption
# rather than in a label that would have to point at both at once
ax.annotate(f'$g=1$ at\n{r_up:.2f} sd $={r_up/K:.2f}\\,k$', xy=q + r_up*u_std,
            xytext=(7, -3), textcoords='offset points', fontsize=6.3, color=INK,
            ha='left', va='top', bbox=tag, zorder=8)
ax.annotate('$q$', xy=q + np.array([-0.52, -0.48]), fontsize=8, color=INK, zorder=8)
ax.annotate('kernel', xy=(q[0] - 1.42, q[1] + 1.42), fontsize=6.3, color=MUTED,
            ha='center', bbox=tag, zorder=8)
ax.set_xlim(q[0] - HALF, q[0] + HALF)
ax.set_ylim(q[1] - HALF, q[1] + HALF)
ax.set_aspect('equal')
ax.set_xlabel('$x_0$', fontsize=8)
ax.set_ylabel('$x_1$', fontsize=8)
ax.set_title('(a) where the claim is a probability', fontsize=8.5, color=INK)
ax.grid(False)

# ---- (b) carrying the coefficient ----------------------------------------------------
ax = axs[1]
t = np.linspace(-WALK, WALK, 601)
walk = np.repeat(q[None, :], t.size, axis=0)
walk[:, J] += t
p_bb = clf.predict_proba(walk)[:, 1]
raw = e_std.surrogate_model.predict(walk)[:, 1]
p_log = 1/(1 + np.exp(-surrogate_logodds(e_log, walk)))
w_walk = costs.weights_based_on_distance(q, walk)
w_walk = w_walk/w_walk.max()

ax.axhspan(-0.42, 0, color='#f2f1ed', zorder=0)
ax.axhspan(1, 1.68, color='#f2f1ed', zorder=0)
ax.fill_between(t, -0.42, -0.42 + 0.30*w_walk, color='#dfe8f4', lw=0, zorder=1)
ax.plot(t, p_bb, color=INK, lw=3.0, label='black box $f$', zorder=3)
ax.plot(t, raw, color=BLUE, ls=':', lw=1.2, zorder=2)
ax.plot(t, np.clip(raw, 0, 1), color=BLUE, lw=1.6, label='standard LIME', zorder=4)
ax.plot(t, p_log, color=ORANGE, lw=1.6, label='Logit-LIME', zorder=4)
ax.axvline(0, color=MUTED, lw=0.6, ls='--', zorder=1)
ax.axhline(1, color=MUTED, lw=0.6, zorder=1)

steps = np.array([0, 1, 2])
claim = g_q + b_std[J]*steps
actual = np.asarray(clf.predict_proba(
    q[None, :] + np.outer(steps, np.eye(d)[J])))[:, 1]
ax.scatter(steps, claim, s=15, facecolor='white', edgecolor=BLUE, linewidth=0.9, zorder=6)
ax.scatter(steps, actual, s=15, facecolor='white', edgecolor=INK, linewidth=0.9, zorder=6)
# the claimed value goes above its marker and the true one below, except at the step where
# the two are close enough to collide - there the arrow annotation carries both
for s, c, a in zip(steps, claim, actual):
    if s > 0:      # at q the arrow annotation below carries both numbers
        ax.annotate(f'{c:.2f}', xy=(s, c), xytext=(0, 6), textcoords='offset points',
                    fontsize=6.4, color=BLUE, ha='center')
        ax.annotate(f'{a:.3f}', xy=(s, a), xytext=(0 if s == 1 else 5, -11),
                    textcoords='offset points', fontsize=6.4, color=INK_2,
                    ha='center' if s == 1 else 'left')
ax.annotate(f'claims {g_q:.2f} at $q$,\nwhere $f$ is {f_q:.2f}', xy=(0, g_q),
            xytext=(0.55, 0.30), fontsize=6.4, color=INK_2, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6,
                            connectionstyle='arc3,rad=0.25', shrinkB=4))
ax.annotate('not a probability', xy=(0.5, 0.955), xycoords='axes fraction', fontsize=6.4,
            color=INK_2, ha='center', va='top')
ax.annotate('locality weight', xy=(0.5, 0.012), xycoords='axes fraction', fontsize=6.3,
            color='#5b7fa8', ha='center')
ax.set_ylim(-0.42, 1.68)
ax.set_xlim(-WALK, WALK)
ax.set_xlabel('steps along $x_1$  (sd)', fontsize=8)
ax.set_ylabel('probability  $p(y{=}1\\,|\\,x)$', fontsize=8.5)
ax.set_title('(b) carrying it one feature at a time', fontsize=8.5, color=INK)

# ---- (c) the size of the number reported ---------------------------------------------
ax = axs[2]
# where the black box saturates to logit_ridge's squash bound over enough of the
# neighbourhood, the logit fit starts to bend and its coefficient drifts; shade those
# points so the drift is not read as a property of the method. On this configuration the
# set is empty - the drift at the ends of the line is smaller than that - which is why
# the figure carries no band and the caption quotes 7%.
sat = line_sat > SAT_SHADE
for i in np.where(sat)[0]:
    half = 0.5*np.min(np.diff(s_line))
    ax.axvspan(s_line[i] - half, s_line[i] + half, color='#f2f1ed', lw=0, zorder=0)
ax.plot(s_line, line_std, color=BLUE, lw=1.6, marker='o', ms=2.6, zorder=4)
ax.plot(s_line, line_tangent, color=MUTED, lw=1.0, ls=(0, (1, 1.6)), zorder=3)
ax.set_ylabel('probability per sd', fontsize=8, color=BLUE)
ax.tick_params(axis='y', colors=BLUE)
ax.set_ylim(-0.04, 1.62)

ax2 = ax.twinx()
ax2.plot(s_line, line_log, color=ORANGE, lw=1.6, marker='o', ms=2.6, zorder=4)
ax2.plot(s_line, line_truth, color=INK, lw=1.3, ls=(0, (4, 1.6)), zorder=5)
ax2.set_ylabel('log-odds per sd', fontsize=8, color=ORANGE)
ax2.tick_params(axis='y', colors=ORANGE)
ax2.set_ylim(-0.12, 4.86)
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(MUTED)

# four curves in two unit systems: a legend box would need the whole panel, so each is
# labelled where it runs and the axis colours say which scale it belongs to
tag = dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.0)
ax2.annotate('Logit-LIME, on the truth', xy=(s_line[5], line_truth[5]),
             xytext=(0, -13), textcoords='offset points', fontsize=6.4, color=ORANGE,
             ha='left', bbox=tag, zorder=8)
ax.annotate('standard LIME', xy=(s_line[7], line_std[7]), xytext=(-5, 7),
            textcoords='offset points', fontsize=6.4, color=BLUE, ha='right', bbox=tag,
            zorder=8)
ax.annotate('tangent at $q$', xy=(s_line[11], line_tangent[11]), xytext=(8, 2),
            textcoords='offset points', fontsize=6.4, color=INK_2, ha='left', bbox=tag,
            zorder=8)
ax.annotate(f'falls {line_std.max()/line_std.min():,.0f}$\\times$ while\n'
            'the truth does not move', xy=(0.02, 0.60), xycoords='axes fraction',
            fontsize=6.4, color=INK_2, ha='left', va='top', bbox=tag, zorder=8)

ax.set_xlabel('query point:  distance from the boundary (sd)', fontsize=8)
ax.set_title('(c) what the size of the number tracks', fontsize=8.5, color=INK)

fig.tight_layout(w_pad=1.6, rect=(0, 0.085, 1, 1))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handlelength=1.6, columnspacing=1.8)

fig.savefig(paths.fig('fig_reading.pdf'))
fig.savefig(paths.fig('fig_reading.png'))

# ------------------------------------------------------- the numbers the caption quotes
readings = {
    'dataset': DATASET, 'model': MODEL, 'query_index': QUERY_INDEX,
    'feature': names[J], 'kernel_width': K,
    'f_q': f_q, 'g_std_q': g_q, 'g_logit_q': float(1/(1 + np.exp(-l_q))),
    'coef_std': float(b_std[J]), 'coef_logit': float(b_log[J]),
    'truth': float(truth[J]),
    'norm_std': float(n_std), 'norm_logit': float(np.linalg.norm(b_log)),
    'norm_truth': float(np.linalg.norm(truth)),
    'r_exit_up': float(r_up), 'r_exit_up_over_k': float(r_up/K),
    'r_exit_down': float(r_down),
    'mass_above_1': mass_hi, 'mass_below_0': mass_lo,
    'walk_claimed': [float(v) for v in claim],
    'walk_actual': [float(v) for v in actual],
    'span_std': float(line_std.max()/line_std.min()),
    'span_logit': float(line_log.max()/line_log.min()),
    'span_logit_unsaturated': float(line_log[~sat].max()/line_log[~sat].min()),
}
json.dump(readings, open(paths.results('reading_readings.json'), 'w'), indent=1)

print(f'fig_reading written   {DATASET} | {MODEL} | point {QUERY_INDEX}  '
      f'({names[J]}, d={d}, k={K:.2f})')
print(f'  f(q) = {f_q:.3f}   standard g(q) = {g_q:.3f}   Logit-LIME g(q) = '
      f'{1/(1+np.exp(-l_q)):.3f}')
print(f'  coefficient on {names[J]}:  standard {b_std[J]:.3f} prob/sd   '
      f'logit {b_log[J]:.3f} log-odds/sd   truth {truth[J]:.3f}')
print(f'  claim leaves [0,1] at {r_up:.2f} sd above ({r_up/K:.2f} k) and '
      f'{r_down:.2f} sd below')
print(f'  kernel mass of its own training sample outside [0,1]: '
      f'{mass_hi:.1%} above 1, {mass_lo:.1%} below 0')
print('  walk: ' + '  '.join(f'{s:+d}sd claimed {c:.2f} actual {a:.4f}'
                             for s, c, a in zip(steps, claim, actual)))
print(f'  ||beta|| span over the 20 query points: standard '
      f'{line_std.max()/line_std.min():.0f}x   Logit-LIME '
      f'{line_log.max()/line_log.min():.2f}x '
      f'({line_log[~sat].max()/line_log[~sat].min():.2f}x unsaturated)')
print(f'  saturated query points (>{SAT_SHADE:.0%} of kernel mass): {int(sat.sum())} of 20')
