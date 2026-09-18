'''
Table: what each coefficient claims, at four query points (-> tables/reading.tex).

Figure fig_reading.pdf shows the geometry at one query point.  This is the same reading
written out as numbers, at four points along the query line, so that the parts of the
argument which are about a trend rather than a picture can be checked:

  - the base value the reading starts from  (g(q) against f(q))
  - what it claims one and two standardised units along a single feature
  - how far it can be carried before it stops being a probability, against the width of
    the locality kernel that defined the neighbourhood in the first place
  - the counterfactual it implies: how far that feature must move to flip the decision

The last row is the one that grows.  A probability coefficient's implied flip distance is
correct near the decision boundary and wrong by a factor that grows with confidence, since
a chord through a sigmoid keeps a finite slope where the sigmoid itself is flattening.  The
log-odds coefficient gets it right wherever the black box's log-odds are linear, which for
this black box is everywhere.

Restricted to black boxes with an analytic gradient (common/gradients.py) - the truth rows
need one.  The flip row additionally needs f to be monotone along the feature within the
range searched, which is automatic for the exactly-linear families and is NOT automatic for
an RBF SVM: the script reports a missing crossing rather than inventing one.

usage:  python analysis/table_reading.py [--dataset D] [--model M] [--points 10,11,13,15]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients

import argparse
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from sweeps.sweep import opts, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

STEPS = (1, 2)           # how far along the feature the reading is carried, in sd
SEARCH = 8.0             # the flip is looked for within this many sd of q


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p/(1 - p))


def surrogate_logodds(expl, X):
    m = expl.surrogate_model
    return float(m.intercept_) + np.atleast_2d(X) @ np.atleast_2d(m.coef_)[-1, :]


def true_flip(clf, q, j, search=SEARCH, n=3201):
    '''
    signed distance along feature j to the black box's own decision boundary, nearest q.

    Found by scanning rather than solved, because f need not be monotone along an axis;
    nan when there is no crossing in range, which is a fact about the black box and is
    reported as one.
    '''
    t = np.linspace(-search, search, n)
    X = np.repeat(np.asarray(q, dtype=float)[None, :], t.size, axis=0)
    X[:, j] += t
    s = np.sign(np.asarray(clf.predict_proba(X))[:, 1] - 0.5)
    cross = np.where(np.diff(s) != 0)[0]
    return float(t[cross[np.argmin(np.abs(t[cross]))]]) if cross.size else float('nan')


def readings(clf, train, test, q, j):
    E = clime.explainer.AVAILABLE_EXPLAINERS
    e_std = E['bLIMEy (normal)'](clf, query_point=q, train_data=train, test_data=test)
    e_log = E['bLIMEy (logit)'](clf, query_point=q, train_data=train, test_data=test)
    b_std = np.asarray(e_std.get_explanation(), dtype=float)
    b_log = np.asarray(e_log.get_explanation(), dtype=float)

    f_q = float(np.asarray(clf.predict_proba(q[None, :]))[0, 1])
    g_q = float(e_std.surrogate_model.predict(q[None, :])[0, 1])       # unclipped
    l_q = float(surrogate_logodds(e_log, q[None, :])[0])

    n_std = float(np.linalg.norm(b_std))
    r_exit = (1 - g_q)/n_std if g_q >= 0.5 else g_q/n_std

    # the surrogate's own training neighbourhood, same draw site (same salt) as bLIMEy
    rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
    X = rng.multivariate_normal(q, np.cov(test['X'].T), 10000)
    w = costs.weights_based_on_distance(q, X)
    g_n = e_std.surrogate_model.predict(X)[:, 1]
    mass = float(np.sum(w*((g_n > 1) | (g_n < 0)))/np.sum(w))

    walk = {}
    for s in STEPS:
        x = q.copy()
        x[j] += s
        walk[s] = (g_q + s*b_std[j],
                   float(1/(1 + np.exp(-(l_q + s*b_log[j])))),
                   float(np.asarray(clf.predict_proba(x[None, :]))[0, 1]))

    return {
        'f_q': f_q, 'g_std': g_q, 'g_log': float(1/(1 + np.exp(-l_q))),
        'coef_std': float(b_std[j]), 'coef_log': float(b_log[j]),
        'odds': float(np.exp(b_log[j])),
        'norm_std': n_std, 'r_exit': float(r_exit), 'mass': mass,
        'walk': walk,
        'flip_std': float((0.5 - g_q)/b_std[j]) if b_std[j] != 0 else float('nan'),
        'flip_log': float(-l_q/b_log[j]) if b_log[j] != 0 else float('nan'),
    }


def num(x, fmt='{:.2f}', dash='---'):
    return dash if not np.isfinite(x) else f'${fmt.format(x)}$'


def feature_label(name):
    '''
    the synthetic datasets name their columns 'feature 0', 'feature 1', which reads badly
    after the word "feature" in a caption; render those as $x_j$ and escape the rest
    '''
    name = str(name)
    if name.startswith('feature ') and name[8:].isdigit():
        return f'$x_{{{name[8:]}}}$'
    return r'\emph{' + name.replace('_', ' ') + '}'


def main(dataset='Gaussian', model='Logistic', points='10,11,13,15'):
    idx = [int(i) for i in str(points).split(',')]
    r = clime.pipeline.run_pipeline(opts(dataset, model, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train, test = r['clf'], r['train_data'], r['test_data']
    names = list(test['feature_names'])
    qs, _ = get_points_between_class_means(test)
    qs = np.asarray(qs, dtype=float)
    d = qs.shape[1]
    K = costs.KERNEL_WIDTH_SCALE*np.sqrt(d)

    # the feature the reading is carried along: the one the black box actually leans on
    truth0 = gradients.grad_logit(clf, model, qs[len(qs)//2])[0]
    J = int(np.argmax(np.abs(truth0)))

    rows, truths, flips = [], [], []
    for i in idx:
        rows.append(readings(clf, train, test, qs[i], J))
        truths.append(float(gradients.grad_logit(clf, model, qs[i])[0][J]))
        flips.append(true_flip(clf, qs[i], J))

    n = len(idx)
    col = 'l' + 'r'*n
    head = ' & '.join(f'point {i}' for i in idx)
    # computed, not asserted: how far the reported coefficient moves across the points
    # shown, while the truth (a constant for an exactly-linear black box) does not
    shown_coefs = [abs(r['coef_std']) for r in rows if r['coef_std'] != 0]
    drop = max(shown_coefs)/min(shown_coefs) if shown_coefs else float('nan')

    def line(label, values):
        return f'    {label} & ' + ' & '.join(values) + r' \\'

    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \footnotesize',
        r'  \caption{What each coefficient claims, at ' + f'{len(idx)}' +
        f' of the {len(qs)} query points on the line between the class means '
        f'({dataset}, {model} black box, feature {feature_label(names[J])}). '
        "One unit is one standard deviation of that feature on the training split, so "
        "a step of one is not a small step. The black box's log-odds are exactly linear "
        "here, so its own gradient is the true local importance and does not change from "
        f"point to point; the standard coefficient falls {drop:.0f}-fold across "
        "these four. " + r'\emph{Reach}' + " is how far the standard surrogate's output "
        "can be carried from $q$ before it leaves $[0,1]$, in units of the locality "
        f"kernel width $k={K:.2f}$ that defined the neighbourhood; " +
        r'\emph{outside $[0,1]$}' + " is the kernel-weighted fraction of the surrogate's "
        r"own $10{,}000$-point training sample on which its unclipped output is not a "
        "probability. " + r'\emph{Flip}' + " is the distance along this feature at which "
        r"each model changes its decision. Figure~\ref{fig:reading} draws point "
        f'{idx[1] if len(idx) > 1 else idx[0]}' + r'.}',
        r'  \label{tab:reading}',
        rf'  \begin{{tabular}}{{@{{}}{col}@{{}}}}',
        r'    \toprule',
        f'    & {head}' + r' \\',
        r'    \midrule',
        line(r'black box $f(q)$', [num(x['f_q'], '{:.3f}') for x in rows]),
        r'    \midrule',
        r'    \multicolumn{' + str(n + 1) + r'}{@{}l}{\emph{the value the surrogate '
        r'starts from}} \\',
        line(r'\quad standard LIME $g(q)$', [num(x['g_std'], '{:.3f}') for x in rows]),
        line(r'\quad Logit-LIME $g(q)$', [num(x['g_log'], '{:.3f}') for x in rows]),
        r'    \midrule',
        r'    \multicolumn{' + str(n + 1) + r'}{@{}l}{\emph{the coefficient reported for '
        r'this feature}} \\',
        line(r'\quad standard LIME (prob.\ per sd)',
             [num(x['coef_std'], '{:.3f}') for x in rows]),
        line(r'\quad Logit-LIME (log-odds per sd)',
             [num(x['coef_log'], '{:.3f}') for x in rows]),
        line(r'\quad truth (log-odds per sd)', [num(t, '{:.3f}') for t in truths]),
        line(r'\quad Logit-LIME as an odds ratio $e^{\beta}$',
             [num(x['odds'], '{:.1f}') for x in rows]),
        r'    \midrule',
    ]
    for s in STEPS:
        lines += [
            r'    \multicolumn{' + str(n + 1) + r'}{@{}l}{\emph{claimed $p$ after '
            f'${s:+d}$ sd of this feature' + r'}} \\',
            line(r'\quad standard LIME', [num(x['walk'][s][0], '{:.3f}') for x in rows]),
            line(r'\quad Logit-LIME', [num(x['walk'][s][1], '{:.3f}') for x in rows]),
            line(r'\quad black box', [num(x['walk'][s][2], '{:.3f}') for x in rows]),
        ]
    lines += [
        r'    \midrule',
        line(r'reach of the standard reading (sd)',
             [num(x['r_exit'], '{:.2f}') for x in rows]),
        line(r'\quad as a fraction of $k$', [num(x['r_exit']/K, '{:.2f}') for x in rows]),
        line(r'\quad training mass outside $[0,1]$',
             [f"${x['mass']:.0%}$".replace('%', r'\%') for x in rows]),
        r'    \midrule',
        r'    \multicolumn{' + str(n + 1) + r'}{@{}l}{\emph{implied flip distance along '
        r'this feature (sd)}} \\',
        line(r'\quad standard LIME', [num(x['flip_std']) for x in rows]),
        line(r'\quad Logit-LIME', [num(x['flip_log']) for x in rows]),
        line(r'\quad black box (truth)', [num(t) for t in flips]),
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]

    open(paths.table('reading.tex'), 'w').write('\n'.join(lines) + '\n')

    print(f'\n{dataset} | {model} | feature {names[J]} | d={d}, k={K:.2f}\n')
    hdr = ' '.join(f'{"pt "+str(i):>10s}' for i in idx)
    print(f'{"":36s} {hdr}')

    def show(label, values, fmt='{:>10.3f}'):
        print(f'{label:36s} ' + ' '.join(fmt.format(v) for v in values))

    show('f(q)', [x['f_q'] for x in rows])
    show('g(q) standard', [x['g_std'] for x in rows])
    show('g(q) logit', [x['g_log'] for x in rows])
    show('coef standard (prob/sd)', [x['coef_std'] for x in rows])
    show('coef logit (log-odds/sd)', [x['coef_log'] for x in rows])
    show('truth (log-odds/sd)', truths)
    for s in STEPS:
        show(f'claim {s:+d}sd standard', [x['walk'][s][0] for x in rows])
        show(f'claim {s:+d}sd logit', [x['walk'][s][1] for x in rows])
        show(f'claim {s:+d}sd black box', [x['walk'][s][2] for x in rows])
    show('reach (sd)', [x['r_exit'] for x in rows])
    show('reach / k', [x['r_exit']/K for x in rows])
    show('mass outside [0,1]', [x['mass'] for x in rows])
    show('flip standard', [x['flip_std'] for x in rows])
    show('flip logit', [x['flip_log'] for x in rows])
    show('flip truth', flips)
    ratio = [abs(x['flip_std']/t) for x, t in zip(rows, flips) if np.isfinite(t)]
    if ratio:
        print(f'\n  standard LIME overstates the flip distance by '
              f'{min(ratio):.1f}x to {max(ratio):.1f}x over these points')
    print('\nwritten tables/reading.tex')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Gaussian')
    p.add_argument('--model', default='Logistic')
    p.add_argument('--points', default='10,11,13,15')
    main(**vars(p.parse_args()))
