'''
Table: one worked explanation, feature by feature (-> tables/example-explanation.tex).

The aggregate agreement numbers (sweep_explanations.py, sweep_ground_truth.py) say the
two surrogates rank features differently and that Logit-LIME is closer to the truth. They
do not show a reader what that looks like on the page, which is the form a LIME user
actually consumes: a short list of features with weights beside them.

This prints that list for a single (dataset, black box, query point), against the black
box's own coefficients as ground truth - so it is restricted to the black boxes with
exactly linear log-odds, where those coefficients ARE the local importances.

Coefficients are not comparable in units across surrogates: one regresses probabilities,
the other log-odds, so they differ by a factor set by the local slope of the sigmoid.
Each vector is therefore scaled to unit maximum magnitude, which leaves ranks, signs and
within-vector relative magnitudes untouched - and those are what a user reads.

The query point is the midpoint of the line between the class means by default, chosen by
position rather than by outcome; --point overrides it.

usage:  python analysis/table_example_explanation.py [--dataset D] [--model M] [--point i]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import argparse
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweeps.sweep import opts, METRICS
from sweeps.sweep_ground_truth import true_coefficients, cosine, SURROGATES
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

SHOW = 6            # features listed, by true importance
COLUMNS = ['standard', 'logit']
HEADING = {'standard': 'standard LIME', 'logit': 'Logit-LIME'}


def unit_max(v):
    '''scale so the largest magnitude is 1 - ranks, signs and ratios are preserved'''
    m = np.max(np.abs(v))
    return v/m if m > 0 else v


def ranks(v):
    '''1-based rank of each feature by |coefficient|, largest first'''
    r = np.empty(v.size, dtype=int)
    r[np.argsort(-np.abs(v))] = np.arange(1, v.size + 1)
    return r


def explanations(clf, test_data, q):
    out = {}
    for label, name in SURROGATES.items():
        e = clime.explainer.AVAILABLE_EXPLAINERS[name](clf, np.array(q),
                                                       test_data=test_data)
        out[label] = np.asarray(e.get_explanation(), dtype=float)
    return out


def escape(name):
    return str(name).replace('_', ' ')


def pct(v):
    '''a percentage for the caption - the bare % would comment out the rest of the line'''
    return f'{np.mean(v):.0%}'.replace('%', r'\%')


def main(dataset='Breast Cancer', model='Logistic', point=None):
    r = clime.pipeline.run_pipeline(opts(dataset, model, SURROGATES['standard'],
                                         METRICS[0]), parallel_eval=False)
    truth = np.asarray(true_coefficients(r['clf'], model), dtype=float)
    names = list(r['test_data']['feature_names'])
    qs, _ = get_points_between_class_means(r['test_data'])
    point = len(qs)//2 if point is None else point
    coefs = explanations(r['clf'], r['test_data'], qs[point])

    # summary statistics at this query point, and averaged over all of them
    here = {k: (cosine(v, truth), spearmanr(np.abs(v), np.abs(truth))[0])
            for k, v in coefs.items()}
    over_all = {k: [] for k in SURROGATES}
    for q in qs:
        for k, v in explanations(r['clf'], r['test_data'], q).items():
            over_all[k].append(float(np.argmax(np.abs(v)) == np.argmax(np.abs(truth))))

    shown = list(np.argsort(-np.abs(truth))[:SHOW])
    # a surrogate's own top pick is the interesting part of the disagreement, so make
    # sure it is on the page even when the true ranking does not put it there
    for k in COLUMNS:
        top = int(np.argmax(np.abs(coefs[k])))
        if top not in shown:
            shown.append(top)

    scaled = {'truth': unit_max(truth), **{k: unit_max(v) for k, v in coefs.items()}}
    rank = {'truth': ranks(truth), **{k: ranks(v) for k, v in coefs.items()}}

    lines = [
        r'\begin{table}[tbp]',
        r'  \centering',
        r'  \small',
        r'  \caption{One explanation, feature by feature: ' + escape(dataset) +
        f' with the {escape(model)} black box, at query point {point} of '
        f'{len(qs)} ' + "along the line between the class means. The black box's own "
        r'coefficients are the true local importances, since its log-odds are exactly '
        r'linear. Each vector is scaled to unit maximum magnitude, because the two '
        r'surrogates regress different quantities and their raw coefficients are not '
        r'comparable in units; ranks and signs are unaffected. Rows are the '
        f'{SHOW} truly most important features, plus any feature a surrogate puts first. '
        r'Ranks are over all ' + f'{truth.size} features. ' +
        f'At this point Logit-LIME attains cosine similarity {here["logit"][0]:.4f} '
        f'against standard LIME\'s {here["standard"][0]:.4f}; over all {len(qs)} query '
        f'points Logit-LIME names the true most important feature at '
        f'{pct(over_all["logit"])} of them and standard LIME at '
        f'{pct(over_all["standard"])}.' + r'}',
        r'  \label{tab:example}',
        r'  \begin{tabular}{lrrrrrr}',
        r'    \toprule',
        r'    & \multicolumn{2}{c}{black box (truth)} & '
        r'\multicolumn{2}{c}{standard LIME} & \multicolumn{2}{c}{Logit-LIME} \\',
        r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
        r'    feature & weight & rank & weight & rank & weight & rank \\',
        r'    \midrule',
    ]
    for j, i in enumerate(shown):
        if j == SHOW:
            lines.append(r'    \midrule')
        cells = []
        for k in ('truth', *COLUMNS):
            bold = r'\textbf' if rank[k][i] == 1 else ''
            cells.append(f'${scaled[k][i]:+.2f}$')
            cells.append(f'{bold}{{{rank[k][i]}}}' if bold else f'{rank[k][i]}')
        lines.append(f'    {escape(names[i])} & ' + ' & '.join(cells) + r' \\')
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    open(paths.table('example-explanation.tex'), 'w').write('\n'.join(lines) + '\n')

    # readable version, for reading off numbers to quote in the prose
    print(f'\n{dataset} | {model} | query point {point} of {len(qs)}   '
          f'({truth.size} features)\n')
    print(f"{'feature':<26s} {'truth':>8s} {'rank':>5s} "
          f"{'std':>8s} {'rank':>5s} {'logit':>8s} {'rank':>5s}")
    for i in shown:
        print(f'{str(names[i]):<26s} '
              + ' '.join(f'{scaled[k][i]:>+8.2f} {rank[k][i]:>5d}'
                         for k in ('truth', *COLUMNS)))
    print()
    for k in SURROGATES:
        print(f'  {HEADING.get(k, k):<16s} cosine {here[k][0]:+.4f}   '
              f'rank rho {here[k][1]:+.4f}   '
              f'top-1 over all points {np.mean(over_all[k]):.0%}')
    print('\nwritten tables/example-explanation.tex')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Breast Cancer')
    p.add_argument('--model', default='Logistic')
    p.add_argument('--point', type=int, default=None)
    main(**vars(p.parse_args()))
