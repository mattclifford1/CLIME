'''
P1 and P2 in one figure: the collapse belongs to the evaluation marginal

(a) standard LIME along the line, scored against the test set and against its own
    training marginal. The gap between the two curves IS the CIKM'23 effect.
(b) every weighting scheme, scored against the test set, on the same configuration.
(c) all configurations at once: how far the score moves along the line under each
    marginal. Variation (max - min), not the boundary-to-tail drop - see the note at the
    top of analysis/analyse_marginal.py for why the latter is the wrong instrument.

usage:  uv run python figures/fig_marginal.py [dataset] [model]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis'))

import numpy as np
import matplotlib.pyplot as plt

from common import paths, style
import common_analysis as ca

METRIC = 'fidelity (local)'
NORMAL = 'bLIMEy (normal)'
DEFAULT = ('Breast Cancer', 'Random Forest')


def panel_a(ax, entry):
    x = np.arange(entry['n_query_points'])
    for eval_data, colour, marker in (('test data', style.ORANGE, 'o'),
                                      ('sample locally', style.BLUE, 's')):
        ax.plot(x, ca.series(entry, NORMAL, METRIC, eval_data), marker=marker, ms=3,
                color=colour, label=f'scored on {eval_data}')
    boundary = ca.boundary_index(entry)
    ax.axvline(boundary, color=style.MUTED, ls=':', lw=1)
    ax.annotate('decision boundary', (boundary, 1.005), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=7, color=style.INK_2)
    ax.set_xlabel('query point (class 0 mean $\\to$ class 1 mean)')
    ax.set_ylabel('local fidelity')
    ax.set_title('(a) standard LIME, two evaluation marginals')
    ax.legend(loc='lower right', fontsize=7)


def panel_b(ax, entry):
    x = np.arange(entry['n_query_points'])
    for scheme, colour in style.SCHEME_COLOURS.items():
        y = ca.series(entry, scheme, METRIC, 'test data')
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, color=colour, lw=1.4, label=style.SCHEME_LABELS[scheme])
    ax.axvline(ca.boundary_index(entry), color=style.MUTED, ls=':', lw=1)
    ax.set_xlabel('query point')
    ax.set_ylabel('local fidelity on test data')
    ax.set_title('(b) every weighting scheme, on test data')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.5)


def panel_c(ax, results):
    test = np.array([ca.variation(e, NORMAL, METRIC, 'test data') for e in results.values()])
    local = np.array([ca.variation(e, NORMAL, METRIC, 'sample locally')
                      for e in results.values()])
    ok = np.isfinite(test) & np.isfinite(local)
    ax.scatter(local[ok], test[ok], s=14, color=style.ORANGE, alpha=0.8, edgecolor='none')
    lim = [0, max(np.nanmax(local[ok]), np.nanmax(test[ok]))*1.05]
    ax.plot(lim, lim, color=style.MUTED, ls='--', lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('variation on its own marginal')
    ax.set_ylabel('variation on test data')
    ax.set_title(f'(c) all {int(ok.sum())} configurations')
    ax.annotate(f'above the line in {(test[ok] > local[ok]).sum()}/{ok.sum()}:\n'
                'the collapse needs\nthe test marginal',
                (0.05, 0.78), xycoords='axes fraction', fontsize=7, color=style.INK_2)


def main(dataset, model):
    results, _, _ = ca.load('results_marginal.json')
    key = f'{dataset}|{model}'
    if key not in results:
        raise SystemExit(f'{key} not in results: try one of\n  ' +
                         '\n  '.join(sorted(results)[:20]))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    panel_a(axes[0], results[key])
    panel_b(axes[1], results[key])
    panel_c(axes[2], results)
    fig.suptitle(f'{dataset} — {model}', fontsize=9)
    out = paths.fig('fig_marginal.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    print('written', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0] if args else DEFAULT[0], args[1] if len(args) > 1 else DEFAULT[1])
