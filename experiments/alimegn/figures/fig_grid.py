'''
where in the space the misalignment lives

Every figure in the CIKM paper walks one line between the class means. This is the same
measurement over a 20x20 grid, on the two-dimensional datasets where the grid IS the
feature space: one row per black box, and for each, standard LIME's fidelity on the test
set, what class weighting adds there, and what it adds when the surrogate is scored on its
own marginal instead.

If P1 and P2 hold, the third column should be close to flat while the second has structure
away from the boundary.

usage:  uv run python figures/fig_grid.py [dataset]
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
CIKM = 'bLIMEy (cost sensitive sampled)'
DEFAULT_DATASET = 'Gaussian'
# Two rows by default rather than every black box in the sweep: a well fitted black box
# shows where the gain lives, and the same model trained on corrupted labels shows the same
# bands an order of magnitude weaker, which is P6 seen spatially. The write-up needs those
# two; pass a comma separated list to see the others.
DEFAULT_MODELS = ['Logistic', 'Logistic (label noise 0.3)']


def _grid(entry, values):
    '''the per-query-point list, back onto the grid it was built on'''
    points = np.array(entry['query_points'], dtype=np.float64)
    side = int(round(np.sqrt(len(points))))
    x, y = points[:, 0], points[:, 1]
    extent = [x.min(), x.max(), y.min(), y.max()]
    return values.reshape(side, side), extent


def _row(axes, entry, model, show_titles=True):
    normal = ca.series(entry, NORMAL, METRIC, 'test data')
    cikm_test = ca.series(entry, CIKM, METRIC, 'test data')
    cikm_local = ca.series(entry, CIKM, METRIC, 'sample locally')
    normal_local = ca.series(entry, NORMAL, METRIC, 'sample locally')

    panels = [(normal, 'standard LIME, on test data', 'viridis', None),
              (cikm_test - normal, 'class weighting gains, on test data', 'RdBu_r', True),
              (cikm_local - normal_local, 'class weighting gains, on its own marginal',
               'RdBu_r', True)]
    for ax, (values, title, cmap, symmetric) in zip(axes, panels):
        image, extent = _grid(entry, values)
        if symmetric:
            span = np.nanmax(np.abs(image)) or 1e-6
            mesh = ax.imshow(image, origin='lower', extent=extent, cmap=cmap,
                             vmin=-span, vmax=span, aspect='auto')
        else:
            mesh = ax.imshow(image, origin='lower', extent=extent, cmap=cmap,
                             aspect='auto')
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        if show_titles:
            ax.set_title(title, fontsize=8)
        ax.grid(False)
    axes[0].set_ylabel(f'{model}\n\nfeature 2', fontsize=8)


def main(dataset, models=None):
    results, meta, _ = ca.load('results_grid.json')
    entries = {k: v for k, v in results.items() if v['dataset'] == dataset}
    if not entries:
        raise SystemExit(f'no grid results for {dataset}: have '
                         f'{sorted({v["dataset"] for v in results.values()})}')
    wanted = models or DEFAULT_MODELS
    models = [m for m in wanted if f'{dataset}|{m}' in entries]
    if not models:
        raise SystemExit(f'none of {wanted} in the sweep: have '
                         f'{sorted(m for m in meta.get("models", []))}')

    fig, axes = plt.subplots(len(models), 3, figsize=(10.5, 2.9*len(models)),
                             squeeze=False)
    for row, model in enumerate(models):
        _row(axes[row], entries[f'{dataset}|{model}'], model, show_titles=row == 0)
    for ax in axes[-1]:
        ax.set_xlabel('feature 1')
    fig.suptitle(f'{dataset}: local fidelity over the grid, 400 query points per panel.\n'
                 'Colour scales are per panel: the gains differ by an order of magnitude '
                 'between black boxes', fontsize=9, y=1.0)
    fig.tight_layout()
    out = paths.fig(f'fig_grid_{dataset.replace(" ", "_")}.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    print('written', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET,
         sys.argv[2].split(',') if len(sys.argv) > 2 else None)
