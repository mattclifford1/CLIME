'''
P3 and P4: what a degraded black box does to the choice between y and yhat

(a) the measured divergence between P(yhat|x) and P(y|x) against how much better
    yhat-derived class weights are than y-derived ones. The x-axis is measured, not the
    label-noise setting that was asked for.
(b) the label-noise ladder, so the trend is visible as a trend.
(c) the two objectives: agreement with f against agreement with the truth. P4 predicted
    configurations would land in the quadrant where the winner depends on which you ask
    for. They do not: yhat-derived weights are better on both.

usage:  uv run python figures/fig_degrade.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'analysis'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from common import paths, style
import common_analysis as ca
from analyse_degrade import divergence, mechanism, noise_rate, LOCAL_Y, LOCAL_YHAT

MECHANISM_COLOUR = {'clean': style.MUTED, 'label noise': style.ORANGE,
                    'underfit': style.BLUE, 'imbalance': style.AQUA}


def _rows(results):
    rows = []
    for entry in results.values():
        rows.append({
            'divergence': divergence(entry),
            'mechanism': mechanism(entry),
            'noise': noise_rate(entry),
            'kl': ca.paired_gain(entry, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)',
                                 'test data'),
            'fidelity': ca.paired_gain(entry, LOCAL_YHAT, LOCAL_Y, 'fidelity (local)',
                                       'test data'),
            'truth': ca.paired_gain(entry, LOCAL_Y, LOCAL_YHAT, 'surrogate vs truth'),
        })
    return rows


def panel_a(ax, rows):
    for name, colour in MECHANISM_COLOUR.items():
        pts = [(r['divergence'], r['kl']) for r in rows if r['mechanism'] == name
               and np.isfinite(r['divergence']) and np.isfinite(r['kl'])]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.scatter(x, y, s=16, color=colour, alpha=0.85, edgecolor='none', label=name)
    x = np.array([r['divergence'] for r in rows])
    y = np.array([r['kl'] for r in rows])
    ok = np.isfinite(x) & np.isfinite(y)
    rho = spearmanr(x[ok], y[ok])
    ax.axhline(0, color=style.MUTED, lw=1, ls='--')
    ax.set_xlabel(r'measured divergence: local rate of $\hat{y} \neq y$')
    ax.set_ylabel(r'$\log_{10}$ KL gain of $\hat{y}$ weights over $y$ weights')
    ax.set_title(f'(a) $\\rho = {rho.statistic:+.2f}$, $n = {ok.sum()}$')
    ax.legend(loc='upper left', fontsize=7)


def panel_b(ax, rows):
    ladder = [r for r in rows if r['mechanism'] in ('clean', 'label noise')]
    rates = sorted({r['noise'] for r in ladder})
    med, lo, hi, div = [], [], [], []
    for rate in rates:
        vals = [r['kl'] for r in ladder
                if r['noise'] == rate and np.isfinite(r['kl'])]
        med.append(np.median(vals))
        lo.append(np.percentile(vals, 25))
        hi.append(np.percentile(vals, 75))
        div.append(np.nanmedian([r['divergence'] for r in ladder if r['noise'] == rate]))
    ax.fill_between(rates, lo, hi, color=style.ORANGE, alpha=0.18, lw=0)
    ax.plot(rates, med, marker='o', ms=4, color=style.ORANGE, label=r'$\hat{y}$ over $y$')
    ax.axhline(0, color=style.MUTED, lw=1, ls='--')
    ax.set_xlabel('label noise applied to the training set')
    ax.set_ylabel(r'$\log_{10}$ KL gain')
    twin = ax.twinx()
    twin.plot(rates, div, color=style.INK_2, lw=1, ls=':', marker='s', ms=3,
              label='divergence')
    twin.set_ylabel('measured divergence', color=style.INK_2, fontsize=7.5)
    twin.tick_params(axis='y', labelsize=7, colors=style.INK_2)
    twin.grid(False)
    ax.set_title('(b) the noise ladder (IQR shaded)')
    ax.legend(loc='upper left', fontsize=7)


def panel_c(ax, rows):
    for name, colour in MECHANISM_COLOUR.items():
        pts = [(r['fidelity'], r['truth']) for r in rows if r['mechanism'] == name
               and np.isfinite(r['fidelity']) and np.isfinite(r['truth'])]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.scatter(x, y, s=16, color=colour, alpha=0.85, edgecolor='none', label=name)
    ax.axhline(0, color=style.MUTED, lw=1, ls='--')
    ax.axvline(0, color=style.MUTED, lw=1, ls='--')
    ax.set_xlabel(r'fidelity to $f$: gain of $\hat{y}$ weights')
    ax.set_ylabel(r'agreement with truth: gain of $y$ weights')
    ax.set_title('(c) P4 refuted: the two objectives agree')
    ax.annotate('P4 predicted this quadrant:\n'
                r'$y$ weights better on truth,' '\n'
                r'$\hat{y}$ weights better on $f$',
                (0.30, 0.80), xycoords='axes fraction', fontsize=6.5,
                color=style.INK_2)
    ax.annotate('most configurations land here:\n'
                r'$\hat{y}$ weights better on both',
                (0.30, 0.06), xycoords='axes fraction', fontsize=6.5,
                color=style.INK_2)


def main():
    results, _, _ = ca.load('results_degrade.json')
    rows = _rows(results)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), constrained_layout=True)
    panel_a(axes[0], rows)
    panel_b(axes[1], rows)
    panel_c(axes[2], rows)
    out = paths.fig('fig_degrade.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    print('written', out)


if __name__ == '__main__':
    main()
