'''figure style: the shared thesis style, with one colour and label per explainer and arm'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import matplotlib as mpl
import matplotlib.pyplot as plt
from thesis_figures import roles
from thesis_figures import style as thesis
from thesis_figures.roles import C, GRID, INK, INK2, MUTED   # noqa: F401
from thesis_figures.style import TEXTWIDTH   # noqa: F401

from . import paths

EXPLAINER_COLOURS = {
    'LIME': C['orange'],
    'CLIME': C['violet'],
    'LIME + re-threshold': C['blue'],
    'LIME + DKW Minimax': C['aqua'],
    'LIME + Gaussian Predictive': C['brown'],
    'Logit-LIME': C['aqua'],
    'Logit-CLIME': C['blue'],
    'Soft logistic': C['brown'],
    'Soft logistic CLIME': C['plum'],
    'KernelSHAP': C['red'],
    'Logit-KernelSHAP': C['yellow'],
    'gradient': INK2,
}
EXPLAINER_LABELS = {
    'LIME': r'LIME ($w_x$)',
    'CLIME': r'CLIME ($w_{xc}$)',
    'LIME + re-threshold': 'LIME, re-thresholded',
    'LIME + DKW Minimax': 'LIME, deltas (DKW)',
    'LIME + Gaussian Predictive': 'LIME, deltas (Gaussian)',
    'Logit-LIME': 'Logit-LIME',
    'Logit-CLIME': 'Logit-CLIME',
    'Soft logistic': 'soft logistic',
    'Soft logistic CLIME': 'soft logistic, CLIME weights',
    'KernelSHAP': 'KernelSHAP',
    'Logit-KernelSHAP': 'KernelSHAP on the logit',
    'gradient': "black box's gradient",
}
EXPLAINER_MARKERS = {'LIME': 'o', 'CLIME': 's', 'LIME + re-threshold': 'D',
                     'LIME + DKW Minimax': '^', 'LIME + Gaussian Predictive': 'v',
                     'Logit-LIME': '^', 'Logit-CLIME': 'v', 'Soft logistic': 'D',
                     'Soft logistic CLIME': 'P', 'KernelSHAP': 'X', 'Logit-KernelSHAP': '*'}
ARM_COLOURS = {
    'plain': INK2,
    'weighted': C['plum'],
    'SMOTE': C['aqua'],
    're-thresholded': C['blue'],
    'CP Minimax': C['red'],
    'Slacks Deltas': C['brown'],
    'Gaussian Predictive': C['orange'],
}
ARM_LABELS = {
    'plain': 'plain (0.5)',
    'weighted': 'class weighted',
    'SMOTE': 'SMOTE',
    're-thresholded': 're-thresholded (validation)',
    'CP Minimax': 'deltas (CP minimax)',
    'Slacks Deltas': 'deltas (ECAI 2024)',
    'Gaussian Predictive': 'deltas (Gaussian predictive)',
}
SETTING_LABELS = {'native': 'own ratio', 'balanced': 'balanced 110:110',
                  'imbalanced': 'imbalanced 200:20'}
FAMILY_SHORT = {'Random Forest': 'RF', 'Logistic': 'LR', 'SVM': 'SVM', 'Gradient Boosting': 'GB',
                'k Nearest Neighbours': 'kNN', 'MLP': 'MLP'}
ZERO_LINE = dict(color=INK, lw=1.0, zorder=1)


def use():
    thesis.apply()
    mpl.rcParams.update({'savefig.dpi': 300, 'axes.grid': True, 'axes.axisbelow': True,
                         'lines.markersize': 3.5})


def save(fig, name):
    path = paths.out(paths.FIGURES, f'{name}.pdf')
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)
    print(f'wrote {path.relative_to(paths.ROOT)}')
