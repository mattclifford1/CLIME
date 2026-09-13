'''
shared matplotlib style for the aLIMEgn figures

A copy of experiments/logit_lime/common/style.py rather than an import of it: both
packages are called `common`, so importing across them needs sys.path games that break
under multiprocessing. The palette is the same validated one, so figures from the two
studies sit together.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# slots 1-4 of the reference categorical palette (light mode), used unmodified.
# documented as passing the all-pairs gates: CVD dE 9.2, normal-vision dE 24.0
BLUE, ORANGE, AQUA, PLUM = '#2a78d6', '#eb6834', '#1baf7a', '#8c5bd8'
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#8a8985'

# one colour per weighting scheme, used consistently across every figure here
SCHEME_COLOURS = {
    'bLIMEy (normal)': INK_2,
    'bLIMEy (cost sensitive sampled)': BLUE,
    'bLIMEy (cost sensitive class)': MUTED,
    'bLIMEy (local y)': ORANGE,
    'bLIMEy (local yhat)': AQUA,
    'bLIMEy (density ratio)': PLUM,
}

SCHEME_LABELS = {
    'bLIMEy (normal)': 'standard LIME',
    'bLIMEy (cost sensitive sampled)': r'class weights from $\hat{y}$ (sample)',
    'bLIMEy (cost sensitive class)': r'class weights from $y$ (global)',
    'bLIMEy (local y)': r'class weights from $y$ (local)',
    'bLIMEy (local yhat)': r'class weights from $\hat{y}$ (local)',
    'bLIMEy (density ratio)': 'density ratio',
}

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.labelcolor': INK, 'text.color': INK,
    'legend.frameon': False, 'lines.linewidth': 1.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
