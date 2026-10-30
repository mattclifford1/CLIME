'''
Shared look for the CLIME thesis figures: colours, fonts, sizes and saving.

Matches deltas/linear_confidence/explanation_figures/style.py so the thesis chapters agree.
Figures are drawn at their printed size: set `width` in `figsize` to the fraction of
\\textwidth used in \\includegraphics.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
# mirrors the thesis's chapters/<chapter>/figures/ as out/<chapter>/
OUT = os.path.join(HERE, 'out')

# thesis type block is 448.13pt wide (preamble.tex)
TEXTWIDTH = 448.13 / 72.27

C = {'blue': '#2a78d6', 'red': '#d0342c', 'orange': '#eb6834', 'aqua': '#1baf7a',
     'violet': '#4a3aa7', 'plum': '#8c5bd8', 'brown': '#9a6324', 'yellow': '#f2c400'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8985', '#d8d7d2'

# class 0 blue and class 1 red, as in the deltas figures and the CIKM text's region colours
CLASS = {0: C['blue'], 1: C['red']}
QUERY = C['yellow']
SAMPLE = C['plum']
# P(class 1) from class 0's colour through white to class 1's
PROBA_CMAP = LinearSegmentedColormap.from_list('proba', [CLASS[0], '#ffffff', CLASS[1]])
SURFACE_ALPHA = 0.55
WEIGHT_CMAP = 'magma_r'

# one colour per weighting scheme, keyed by explainer registry name
SCHEME_COLOURS = {
    'bLIMEy (normal)': C['orange'],
    'bLIMEy (cost sensitive sampled)': C['violet'],
    'bLIMEy (cost sensitive class)': MUTED,
    'bLIMEy (local y)': C['aqua'],
    'bLIMEy (local yhat)': C['brown'],
    'bLIMEy (density ratio)': C['blue'],
}
SCHEME_LABELS = {
    'bLIMEy (normal)': r'standard: $w_x$',
    'bLIMEy (cost sensitive sampled)': r'class balanced: $w_{xc}$',
    'bLIMEy (cost sensitive class)': r'class weights from $y$ (global)',
    'bLIMEy (local y)': r'class weights from $y$ (local)',
    'bLIMEy (local yhat)': r'class weights from $\hat{y}$ (local)',
    'bLIMEy (density ratio)': 'density ratio',
}

# the thesis body font is New Century Schoolbook (fouriernc); TeX Gyre Schola is its clone
FONT_DIRS = ['/usr/share/texmf/fonts/opentype/public/tex-gyre',
             '/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre']
FONT = 'TeX Gyre Schola'


def _register_font():
    for d in FONT_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('texgyreschola'):
                    font_manager.fontManager.addfont(os.path.join(d, f))
    return any(f.name == FONT for f in font_manager.fontManager.ttflist)


def apply():
    serif = [FONT] if _register_font() else ['DejaVu Serif']
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': serif,
        'mathtext.fontset': 'custom', 'mathtext.rm': serif[0],
        'mathtext.it': f'{serif[0]}:italic', 'mathtext.bf': f'{serif[0]}:bold',
        'mathtext.fallback': 'cm',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.5, 'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def figsize(width=1.0, aspect=0.4):
    '''(w, h) in inches for a figure `width` x \\textwidth wide and `aspect` x as tall.'''
    w = width * TEXTWIDTH
    return w, w * aspect


def save(fig, name):
    '''Write `name`.pdf under OUT, where `name` is `<chapter>/<path in its figures folder>`.'''
    path = os.path.join(OUT, name + '.pdf')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # no creation date, so an unchanged figure is byte-identical and sync skips it
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)
    print(f'wrote {os.path.relpath(path)}')


apply()
