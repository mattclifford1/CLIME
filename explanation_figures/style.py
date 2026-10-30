'''
The CLIME thesis figures' style: the shared thesis style (~/Repos/lib/thesis_figures) saving
into out/, plus one colour per weighting scheme.

Set `width` in `figsize` to the fraction of \\textwidth used in \\includegraphics.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import os

from thesis_figures.roles import *  # noqa: F401,F403
from thesis_figures.roles import C, MUTED
from thesis_figures.style import TEXTWIDTH, figsize  # noqa: F401
from thesis_figures import style as _shared

HERE = os.path.dirname(os.path.abspath(__file__))
# mirrors the thesis's chapters/<chapter>/figures/ as out/<chapter>/
OUT = os.path.join(HERE, 'out')

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


def save(fig, name):
    '''Write `name`.pdf under OUT, where `name` is `<chapter>/<path in its figures folder>`.'''
    _shared.save(fig, name, OUT)
