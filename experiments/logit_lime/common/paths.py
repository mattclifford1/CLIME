'''
Where things live, resolved from this file rather than from the working directory.

The scripts here are run from several places - by hand, by rerun_all.sh, and by
sweep_seeds.py in a subprocess - so a bare 'results_taxonomy.json' cannot mean "relative
to wherever you happen to be standing". Every script asks this module instead.

A name with no directory component is placed in the matching output folder; a name that
already has one (or is absolute) is passed through untouched, so a command line argument
can still point anywhere.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT/'results'
ARCHIVE = RESULTS/'archive'
FIGS = ROOT/'figs'
TABLES = ROOT/'tables'
LOGS = ROOT/'logs'
DATASETS = ROOT/'extra_datasets'


def _place(folder, name):
    p = Path(name)
    if p.is_absolute() or len(p.parts) > 1:
        return os.fspath(p)
    folder.mkdir(parents=True, exist_ok=True)
    return os.fspath(folder/name)


def results(name):
    return _place(RESULTS, name)


def fig(name):
    return _place(FIGS, name)


def table(name):
    return _place(TABLES, name)
