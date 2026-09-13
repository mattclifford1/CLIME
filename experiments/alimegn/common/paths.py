'''
Where things live, resolved from this file rather than from the working directory.

Mirrors experiments/logit_lime/common/paths.py - the scripts here are run by hand, by
rerun_all.sh and by multiprocessing workers, so a bare 'results_marginal.json' cannot mean
"relative to wherever you happen to be standing".

A name with no directory component is placed in the matching output folder; a name that
already has one (or is absolute) is passed through untouched.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT/'results'
CACHE = RESULTS/'cache'
FIGS = ROOT/'figs'
TABLES = ROOT/'tables'
LOGS = ROOT/'logs'


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


def cache_dir(sweep):
    d = CACHE/sweep
    d.mkdir(parents=True, exist_ok=True)
    return d
