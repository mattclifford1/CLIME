'''where things live, resolved from this file'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT/'results'
RAW = RESULTS/'raw'
FIGURES = ROOT/'figures'
TABLES = ROOT/'tables'
PAPER = ROOT.parents[2]/'papers'/'CIKM-2023-extra-datasets'


def raw(run):
    d = RAW/run
    d.mkdir(parents=True, exist_ok=True)
    return d


def out(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    return folder/name
