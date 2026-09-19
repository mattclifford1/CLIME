'''
Every experiment of the fifth registration, on the full 71-dataset grid, one dataset per
process (sweeps/parallel.py). Each job writes its own results/*_full.json; nothing here
writes to a file the paper already cites.

    uv run python sweeps/run_full.py                 # every job, in the order below
    uv run python sweeps/run_full.py full seeds      # just these
    uv run python sweeps/run_full.py --jobs 16 ...   # fewer processes

Each job resumes from its shards in results/shards/<name>/, so re-running after an
interruption, or after adding a dataset to full_grid.py, only computes what is missing.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from sweeps import parallel, full_grid

FULL = "from sweeps import full_grid\nfull_grid.configure()\n"

JOBS = {
    # name: (output file, datasets, setup, call)
    'full': ('results_full.json', full_grid.DATASETS,
             FULL + "from sweeps import sweep as m",
             "m.DATASETS = {datasets!r}\nm.run({out!r})"),
    'checks': ('results_diagnostic_checks.json', full_grid.DATASETS,
               FULL + "from sweeps import sweep_diagnostic_checks as m",
               "m.run({out!r}, {datasets!r}, full_grid.CHECK_MODELS)"),
    'querypoints': ('results_querypoints_full.json', full_grid.DATASETS,
                    FULL + "from sweeps import sweep as m\nm.QUERY_POINTS = 'random_test_points'",
                    "m.DATASETS = {datasets!r}\nm.run({out!r})"),
    'gradient_truth': ('results_gradient_truth_full.json', full_grid.DATASETS,
                       FULL + "from sweeps import sweep_gradient_truth as m",
                       "m.run({out!r}, {datasets!r}, m.MODELS)"),
    'taylor': ('results_taylor_full.json', full_grid.DATASETS,
               FULL + "from sweeps import sweep_taylor as m",
               "m.run({out!r}, {datasets!r}, m.MODELS)"),
    'fidelity': ('results_fidelity_full.json', full_grid.DATASETS,
                 FULL + "from sweeps import sweep_fidelity as m",
                 "m.DATASETS = {datasets!r}\nm.run({out!r}, models=full_grid.FIDELITY_MODELS)"),
    'null': ('results_null_full.json', full_grid.DATASETS,
             FULL + "from sweeps import sweep_null as m\n"
                    "m.MODELS = full_grid.MODELS\nm.GROUP_OF = full_grid.GROUP_OF",
             "m.run({out!r}, {datasets!r})"),
    'explanations': ('results_explanations_full.json', full_grid.DATASETS,
                     FULL + "from sweeps import sweep_explanations as m\n"
                            "m.MODELS = full_grid.MODELS\nm.GROUP_OF = full_grid.GROUP_OF",
                     "m.DATASETS = {datasets!r}\nm.run({out!r})"),
    **{f'seed{s}': (f'results_full_seed{s}.json', full_grid.DATASETS,
                    FULL + "from sweeps import sweep as m",
                    "m.DATASETS = {datasets!r}\nm.run({out!r}, seed=%d)" % s)
       for s in (1, 2, 3, 4)},
    'kernel': ('results_kernel_full.json', full_grid.DATASETS,
               FULL + "from sweeps import sweep_kernel as m\nm.MODELS = full_grid.MODELS",
               "m.DATASETS = {datasets!r}\nm.run({out!r})"),
    'ridge_alpha': ('results_ridge_alpha.json', full_grid.REGISTERED,
                    "from sweeps import sweep_ridge_alpha as m, sweep",
                    "m.run({out!r}, {datasets!r}, sweep.MODELS)"),
}

# the slowest datasets first, so the long ones do not start last and hold up the job
SLOW_FIRST = ['Arrhythmia', 'MakeClf d200 i100', 'MakeClf d200 i5', 'MakeClf d100 i50',
              'MakeClf d100 i5', 'Digits 3 vs 8', 'Z-Alizadeh Sani CAD', 'MakeClf d60 i30',
              'MakeClf d60 i5', 'Sonar Rocks vs Mines', 'HCC Survival', 'SPECTF Heart']


def ordered(datasets):
    first = [d for d in SLOW_FIRST if d in datasets]
    return first + [d for d in datasets if d not in first]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('jobs', nargs='*', default=list(JOBS))
    p.add_argument('--jobs', dest='n', type=int, default=30)
    a = p.parse_args()
    for name in a.jobs:
        out, datasets, setup, call = JOBS[name]
        parallel.run(out, ordered(datasets), setup, call, jobs=a.n)
