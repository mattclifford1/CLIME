'''run jobs in parallel, one cache file per job and table, so an interrupted run resumes'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import re
import time
import traceback
import warnings

import pandas as pd
from joblib import Parallel, delayed

from . import paths

LAST = 'scores'   # written last: its file marks a finished job
N_JOBS = int(os.environ.get('N_JOBS', 16))


def _name(key):
    return re.sub(r'[^A-Za-z0-9.=-]+', '_', '__'.join(str(k) for k in key))


def _work(run, key, kwargs, job):
    warnings.filterwarnings('ignore')
    folder = paths.raw(run)
    started = time.time()
    if (folder/f'{_name(key)}.{LAST}.parquet').exists():
        return key, None, 0.0   # finished by another process since the list was made
    try:
        result = job(**kwargs)
    except Exception as e:   # noqa: BLE001 - recorded, the run continues
        (folder/f'{_name(key)}.error').write_text(traceback.format_exc())
        return key, f'{type(e).__name__}: {e}', time.time() - started
    for table in sorted(result, key=lambda t: t == LAST):
        tmp = folder/f'{_name(key)}.{table}.tmp'
        pd.DataFrame(result[table]).to_parquet(tmp)
        os.replace(tmp, folder/f'{_name(key)}.{table}.parquet')
    return key, None, time.time() - started


def run_jobs(run, jobs, n_jobs=N_JOBS, job=None):
    '''jobs: (key tuple, kwargs for `job`, evaluate.run by default); a job with a scores file is skipped'''
    if job is None:
        from .evaluate import run as job
    folder = paths.raw(run)
    todo = [(k, kw) for k, kw in jobs
            if not (folder/f'{_name(k)}.{LAST}.parquet').exists()]
    print(f'{run}: {len(jobs)} jobs, {len(todo)} to run on {n_jobs} workers', flush=True)
    started = time.time()
    results = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(_work)(run, k, kw, job) for k, kw in todo)
    failed = 0
    for i, (key, error, seconds) in enumerate(results, start=1):
        failed += error is not None
        if error or i % 25 == 0 or i == len(todo):
            print(f'[{i:5d}/{len(todo)}] {(time.time() - started)/60:6.1f} min  '
                  f'{" | ".join(str(k) for k in key)}  '
                  f'{"FAILED " + error[:80] if error else f"{seconds:.0f}s"}', flush=True)
    print(f'{run}: done in {(time.time() - started)/60:.1f} min, {failed} failed', flush=True)


def load(run, table):
    '''every cached job of a run, as one frame'''
    files = sorted(paths.raw(run).glob(f'*.{table}.parquet'))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
