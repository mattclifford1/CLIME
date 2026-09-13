'''
run a list of configurations in parallel, cached one file each

Each configuration is independent and costs minutes, so the sweeps here are embarrassingly
parallel. Work is split by configuration rather than by query point (which is what
`parallel_eval` in the pipeline does) for two reasons: a whole configuration shares one
fitted black box, and one file per configuration gives resume and incremental extension
for free (see store.py).

Set OMP_NUM_THREADS=1 when launching: sklearn's BLAS threads and these worker processes
otherwise oversubscribe the machine and everything slows down.

    OMP_NUM_THREADS=1 uv run python sweeps/sweep_marginal.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import multiprocessing as mp
import time
import traceback

import numpy as np
import clime

from . import store, evaluate


_DEFAULT_SEED = int(clime.RANDOM_SEED)


def _work(job):
    sweep, key, kwargs = job
    # a job without a seed means the default one, NOT "whatever this worker was left on":
    # a pool worker handles many jobs, so a seeded job earlier in the queue would otherwise
    # silently carry its seed into every later job it happened to pick up
    seed = int(kwargs.pop('seed', None) or _DEFAULT_SEED)
    if seed != int(clime.RANDOM_SEED):
        # the dataset split and every model read this at construction time, so a worker
        # that has already built something under another seed must forget it
        clime.RANDOM_SEED = seed
        np.random.seed(seed)
        evaluate._CACHE.clear()
    started = time.time()
    try:
        result = evaluate.evaluate_config(**kwargs)
    except Exception as e:                       # noqa: BLE001 - recorded, sweep continues
        result = {'error': f'{type(e).__name__}: {e}',
                  'traceback': traceback.format_exc()}
    result['key'] = key
    result['seed'] = seed
    result['seconds'] = round(time.time() - started, 1)
    store.save(sweep, key, result)
    return key, result.get('error'), result['seconds']


def run_jobs(sweep, jobs, processes=12, force=False, meta=None, out_name=None):
    '''
    jobs: list of (key, kwargs-for-evaluate_config) pairs

    A key already in the cache is skipped unless force=True, so adding a dataset or a black
    box to a sweep and re-running only computes the new cells.
    '''
    todo = [(sweep, key, dict(kwargs)) for key, kwargs in jobs
            if force or not store.has(sweep, key)]
    print(f'{sweep}: {len(jobs)} configurations, {len(jobs)-len(todo)} already cached, '
          f'{len(todo)} to run on {processes} processes', flush=True)

    if todo:
        started = time.time()
        with mp.Pool(processes) as pool:
            for i, (key, error, seconds) in enumerate(
                    pool.imap_unordered(_work, todo), start=1):
                status = 'FAILED ' + error[:60] if error else f'{seconds:6.1f}s'
                print(f'[{i:4d}/{len(todo)}] {key:70s} {status}', flush=True)
        print(f'{sweep}: finished in {(time.time()-started)/60:.1f} min', flush=True)

    return store.merge(sweep, out_name=out_name, meta=meta)
