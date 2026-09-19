'''
Run any sweep here one dataset per process, then merge the shards into one results file.

Every sweep in this directory is serial by design (see the README: parallel evaluation
inside the pipeline is nondeterministic at 1e-3, which is the size of the effects). But
the sweeps are embarrassingly parallel *across datasets*: each (dataset, black box) pair
is computed from scratch, sampling is seeded per query point (FINDINGS B10), and each
sweep resumes from whatever its output file already holds. So running one dataset per
process and merging gives exactly the serial result, on as many cores as there are.
"Exactly" is checked, not assumed: `verify` recomputes configurations that are already
in a serial results file and compares every stored score.

    from sweeps import parallel
    parallel.run(name='results_foo.json', datasets=[...], jobs=30,
                 setup="from sweeps import sweep_foo as m\\nm.MODELS = [...]",
                 call="m.run({out!r}, {datasets!r})")

`setup` and `call` are Python source run in the child. `call` is formatted with `out`
(the shard's path) and `datasets` (a one-element list), so it can hand the dataset list
to a sweep whose run() takes one, or `setup` can assign it to a module global for a
sweep that reads DATASETS directly - both styles exist here.

Shards live in results/shards/<name without .json>/, one file per dataset, and are kept
after merging: they are the resume state, so an interrupted run restarts from them.
BLAS is pinned to one thread per process, since 30 processes each spawning 32 threads
is slower than serial.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

CHILD = '''
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, {root!r})
{setup}
{call}
'''

THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def shard_dir(name):
    d = paths.RESULTS/'shards'/os.path.splitext(os.path.basename(name))[0]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(dataset):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in dataset)


def _run_one(dataset, name, setup, call):
    d = shard_dir(name)
    out = str(d/f'{_safe(dataset)}.json')
    log = d/f'{_safe(dataset)}.log'
    code = CHILD.format(root=str(paths.ROOT), setup=setup,
                        call=call.format(out=out, datasets=[dataset]))
    t = time.time()
    with open(log, 'a') as f:
        rc = subprocess.run([sys.executable, '-u', '-c', code], cwd=paths.ROOT,
                            stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **THREAD_ENV}).returncode
    return dataset, rc, time.time() - t


def merge(name, datasets=None):
    '''merge every shard (or those for `datasets`) into results/<name>'''
    d = shard_dir(name)
    files = sorted(d.glob('*.json')) if datasets is None else \
        [d/f'{_safe(x)}.json' for x in datasets]
    merged, meta = {}, None
    for f in files:
        if not f.exists():
            continue
        part = json.load(open(f))
        for k, v in part.items():
            if k.startswith('_'):
                meta = meta or {}
                meta.setdefault(k, v)
            else:
                merged[k] = v
    out = {**(meta or {}), **merged}
    path = paths.results(name)
    json.dump(out, open(path, 'w'))
    n_err = sum(1 for k, v in merged.items() if isinstance(v, dict) and 'error' in v)
    print(f'merged {len(files)} shards -> {path}: {len(merged)} entries, {n_err} errors',
          flush=True)
    return out


def run(name, datasets, setup, call, jobs=30):
    '''run `call` for each dataset in its own process, `jobs` at a time, then merge'''
    t0 = time.time()
    print(f'{name}: {len(datasets)} datasets on {jobs} processes', flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for dataset, rc, dt in ex.map(lambda x: _run_one(x, name, setup, call), datasets):
            status = 'ok' if rc == 0 else f'EXIT {rc}'
            if rc != 0:
                failed.append(dataset)
            print(f'  {status:8s} {dataset:36s} {dt/60:6.1f} min', flush=True)
    print(f'{name}: done in {(time.time()-t0)/60:.1f} min, {len(failed)} failed '
          f'{failed if failed else ""}', flush=True)
    return merge(name, datasets)


def _scores(entry):
    '''every per-query-point score list in an entry, keyed by its path'''
    out = {}
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (k,))
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out[path] = [x]
        elif isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x):
            out[path] = list(x)
    walk(entry, ())
    return out


def verify(reference, candidate, keys=None, atol=0.0):
    '''
    compare every number stored under `keys` (default: all shared keys) in two results
    files; returns the largest absolute difference and prints any that exceed atol
    '''
    import numpy as np
    a, b = json.load(open(paths.results(reference))), json.load(open(paths.results(candidate)))
    keys = keys or [k for k in b if not k.startswith('_') and k in a]
    worst = 0.0
    for k in keys:
        sa, sb = _scores(a[k]), _scores(b[k])
        for path in set(sa) & set(sb):
            x, y = np.asarray(sa[path], float), np.asarray(sb[path], float)
            if x.shape != y.shape:
                print(f'  shape differs {k} {path}')
                continue
            both = np.isfinite(x) & np.isfinite(y)
            if (np.isfinite(x) != np.isfinite(y)).any():
                print(f'  finiteness differs {k} {path}')
            if both.any():
                dmax = float(np.max(np.abs(x[both] - y[both])))
                worst = max(worst, dmax)
                if dmax > atol:
                    print(f'  {k} {"/".join(path)}: max |diff| = {dmax:.3g}')
    print(f'verified {len(keys)} configurations: max |diff| = {worst:.3g}')
    return worst
