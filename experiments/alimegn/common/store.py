'''
one JSON file per configuration, so nothing is ever computed twice

Every sweep here is a loop over independent configurations, each costing minutes. Writing
one file per configuration (rather than one file per sweep, rewritten as it goes) buys
three things:

  - resume for free: a configuration with a cache file is skipped, so an interrupted or
    extended sweep only computes what is missing
  - parallelism without locking: workers never write the same file
  - a re-run after adding one dataset or one black box costs only the new cells

`merge` collects the cache into a single results/<sweep>.json for the analysis scripts,
which then need no knowledge of any of this.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import json
import os
import re
import numpy as np

from . import paths


def _filename(key):
    '''a configuration key is human readable with spaces and |; a filename should not be'''
    safe = re.sub(r'[^A-Za-z0-9._|=-]+', '_', key).replace('|', '__')
    return f'{safe}.json'


def _jsonable(obj):
    '''numpy types do not survive json.dump'''
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def has(sweep, key):
    return os.path.exists(paths.cache_dir(sweep)/_filename(key))


def load(sweep, key):
    path = paths.cache_dir(sweep)/_filename(key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save(sweep, key, value):
    path = paths.cache_dir(sweep)/_filename(key)
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(_jsonable(value), f)
    os.replace(tmp, path)   # atomic, so a killed worker cannot leave half a file
    return path


def merge(sweep, out_name=None, meta=None):
    '''collect every cached configuration into one results/<sweep>.json'''
    out = {'_meta': _jsonable(meta or {})}
    for path in sorted(paths.cache_dir(sweep).glob('*.json')):
        with open(path) as f:
            entry = json.load(f)
        out[entry.get('key', path.stem)] = entry
    out_path = paths.results(out_name or f'results_{sweep}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    n = len(out) - 1
    print(f'merged {n} configurations into {out_path}')
    return out_path
