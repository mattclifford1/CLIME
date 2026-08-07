# Pipeline

Everything runs through [`make_pipeline.py`](./make_pipeline.py). A single flat `opts`
dict selects one method for each of the eight pipeline stages by string key.

## The `opts` dict

```python
opts = {
    'dataset':             'Gaussian',
    'data params': {
        'class_samples':   [200, 200],   # synthetic datasets only
        'percent_of_data': 0.05,         # costcla datasets only
        'moons_noise':     0.2,
        'gaussian_means':  [[-1, -1], [1, 1]],
        'gaussian_covs':   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
    },
    'standardise data':    True,          # optional; zero mean, unit variance
    'dataset rebalancing': 'none',
    'model':               'Random Forest',
    'model balancer':      'none',
    'explainer':           'bLIMEy (normal)',
    'evaluation metric':   'fidelity (local)',
    'evaluation points':   'between_class_means',
    'evaluation data':     'test data',
}
```

Every value except `data params` and `standardise data` must be a key of the
corresponding registry in `AVAILABLE_MODULES` ([`__init__.py`](./__init__.py)). An
unknown key raises `ValueError` with the list of valid options.

## Running

```python
import clime

result = clime.pipeline.run_pipeline(opts, parallel_eval=True)   # cached
# or, uncached:
result = clime.pipeline.construct(opts).run(parallel_eval=True)
```

`result` keys: `'score'`, `'model_stats'`, `'clf'`, `'train_data'`, `'test_data'`.

`result['score']` comes from the evaluation runner and always has `'avg'` and `'std'`,
plus `'scores'` (per query point), `'eval_points'`, `'2D results'` (whether the points
lie on a line and so can be plotted as a graph rather than a heatmap), and optionally
`'class_weights'` and `'majority influence'`.

## Order of operations

```
run_section('dataset')          -> train_data, test_data
check_data_dict                 -> fills in feature_names
normaliser                      -> if opts['standardise data']  (fit on train, applied to both)
run_section('dataset rebalancing', data=train_data)
run_section('model', data=train_data)
run_section('model balancer', model=clf, data=train_data, weight=1)
get_evaluation(...)             -> loops query points, builds an explainer at each, scores it
```

Note the black box is trained on `train_data` but the explainer's local sampling
covariance comes from `test_data`, and evaluation is against `test_data`. That asymmetry
is deliberate — it is the train/eval distribution gap the project studies.

## Sweeping configurations

Put lists in the values and expand:

```python
all_opts = {**opts, 'explainer': ['bLIMEy (normal)', 'bLIMEy (logit)']}
perms = clime.utils.get_all_dict_permutations(all_opts)
title, labels = clime.utils.get_opt_differences(perms)   # what's shared vs what varies
```

`get_opt_differences` returns the constant settings (for the figure title) and the
varying ones (for per-subplot labels) — that's how the notebook builds its legends.

## Caching

`run_pipeline` is `@freezeargs` + `@functools.cache`. Identical `opts` return the
memoised result instantly. Two caveats:

- Freezing **mutates the dict you passed in** (values become `frozendict`/`tuple`).
- The cache does not know about code changes. Restart the kernel when comparing
  behaviour before and after editing a module, or `%autoreload` will lie to you.

## Parallelism

`parallel_eval=True` multiprocesses the per-query-point loop inside
`get_key_points_score`. Worth it for `evaluation points: 'grid'` (400 explainers) and
`'all_test_points'`; overhead-dominated for `'class_means'` (2 points).

## Adding a stage method

Add one entry to the relevant registry dict. The notebook widgets, the permutation
sweeper and `pytest` all enumerate those dicts, so a new entry is offered in the UI and
covered by tests automatically. `test_pipeline.py` runs every method once (holding the
others at their first-listed default) — it asserts runs *complete*, not that values are
correct.
