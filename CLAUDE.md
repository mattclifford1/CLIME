# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

CLIME is a **research testbed for local surrogate explainers** (LIME / bLIMEy), built to
study how the *training* objective of a surrogate model relates to the *evaluation*
objective used to judge it.

It is not a library for end users. It is a configurable pipeline where every stage —
dataset, black box model, explainer, evaluation metric, evaluation locations — can be
swapped, so that combinations can be swept and compared.

It backs the CIKM'23 short paper *"Reconciling Training and Evaluation Objectives in
Location Agnostic Surrogate Explainers"* and two follow-up threads (Logit-LIME,
aLIMEgn). See `FINDINGS.md` for the research state, results and open questions.

## Environment

**uv, not conda.** `pyproject.toml` + `uv.lock` are the source of truth.

```bash
uv sync --extra notebook --extra datasets   # create/refresh .venv (python 3.13, sklearn 1.9, numpy 2.4)
uv run python -m clime.pipeline.make_pipeline
uv run --group dev pytest
```

`uv run` resolves the env itself — there is nothing to activate. The project installs
editable, so edits take effect immediately with no build step.

**Name every extra you want on each `uv sync`.** It makes the environment match exactly
what is asked for, so `uv sync --extra datasets` alone *uninstalls* the notebook packages.

The `datasets` extra is `~/Repos/toy_datasets`, as an editable path source. It is optional
because that path only exists on this machine; without it the repo still resolves and syncs
anywhere. It is also what `clime/data/loaders/exported_npz.py` exists to avoid needing —
see the note on crossing that boundary below.

**Versions are pinned exactly and `uv.lock` is committed** — they decide the numbers, and
this repo's output is results compared across months. Do not delete the lock to "fix" a
resolution problem. `uv` resolves for linux only (`tool.uv.environments`), because shap's
macOS numba requirement caps numpy below the version pinned here.

`requirements.txt` is superseded and kept only for reference; `setup.py` is gone. The old
`~/anaconda3/envs/clime` still exists and is unused.

**On upgrading scikit-learn.** The repo sat on 1.1.3 for years behind an undiagnosed
"incompatibility with 1.2.2" note. It was upgraded to 1.9 on 2026-08-10; the blocker was
six latent bugs, not the science (`FINDINGS.md` B17). If you upgrade again, expect the
same *class* of breakage rather than numerical drift:

- sklearn validates estimator parameters from `fit()` — every `__init__` argument must be
  stored unmodified under the same name. CLIME's models subclass sklearn estimators and
  take a `data` argument, which is exactly the pattern that breaks.
- parameter *types* are validated now (a list where a tuple is wanted is rejected).
- `y` must be an integer array, not `dtype=object`. `check_data_dict` normalises this
  centrally, so new loaders are covered.
- LDA and QDA raise on rank-deficient covariance instead of silently fitting one. Both
  have escalating-regularisation fallbacks; without them, wide datasets fail outright.

## Running things

```bash
uv run python -m clime.pipeline.make_pipeline   # single hard-coded run, edit opts at bottom of file
uv run jupyter-notebook experiments.ipynb       # widget UI: pick options, click RUN PIPELINE
uv run python experiments/lime_vs_clime-sampling.py   # paper figure scripts
uv run --group dev pytest                       # ~10 min, sweeps every pipeline module once
```

`experiments.ipynb` is the primary interface. Cell 3 builds ipywidgets, cell 5 runs every
permutation of the multi-select options, cells 6–8 plot. Selecting several explainers or
metrics produces one subplot per combination.

## Architecture

The whole system is one pipeline (`clime/pipeline/make_pipeline.py::construct`) driven by
a flat `opts` dict. Every stage looks its method up by string key in a registry dict:

| `opts` key             | registry                                        | defined in                    |
|------------------------|-------------------------------------------------|-------------------------------|
| `dataset`              | `data.AVAILABLE_DATASETS`                        | `clime/data/__init__.py`      |
| `dataset rebalancing`  | `data.AVAILABLE_DATA_BALANCING`                  | `clime/data/__init__.py`      |
| `model`                | `models.AVAILABLE_MODELS`                        | `clime/models/__init__.py`    |
| `model balancer`       | `models.AVAILABLE_MODEL_BALANCING`               | `clime/models/__init__.py`    |
| `explainer`            | `explainer.AVAILABLE_EXPLAINERS`                 | `clime/explainer/__init__.py` |
| `evaluation metric`    | `evaluation.AVAILABLE_EVALUATION_METRICS`        | `clime/evaluation/__init__.py`|
| `evaluation points`    | `evaluation.AVAILABLE_EVALUATION_POINTS`         | `clime/evaluation/__init__.py`|
| `evaluation data`      | `evaluation.AVAILABLE_EVALUATION_DATA`           | `clime/evaluation/__init__.py`|

`clime/pipeline/__init__.py::AVAILABLE_MODULES` aggregates all eight. **Adding a new
method means adding one entry to the relevant registry dict — nothing else.** The
notebook widgets, the permutation sweeper and `pytest` all enumerate these dicts, so a
new entry is automatically offered in the UI and automatically tested.

Flow:

```
dataset loader -> (standardise) -> (rebalance) -> model -> (model balancer)
                                                             |
                     for each evaluation point q:            v
                        explainer(clf, train, test, q) -> metric(expl, clf, eval_data, q)
```

`get_key_points_score` (`clime/evaluation/key_points.py`) owns that loop. It decides
*where* explainers are built (`evaluation points`: a line between class means, a PCA
grid, class means, data edges, all test points) and *what they are scored against*
(`evaluation data`: the real test set, or points sampled locally around `q`). This
train/eval-distribution split is the whole point of the project — see `FINDINGS.md`.

### Data contract

Datasets return `(train_data, test_data)`, each a dict with `'X'`, `'y'`,
`'feature_names'`, optionally `'costs'`. `check_data_dict` fills in missing feature
names. Models are sklearn-style: `.predict`, `.predict_proba`. Explainers are
constructed per query point and expose `.predict`, `.predict_proba`,
`.get_explanation()`.

### Caching

`run_pipeline` is `@utils.freezeargs`-decorated then `@cache`d, so repeated identical
`opts` return the memoised result. `freezeargs` recursively converts dicts to
`frozendict` and lists to tuples, building new containers rather than freezing in place
(it used to mutate the caller's dict — `FINDINGS.md` B9).

If you change code mid-session in the notebook, `%autoreload` will not invalidate the
cache. Restart the kernel when comparing before/after a code change.

## Conventions

- Author header comment at the top of each module.
- Registry keys are human-readable strings with spaces, e.g. `'bLIMEy (cost sensitive
  sampled)'` — these strings appear directly as plot labels, so renaming one changes
  every figure legend.
- Explainer variants are thin `*args, **kwargs` wrappers around `bLIMEy` that flip one
  boolean flag; put new variants next to them in `clime/explainer/__init__.py`.
- Metrics take `(expl, black_box_model, data, query_point=..., **kwargs)` and return a
  single float. Peel unused args with `**kwargs`.
- Per-subpackage `readme.md` / `README.md` files document the contract for extending
  that stage — keep them in sync when you change an interface.

## Known traps

**Neighbourhood sampling is seeded per query point, and the salts must differ.**
`clime/utils/seeding.py::rng_from_point` derives a generator from the query point via
`hashlib.sha256`, so sampling no longer depends on call order or on which worker handles
which point (B10). The two draw sites pass *different* `salt` strings —
`'surrogate training sample'` in `bLIMEy._sample_locally`, `'local evaluation sample'` in
`key_points.get_local_points`. Give a new draw site its own salt. Reusing an existing one
makes two sites return identical points; sharing the evaluation salt would score every
surrogate on its own training sample.

**The `@cache` on `run_pipeline` cannot see code changes.** Restart the kernel when
comparing behaviour before and after editing a module. For changes to module-level
constants that are not part of `opts` — the locality kernel width
(`costs.KERNEL_WIDTH_SCALE`), `clime.RANDOM_SEED` — call `run_pipeline.cache_clear()`;
`freezeargs` passes it through explicitly, since `@wraps` does not carry it across.

**`~/Repos/toy_datasets` is a real dependency now (the `datasets` extra), not a forbidden
import.** The old "this env is pinned to sklearn 1.1.3, cross the boundary with data not
code" rule died with the 2026-08-10 upgrade; `toy_datasets` wants numpy ≥2.3.5 / sklearn
≥1.7.2 / scipy ≥1.16.3 / pandas ≥2.3.3 / python ≥3.11 and this env clears all of them.
`~/Repos/projection_models` (sklearn ≥1.6 for `validate_data`) also imports fine but is
not yet declared — `PYTHONPATH` for now.

Adding it cost four light packages and moved no pin, but that is a property of how
`toy_datasets` is packaged and it is easy to lose. Its heavy loaders live behind its own
`image` / `medmnist` / `embeddings` extras, which we do not request, and it imports torch
inside the methods that use it rather than at module scope. Ask for `toy-datasets[image]`
here, or let a module-level `import torch` back into that package, and this repo's lock
acquires the CUDA stack (+36 packages) for datasets it never loads.

**Depending on it does not make `experiments/logit_lime/sweeps/export_toy_datasets.py`
obsolete.** That script survives as a deliberate **snapshot** step: the UCI loaders fetch
over the network at load time and cache nothing, so freezing them to `.npz` is what keeps
a published sweep reproducible against an endpoint that may move or disappear. It now runs
under plain `uv run` rather than that repo's interpreter, and writes into
`experiments/logit_lime/extra_datasets/`, which `clime/data/loaders/exported_npz.py` picks
up automatically — dropping a new `.npz` there registers a new dataset with no code change.
Re-running it **overwrites the data behind `results_extended.json`**; it takes an output
directory argument, so pass one when you only mean to look. A fresh export reproduces 14 of
the 15 files byte-for-byte; XOR differs because it is generated, which is the same reason
`AVAILABLE_DATASETS` keeps live loaders for the synthetic datasets instead of snapshots.
Save string columns as `dtype=str`, never `dtype=object`: an object array is pickled, and
the pickle carries an absolute numpy module path that a different numpy cannot import.

**Plot axes are metric-aware.** `clime.evaluation.METRIC_RANGES` declares a fixed range
for bounded metrics and `None` for unbounded ones. A new metric must be added there or
`test_every_metric_has_a_plot_range` fails.

**Prefer `'KL divergence (local)'` over `'log loss (local)'`.** The log loss metrics are
cross-entropy and carry an irreducible floor equal to the black box's own entropy, which
can be most of the reported number. `'mutual information (RBIG)'` is not a divergence
despite once being called `'KL'`.

**Degenerate neighbourhoods are a real regime, not an edge case.** Far from the decision
boundary the black box predicts one class over the whole local sample. Two surrogates used
to crash there (`FINDINGS.md` B12, B13). Any new explainer or weighting scheme must handle
it; `test_explainer_builds_far_from_the_boundary` enforces this for every registered
explainer.

**The query-point line runs class 0 → class 1.** This orientation was arbitrary before
B7 was fixed, so figures regenerated for Breast Cancer come out mirrored relative to the
published version (same content — see `FINDINGS.md`, "Effect on published results").

B1–B9 in `FINDINGS.md` were fixed on 2026-08-07 and each has a regression test. Before
changing any weighting, metric or query-point code, read that section — it records what
the old behaviour was and which published numbers were checked against the change.
