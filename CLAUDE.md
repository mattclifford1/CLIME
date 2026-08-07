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

```bash
conda activate clime          # python 3.9, sklearn 1.1.3, numpy 1.24.4
```

The env already exists at `~/anaconda3/envs/clime`. `pip install -e .` was used, so the
repo is on the path directly — there is no build step and edits take effect immediately.

**Do not upgrade scikit-learn.** `requirements.txt` pins `1.1.3`; `todo.txt` records a
known incompatibility with 1.2.2 that was never diagnosed. Bumping it will silently
change results across every experiment.

## Running things

```bash
python -m clime.pipeline.make_pipeline     # single hard-coded run, edit opts at bottom of file
jupyter-notebook experiments.ipynb         # widget UI: pick options, click RUN PIPELINE
python experiments/lime_vs_clime-sampling.py   # paper figure scripts
pytest                                     # ~10 min, sweeps every pipeline module once
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
`frozendict` and lists to tuples. Two consequences to be aware of:

- It **mutates the caller's dict in place** while freezing.
- If you change code mid-session in the notebook, `%autoreload` will not invalidate the
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

**Results are not reproducible with `parallel_eval=True`.** `bLIMEy._sample_locally`
draws from the global numpy RNG, so results depend on which worker handles which query
point. Same code, same config, two runs → differences of ~1e-3. Serial runs *are*
deterministic. Use `parallel_eval=False` whenever the effect you are measuring is small
(the Logit-LIME comparisons live in exactly this regime). Tracked as B10 in
`FINDINGS.md`, with the fix written out but deliberately not applied — applying it
changes every number in the repo.

**The `@cache` on `run_pipeline` cannot see code changes.** Restart the kernel when
comparing behaviour before and after editing a module.

**Plot axes are metric-aware.** `clime.evaluation.METRIC_RANGES` declares a fixed range
for bounded metrics and `None` for unbounded ones. A new metric must be added there or
`test_every_metric_has_a_plot_range` fails.

**The query-point line runs class 0 → class 1.** This orientation was arbitrary before
B7 was fixed, so figures regenerated for Breast Cancer come out mirrored relative to the
published version (same content — see `FINDINGS.md`, "Effect on published results").

B1–B9 in `FINDINGS.md` were fixed on 2026-08-07 and each has a regression test. Before
changing any weighting, metric or query-point code, read that section — it records what
the old behaviour was and which published numbers were checked against the change.
