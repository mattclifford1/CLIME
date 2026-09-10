# Data

Registered in `AVAILABLE_DATASETS` / `AVAILABLE_DATA_BALANCING` in
[`__init__.py`](./__init__.py).

## Format

A loader returns `(train_data, test_data)`, each a dict:

| key | required | contents |
|---|---|---|
| `'X'` | yes | `(n, d)` float array |
| `'y'` | yes | `(n,)` int array of class labels |
| `'feature_names'` | filled in | list of `d` strings; `check_data_dict` adds `feature 0…d-1` if absent |
| `'costs'` | no | example-dependent cost matrix (costcla datasets) |

Synthetic loaders also return `'means'` and `'covariances'`.

`proportional_split` splits every value that is a per-instance array and copies everything
else — `'feature_names'`, `'description'`, a dataset-level `'costs'` — into both splits
unchanged, so a loader can set them once before splitting. It did not always: they used to
be dropped from the test split, which is the split an explanation is labelled from
(`FINDINGS.md` B18).

**Binary classification only.** Several components (`get_points_between_class_means`,
the fidelity metrics, `bLIMEy.get_explanation`) assume two classes and will raise or
silently misbehave otherwise.

## Layout

```
loaders/      one module per dataset -> raw (train, test) tuple
processing/   normalise, balance, downsample, synthetic proportion sampling
utils/        checkers (format validation) and costs (class/distance weights)
datasets/     the CSVs themselves
tests/        pytest
```

> These `__init__.py` files were previously named `__init__,py` with a **comma**, which
> silently excluded them from built wheels. Fixed — see `FINDINGS.md` B1.

## Available datasets

**Synthetic** (wrapped in `sample_dataset_to_proportions`, so `class_samples` controls
both the size and the class imbalance by undersampling): `Gaussian`, `Moons`, `Circles`,
`Blobs`.

`Gaussian` takes `gaussian_means` and `gaussian_covs` through `data params` and is the
main illustrative dataset — `μ = ±1` gives overlapping classes, `μ = ±3` gives separable
ones. Those are the paper's two synthetic settings.

**UCI / real** (loaded whole, `percent_of_data` only applies to the costcla ones):
`Breast Cancer`, `Banknote Authentication`, `Pima Indian Diabetes`, `Iris`, `Wine`,
`Sonar Rocks vs Mines`, `Abalone Gender`, `Ionosphere`, `Wheat Seeds`.

The first three are the paper's UCI datasets.

**Example-dependent cost** (from [costcla](http://albahnsen.github.io/CostSensitiveClassification/)):
`Credit Scoring 1` (Kaggle 2011), `Credit Scoring 2` (PAKDD 2009), `Direct Marketing`.
These carry a `'costs'` matrix. They are large — keep `percent_of_data` small. Nothing in
the current pipeline actually consumes `'costs'`; they were loaded for a cost-sensitive
direction that was not pursued.

## Processing

- `normaliser` — `StandardScaler` **fit on the training set only**, then applied to both
  splits. Enabled by `opts['standardise data']`. The paper standardises everything;
  `experiments/gaussian_lime_vs_clime.py` deliberately turns it off for distant class
  means, so that the other class stays outside the locality kernel.
- `balance_oversample` / `unbalance_undersample` — the `dataset rebalancing` stage.
- `proportional_downsample` / `proportional_split` — subsample or split while preserving
  class proportions. Seeded from `clime.RANDOM_SEED` by default.

## Weights (`utils/costs.py`)

- `weights_based_on_distance(q, X)` — LIME's exponential kernel,
  `√exp(-‖X−q‖² / k²)` with `k = 0.75·√d`. Used both to weight surrogate training data
  **and** to define "local" in the local fidelity metrics — deliberately the same
  definition on both sides.
- `weight_based_on_class_imbalance(data)` — `1/n_c`, normalised so the smallest weight
  is 1. Returns `[1, 1]` if there is only one class present (which happens routinely for
  query points far from the boundary).
- `weights_based_on_class_either_side_of_prob(data, query_probs)` — same idea but the
  class split is taken relative to the query point's predicted probability instead of
  0.5.
- `get_instance_class_weights(data)` — per instance weights for model training, equal to
  sklearn's `class_weight='balanced'`. Used by all three `*_balanced_training` models.
  (Previously returned the weights swapped — `FINDINGS.md` B2.)
