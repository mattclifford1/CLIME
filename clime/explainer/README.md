# Explainers

An explainer is constructed **per query point** and behaves like a small sklearn model
afterwards. Register new ones in `AVAILABLE_EXPLAINERS` in
[`__init__.py`](./__init__.py) — the key string is what appears in the notebook widget
and in every plot legend.

## Interface

Called by the pipeline as:

```python
expl = explainer(black_box_model=clf,
                 query_point=q,          # 1-D numpy array
                 train_data=train_data,  # data dict
                 test_data=test_data)    # data dict
```

Peel off anything you don't need with `**kwargs`. Must then provide:

| method | returns |
|---|---|
| `predict(X)` | hard class predictions, shape `(n,)` |
| `predict_proba(X)` | class probabilities, shape `(n, 2)` |
| `get_explanation()` | feature importances, shape `(n_features,)` |

Only `predict` / `predict_proba` are exercised by the fidelity metrics.
`get_explanation()` is the actual human-facing explanation and is currently **broken for
the logit surrogate** — see `FINDINGS.md` B4.

## bLIMEy

[`BLIMEY.py`](./BLIMEY.py) is the workhorse. It is a simplified
[bLIMEy](https://arxiv.org/abs/1910.13016) with no interpretable-domain transform,
i.e. LIME applied directly in the tabular feature space. Steps:

1. Sample `n=10000` points from `N(q, Σ)` where `Σ` is the covariance of `test_data`
   (identity if no data given).
2. Label them with the black box: `y = f.predict(X_s)`, `p = f.predict_proba(X_s)`.
3. Build sample weights (below).
4. Fit a surrogate regressor on `(X_s, p)` with those weights.

All variants are one-line wrappers in `__init__.py` that flip a boolean:

### Weighting flags

| flag | effect |
|---|---|
| `weight_locally=True` *(default)* | exponential distance kernel `w_x`, width `0.75·√|D|` |
| `class_weight_sampled` | multiply by `w_c ∝ 1/n_c` over the **sampled** labels `f(X_s)` — this is the CIKM paper's `w_xc` |
| `class_weight_data` | multiply by `w_c` over the **black box's training data** class counts |
| `class_weight_sampled_probs` | as `class_weight_sampled`, but "class" means *above/below the query point's predicted probability* rather than above/below 0.5 |
| `rebalance_sampled_data` | oversample the minority class in `X_s` instead of weighting (registered but commented out) |

`class_weight_sampled` vs `class_weight_data` is a substantive experimental contrast —
local vs global imbalance as the signal. See `FINDINGS.md` §3 and E4.

### Surrogate model flags

| flag | surrogate | notes |
|---|---|---|
| *(default)* | `sklearn.linear_model.Ridge` | regresses probabilities directly; unbounded, can predict outside `[0,1]` (clipped at predict time) |
| `train_logits` | `clime.models.logit_ridge` | ridge in **logit space**, sigmoid on the way out |
| `logistic_regression` | `clime.models.logistic_regression` | sklearn logistic regression on **rounded** probabilities, i.e. on hard labels |

The latter two are the Logit-LIME thread. Their coefficients are on different scales from
each other and from the default, so don't compare `get_explanation()` outputs across
them without normalising.

## Other explainers

- [`LIME.py`](./LIME.py) — `LIME_fatf`, the reference implementation via
  [FAT-Forensics](https://fat-forensics.org/), *with* the discretiser/binariser
  interpretable domain. Use it to sanity-check bLIMEy. The second class in the file
  (`LIME`, wrapping `lime.lime_tabular`) is an unfinished stub and does not run.
- [`SHAP.py`](./SHAP.py) — `kernal_SHAP`, Kernel SHAP. Requires access to the training
  data, unlike bLIMEy. Slow; expect long runs when it is selected in a sweep.
