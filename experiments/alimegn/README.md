# aLIMEgn experiments

The third CLIME thread (`FINDINGS.md` §5): a surrogate's training sample should be aligned
to the black box's own predictions, `P(X, ŷ)`, rather than to the data distribution
`P(X, y)`. Until now it was a one-page framing note in `~/Repos/Overleaf/aLIMEgn/` with no
code. This directory is the code, and the write-up it feeds.

`PREREGISTRATION.md` holds the six predictions, fixed before any sweep ran.

## The framing correction this study starts from

Two of the note's own open questions are settled by reading the pipeline rather than by
experiment, and the answers decide what there is to measure:

- **Sampling can only choose a marginal over `X`.** A surrogate's training labels always
  come from `f`, in every LIME variant, so "train on `P(X, ŷ)`" is automatic for the label
  half. The free choices are the `X`-marginal and the sample weights.
- **Every fidelity metric here already scores against `ŷ`.** `fidelity`, `Brier` and `KL`
  compare `g` to `f`, never to `y`. So `evaluation data` (`'test data'` vs
  `'sample locally'`) contrasts two `X`-marginals, both labelled by `f` — it is *not* the
  `P(X,y)` vs `P(X,ŷ)` contrast that `FINDINGS.md` §5/E2 claimed it was.

`y` versus `ŷ` therefore enters in exactly two places, and those are what the sweeps
manipulate: **class-conditional weighting** (frequencies from `y` or from `ŷ` over the same
points) and **a deliberately degraded `f`**, which is what makes the two differ at all.

## Layout

```
common/      paths.py (where things live), store.py (one cache file per configuration),
             runner.py (parallel over configurations), evaluate.py (one pass per
             configuration: build each surrogate once, score it against everything),
             weights.py (the weighting schemes + their explainers),
             degrade.py (black boxes whose predictions depart from the data),
             register.py (adds all of that to clime's registries without editing clime),
             style.py (shared matplotlib style)
sweeps/      the experiments; each writes results/<name>.json
analysis/    reads the json, prints the findings against the registered predictions
figures/     reads the json, writes figs/*.pdf
results/     merged sweep output, plus cache/ (one file per configuration)
figs/        generated figures, named to match the write-up's \includegraphics paths
tables/      generated LaTeX
logs/        stdout from the sweeps
```

Nothing in `clime/` is modified. `common/register.py` adds this study's black boxes,
explainers and rebalancing options to the registries at import time, which works because
`clime.pipeline.AVAILABLE_MODULES` holds references to the same dict objects. The repo's
test suite and the other experiment directories never import this package, so they are
unaffected.

## Running

```bash
cd experiments/alimegn
./rerun_all.sh                  # every sweep, ~1 h on 16 processes
uv run python analysis/analyse_marginal.py
uv run python analysis/analyse_degrade.py
./make_writeup.sh               # figures and tables, then copy into the Overleaf clone
```

**Always launch sweeps with `OMP_NUM_THREADS=1`** (`rerun_all.sh` does). sklearn's BLAS
threads and the worker processes otherwise oversubscribe the machine.

Every configuration is cached as its own file under `results/cache/<sweep>/`, so an
interrupted sweep resumes, and adding one dataset or one black box costs only the new
cells. Delete a cache file to recompute just that configuration; pass `force=True` to
`run_jobs` to recompute everything.

## The weighting schemes

All six are methods a deployer could actually run — none needs the test set or a label the
deployer would not have.

| explainer key | class frequencies from | needs |
|---|---|---|
| `bLIMEy (normal)` | — (locality kernel only) | nothing |
| `bLIMEy (cost sensitive sampled)` | `ŷ` on the surrogate's own sample | nothing (CIKM'23) |
| `bLIMEy (cost sensitive class)` | `y` over the whole training set | training labels |
| `bLIMEy (local y)` | `y` on nearby training points | training data + labels |
| `bLIMEy (local yhat)` | `ŷ` on those same nearby training points | training data, no labels |
| `bLIMEy (density ratio)` | — (estimated `p_train(x)/p_sample(x)`) | training data, no labels |

`local y` against `local yhat` is the `y`-versus-`ŷ` contrast proper: identical points,
identical mechanism, only the label source differs. The last row is the covariate-shift
reading — if the CIKM trick works because it corrects a shifted `X`-marginal, then
estimating that shift directly should work at least as well.

## What each sweep produces

| sweep | output | what it is |
|---|---|---|
| `sweep_marginal.py` | `results_marginal.json` | 14 datasets × 6 black boxes × 6 schemes, scored under both evaluation marginals at 20 query points (P1, P2, P5) |
| `sweep_degrade.py` | `results_degrade.json` | 6 datasets × (3 clean + 15 label-noise + 4 underfit) black boxes + 2 imbalance settings, plus seeds 1 and 2 on a subset (P3, P4, P6) |
| `sweep_balance.py` | `results_balance.json` | 14 datasets × 3 model families × {normal, balanced training} × {natural, undersampled training data} — `FINDINGS.md` E4, local vs global class imbalance (P7, P8) |
| `sweep_grid.py` | `results_grid.json` | 2-D datasets on a 20×20 PCA grid — 400 query points per configuration, for the heatmaps |

## Three things to know

**Scores are stored per query point, never as a configuration mean.** P1 is a statement
about how a score varies *along* the line; a mean would average it away.

**The local evaluation sample is 500 points, not the 100 of `key_points.get_local_points`.**
Schemes are compared at the same query point, and 100 points put more noise on that
difference than the difference itself in the tails. This is local to this study —
`experiments/logit_lime` is untouched.

**The degradation axis is measured, not assumed.** `local disagreement` in the diagnostics
is the locality-weighted rate at which `f` disagrees with the true label near the query
point, so "how far apart `P(ŷ|x)` and `P(y|x)` ended up" is a recorded quantity rather
than a label-noise setting that may or may not have bitten.
