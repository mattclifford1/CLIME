# Logit-LIME experiments

Everything behind the paper draft in `~/Repos/Overleaf/Logit-LIME/`
(*Logit-LIME: Is there too much faith in the agnosticity of LIME?*). See `FINDINGS.md` §4
in the repo root for the research state, and `PREREGISTRATION.md` here for the prediction
that was fixed before the main sweep ran.

## The question

LIME fits a linear model to the black box's probabilities. Probabilities are bounded,
linear models are not. Does fitting in logit space instead help? Answer: it depends on the
black box, and the dependence is measurable in advance — from the black box alone, before
any surrogate is built.

## Layout

```
common/      style.py (shared matplotlib style), paths.py (where everything lives),
             gradients.py (analytic d/dx logit f for every differentiable black box),
             taylor.py (the explanation-targeted surrogate built from that gradient)
sweeps/      the experiments; each writes one results/*.json and resumes from it
analysis/    reads results/*.json, prints findings, writes tables/*.tex
figures/     reads results/*.json, writes figs/*.pdf and .png
results/     sweep output, plus archive/ (superseded runs kept for provenance)
tables/      generated LaTeX, named to match the paper's \input paths
figs/        generated figures, named to match the paper's \includegraphics paths
logs/        stdout from long sweeps
extra_datasets/  .npz files exported from ~/Repos/toy_datasets (see below)
```

Scripts resolve every path through `common/paths.py`, not through the working directory,
so they can be run from anywhere:

```bash
uv run python sweeps/sweep.py results_taxonomy.json   # writes results/results_taxonomy.json
uv run python analysis/analyse.py                     # reads it back
```

## Running

```bash
uv sync                    # from the repo root; then prefix commands with `uv run`

./rerun_all.sh             # every sweep, from scratch, ~8 h
./make_paper.sh            # every figure and table, then copy into the Overleaf clone
```

`rerun_all.sh` archives the existing `results/*.json` before deleting them, because the
point of a re-run is usually to find out how much something moved the numbers, which needs
both sets. Every sweep resumes from its output file, so an interrupted run is safe to
restart.

`results/archive/` holds runs that are no longer current but are kept for provenance:
`sklearn1.1.3/` is every sweep as it stood before the 2026-08-10 stack upgrade
(`analysis/compare_stacks.py` diffs the two), and `results.json` is the first-generation
sweep of 4 datasets × 7 black boxes, which predates the per-query-point seeding fix
(`FINDINGS.md` B10) and is superseded by `results_taxonomy.json`. Its generator was
removed in the 2026-08-13 tidy-up; recover it from git history if it is ever needed.

`make_paper.sh` takes the Overleaf path as an optional argument and defaults to
`~/Repos/Overleaf/Logit-LIME`. Generated names match the paper's `\input` and
`\includegraphics` paths exactly, so the copy is a mirror with no renaming step to get
wrong.

## What each sweep produces

| sweep | output | what it is |
|---|---|---|
| `sweep.py` | `results_taxonomy.json` | the registered grid: 14 datasets × 12 black boxes × 3 surrogates × 2 metrics |
| `sweep_extended.py` | `results_extended.json` | 29 datasets × 16 black boxes, exploratory; carries the registered rows over rather than recomputing them |
| `sweep_kernel.py` | `results_kernel.json` | the locality kernel width swept over a twentyfold range |
| `sweep_explanations.py` | `results_explanations.json` | do the two surrogates give the same feature ranking? |
| `sweep_explanations_extended.py` | `results_explanations_extended.json` | the same question on the extended grid, exploratory and kept separate for the same reason as `sweep_extended.py` |
| `sweep_ground_truth.py` | `results_ground_truth.json` | for exactly-linear black boxes, score each surrogate against the model's own coefficients |
| `sweep_gradient_truth.py` | `results_gradient_truth.json` | the same question for every *differentiable* black box, against the analytic gradient of its log-odds — and the surrogate's fidelity at the same query point, so the two can be correlated |
| `validate_gradients.py` | — | checks every closed-form gradient against finite differences; run it before trusting the sweep above |
| `sweep_taylor.py` | `results_taylor.json` | the explanation-targeted surrogate (`common/taylor.py`): what an exact explanation costs in fidelity, and how well the same gradient can be estimated from black-box queries alone |
| `sweep_patches.py` | `results_patches.json` | the interpretable-domain transform (`common/patches.py`): surrogates fitted on binary 2×2 patch indicators rather than pixels, scored against the exact patch-space truth |
| `sweep_fidelity.py` | `results_fidelity.json` | the 2×2 of (Brier vs fidelity) × (local sample vs test set) — whether the result survives the CIKM'23 evaluation protocol |
| `sweep_fidelity.py --models …` | `results_fidelity_extended.json` | the same 2×2 for the five extended-grid black boxes that have a gradient truth but were never swept here, run blind against the third pre-registration. **Kept in its own file**: `results_fidelity.json` is the registered 168 and stays that way |
| `sweep_instruments.py` | `results_instruments.json` | what each instrument responds to, on a family of surrogates whose error is prescribed rather than fitted (`common/surrogates.py`) — the arithmetic behind the claim that a 0.5 threshold cannot see confidence |
| `sweep_null.py` | `results_null.json` | the base rate: what an explainer that explains nothing scores under each instrument |
| `sweep_range.py` | `results_range.json` | how far the *reported coefficient* can be carried before a standard LIME surrogate stops being a probability, against the width of the kernel that defined the neighbourhood — plus the flip distance each surrogate implies, against the black box's own |
| `sweep_seeds.py` | `results_seed*.json` | a subset repeated under five random seeds |
| `sweep_diagnostic_checks.py` | `results_diagnostic_checks.json` | what R²_logit is measuring (fifth registration, C1–C5): clip sensitivity, same-space curvature, gradient dispersion (no fit), in-sample advantage, its own salt |
| `sweep_ridge_alpha.py` | `results_ridge_alpha.json` | both surrogates refitted at five ridge penalties on the same neighbourhood; α = 1 reproduces `results_taxonomy.json` exactly |
| `run_full.py` | `results_*_full.json` | every experiment above again on the full grid (`full_grid.py`, 71 datasets × 16 black boxes), plus `results_querypoints_full.json` (20 random test points as query points) and `results_full_seed{1..4}.json` |

The instrument and null sweeps exist because every fidelity number elsewhere is reported
without a floor or a control. `sweep_instruments.py` supplies the control — it moves one
property of a surrogate at a time and records what each instrument does, so "fidelity is
blind to confidence" is a measured zero rather than an argument — and `sweep_null.py`
supplies the floor. Their analysis is `analysis/analyse_fidelity_explanation.py`, which
joins `results_fidelity.json` to `results_gradient_truth.json` per query point (the two
sweeps walk the same 20 points with the same seeds; the join asserts their local Brier
scores agree before trusting it), and `analysis/assess_blind.py`, which scores the third
pre-registration with the blind 70 configurations kept separate from the 84 that had
already been seen.

`sweep_range.py` is about a different question from every other sweep here: not how well a
surrogate reproduces the black box, but what the number it hands the user actually claims.
A standard LIME surrogate is a linear model of a probability, so it *is* a probability only
inside a slab of width `1/||beta||` about its own boundary; whether that slab is narrower
than the neighbourhood it was fitted on is an empirical question, and this answers it over
the registered grid. Its analysis is `analysis/analyse_range.py` (fourth pre-registration),
and `figures/fig_reading.py` and `analysis/table_reading.py` are the single-configuration
illustration. The flip-distance rows are restricted to group A and to query points where
the black box crosses its own boundary exactly once along the feature — the same
non-monotonicity trap the worked-example selection hit, counted rather than assumed away.

Choosing the worked example is itself scripted, in `analysis/select_example.py` and
`analysis/check_example.py`, and both log what they rejected. The filter that does the most
work is monotonicity: the most dramatic candidates are all RBF SVMs whose decision function
turns round inside the neighbourhood, where a surrogate pointing "the wrong way" may be
fitting a second, real boundary of the black box rather than being wrong. Every one of them
is rejected. See the docstrings for why case 3 was added after the fact and what that costs.

Three figure scripts do not read a `results/*.json` either, re-running a single
configuration instead: `figures/fig_justification.py`, `figures/fig_digits.py` and
`figures/fig_patches.py`. The last two are the image-domain pair behind the paper's
"Images: pixels and the interpretable domain" subsection, both on `Digits 3 vs 8` — an 8×8
pixel grid registered in `clime/data/loaders/sklearn_toy.py`, the only registered dataset
whose features have a spatial layout. Both explain a logistic black box, so a ground truth
is available exactly as in `sweep_ground_truth.py`.

`fig_digits.py` explains in raw pixels and draws each surrogate's signed error against the
truth, because at a cosine of ~0.9 the two explanations look alike side by side and the
error maps are what actually separate them. It also shows real examples of both classes
first: a coefficient map means nothing until the reader can see what a 3 and an 8 look like.

**Draw the image panels in raw pixel values, not standardised ones**, and note that this is
not a colour-scale problem — two versions of the figure failed before this was understood.
Standardising divides each pixel by its own standard deviation. The nearly-always-blank
border pixels have a tiny one, so a stray mark there reaches +12 while the strokes that draw
the digit sit between -1 and +3; on a min/max scale those outliers own the colourmap and
every digit renders as flat grey. Switching to a percentile scale restores the contrast but
*not* the shapes, because dividing each pixel separately has already removed the shared
stroke structure that makes a 3 look like a 3 — the information is gone from the values, not
from the colour range. So the script reconstructs the pipeline's standardisation from the
training split and inverts it (which reproduces the pipeline's own arrays exactly, max
difference 0) and draws the images in pixel units. The coefficient panels stay in the
standardised units the surrogate was fitted in.

`fig_patches.py` is the interpretable-domain version, and re-runs the configuration that
`sweep_patches.py` sweeps.

One analysis script does not read a `results/*.json`:
`analysis/table_example_explanation.py` re-runs a single configuration to print one
explanation feature by feature, against the black box's own coefficients, and writes
`tables/example-explanation.tex`. It defaults to the case quoted in the paper and takes
`--dataset`, `--model` and `--point` to look at any other; the contrasting LDA numbers in
the same subsection come from `--model LDA`. The model has to be one of the exactly-linear
family that `sweep_ground_truth.py` uses, since it is their coefficients that supply the
ground truth.

`sweeps/export_toy_datasets.py` freezes 15 further datasets from `~/Repos/toy_datasets`
into `extra_datasets/`, where `clime/data/loaders/exported_npz.py` registers them
automatically. It used to need that repo's own interpreter; since `toy_datasets` became a
declared dependency here (the `datasets` extra) it runs under `uv run` like everything
else. The `.npz` remain because those loaders fetch over the network and cache nothing —
freezing them is what stops a published sweep depending on a UCI endpoint. Re-running the
script overwrites the data behind `results_extended.json`, so pass an output directory if
you only mean to look:

```bash
uv run python sweeps/export_toy_datasets.py /tmp/check
```

## Three things to know

**Everything runs with `parallel_eval=False`.** The differences that matter here are
~1e-3, and that is also the size of the run-to-run nondeterminism parallel evaluation
introduces. Do not switch it on for these experiments.

**A ground truth for the explanation exists more often than it looks.** `coef_` is the
true local importance vector only for exactly-linear log-odds, but its local
generalisation — `d/dx logit f(q)` — is defined for every differentiable black box and
reduces to `coef_` when the log-odds are linear. `common/gradients.py` derives it in
closed form for eleven of them, which is what makes `sweep_gradient_truth.py` possible.
Closed form rather than finite differences because 11% of query points are saturated,
where a difference quotient of `logit f` collapses to zero and the truth would be lost.
Two of the 154 configurations are saturated at *every* query point. There is deliberately
no gradient for the piecewise-constant families: they have none, and that is a result.

**The diagnostic is the point, and it is R²_logit, not the gap.** `sweep.py::diagnostic`
fits an unregularised locality-weighted linear model to the black box's log-odds and to its
probabilities on the same 2,000 locally sampled points. The registered diagnostic was the
gap Δ = R²_logit − R²_p; the fifth registration replaced it with R²_logit alone, which ranks
the advantage better (ρ 0.80 against 0.66 on the extended grid) because R²_p carries
negative information once R²_logit is known. It is the in-sample fit of an unregularised
Logit-LIME, so it does *not* avoid building a surrogate; what it avoids is a held-out
sample and a metric. `diagnostic_detail` also returns `_guarded` values, which skip query
points whose target is constant to rounding: there `weighted_r2` returns a ratio of two
rounding errors, which is where every |Δ| > 1 "degenerate" configuration came from.

**`sweeps/parallel.py` runs any sweep one dataset per process** and merges the shards
(`results/shards/<name>/`, kept as the resume state). It pins BLAS to one thread per
process. Checked against the serial `results_taxonomy.json` on 36 configurations: every
score agrees to 1e-11, except the unguarded R²_logit of degenerate configurations, which is
rounding noise and moves by up to 28 between runs. Against all 464 of
`results_extended.json`, standard LIME and Logit-LIME agree to 1e-15 everywhere except two
configurations of the degree-2 polynomial logistic black box on wide data (Arrhythmia,
Thyroid Sick), and the hard-label logistic-regression surrogate differs by up to 7e-3 in
KL on 20 wide or rank-deficient configurations. Both are lbfgs fits that stop at their
iteration cap on near-separable data, and where they stop depends on the BLAS thread
count. Nothing else in the pipeline is thread-sensitive: the local samples themselves are
bit-identical at 1 and 32 threads, rank-deficient covariance included.

**The synthetic families are generated in `export_toy_datasets.py`, not by toy_datasets.**
Its `GaussianGenerator` re-seeds before each class, so class 1 is an exact translate of
class 0, point for point.

**Figure style is not ad hoc.** `common/style.py` holds the shared matplotlib style. The
three series colours are slots 1–3 of a validated categorical palette, used unmodified —
they clear the colourblind-separation and contrast gates as an all-pairs triple, which
matters because Figure 2 is a scatter plot. Do not substitute colours by eye.
