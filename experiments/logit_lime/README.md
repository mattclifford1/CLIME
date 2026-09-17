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
| `sweep_seeds.py` | `results_seed*.json` | a subset repeated under five random seeds |

Two figure scripts do not read a `results/*.json` either, re-running a single configuration
instead: `figures/fig_justification.py` and `figures/fig_digits.py`. The latter is the only
figure here on data whose features have a spatial layout — `Digits 3 vs 8`, an 8×8 pixel
grid registered in `clime/data/loaders/sklearn_toy.py` — which lets an explanation be shown
as an image rather than a bar chart. It explains a logistic black box, so `coef_` is the
ground truth exactly as in `sweep_ground_truth.py`, and it draws each surrogate's signed
error against that truth: at a cosine of ~0.9 the two explanations look alike side by side,
and the error maps are what actually separate them.

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

**The diagnostic is the point.** `sweep.py::diagnostic` fits a locality-weighted linear
model to the black box's log-odds and to its probabilities on the same locally sampled
points, and returns the gap in weighted R². That gap predicts whether Logit-LIME will help
(Spearman ρ = 0.74, n = 165) and needs only the black box, not any surrogate.

**Figure style is not ad hoc.** `common/style.py` holds the shared matplotlib style. The
three series colours are slots 1–3 of a validated categorical palette, used unmodified —
they clear the colourblind-separation and contrast gates as an all-pairs triple, which
matters because Figure 2 is a scatter plot. Do not substitute colours by eye.
