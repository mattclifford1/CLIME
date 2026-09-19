# Plan: replacing the Δ diagnostic, and widening every experiment's datasets

Written 2026-09-18 as a handover. Predictions are in the fifth registration of
`PREREGISTRATION.md`.

## STATUS: executed 2026-09-19. Where execution departed from the plan, the departure is
## the finding.

- The grid is 71 datasets, not ~40 + families: the families were sized at 24 and 9.
- The Gaussian family is generated in `export_toy_datasets.py`, not by toy_datasets,
  whose generator makes class 1 an exact translate of class 0.
- "d ≥ n_test" was the wrong flag for a bad sampling covariance: 11 datasets are rank
  deficient, several from constant or collinear columns (Ionosphere, Thyroid Sick).
- The plan did not foresee that the paper's R² returns rounding noise on a constant
  target. `sweep.guarded_r2` was added, and it explains every |Δ| > 1 exclusion.
- Seeds and kernel widths were run on the full grid, not the registered one.
- Outcome: 13 of 16 tested predictions held (N7 was a report); N8, C1 and C2 failed. See
  `PREREGISTRATION.md` and `FINDINGS.md` §4.

## 1. Section 3.2 (the diagnostic)

What the code does (`sweeps/sweep.py::diagnostic`): at each of the 20 query points, draw
2,000 points with the *evaluation* sampler, weight them with the LIME kernel, fit two
unregularised weighted least-squares lines on the same points — one to f's probabilities,
one to its log-odds clipped at 1e-9 — record each weighted R², average over the points, and
subtract. Three problems with the write-up:

- R² is never defined. The fit, sample and clip are not stated.
- It subtracts fit qualities from two spaces with different variance denominators; the logit
  variance is dominated by the clipped tails, which is what produces the |Δ| > 1 cases the
  paper has to exclude. The outcome is measured in probability space anyway.
- "Needs only the black box, before any surrogate is built" is false. The two fits *are* the
  two surrogates minus the ridge term. What it saves is a held-out sample and a metric.

From the stored results, R²_logit alone beats Δ on every measure (ρ 0.76 vs 0.74 registered,
0.80 vs 0.66 extended; AUC for advantage > 2 of 0.98 vs 0.94 / 0.90), and R²_p has a
negative partial correlation with the advantage given R²_logit.

Rewrite: define weighted R²; make R²_logit the diagnostic; keep Δ once, as the registered
quantity and why it was dropped; report the registered test in Δ as written; replace the
|Δ| > 1 exclusion with a stated cause (the weighted variance of f's probabilities); fix the
intro table's n = 456 against §5's n = 444; drop "needs only the black box".

Checks: C1 clip sensitivity over ε; C2 same-space curvature (diagonal quadratic terms);
C3 surrogate-free gradient dispersion for the 11 differentiable black boxes; C4 in-sample
advantage of the two OLS fits; C5 the diagnostic on its own salt.

## 2. Datasets

Registered grid stays 14 (frozen). Exploratory grid 29 → 71: six new real sets from
`toy_datasets`, three CLIME-registered but unswept, a 24-member Gaussian family
(d × covariance ratio × separation) and a 9-member make_classification family (d ×
informative features). Record d / n_test per dataset and report rank-deficient sampling
covariances separately rather than changing the sampler.

## 3. Work

1. Diagnostic analysis from stored results, with cluster-bootstrap intervals over datasets.
2. Diagnostic checks C1–C5 as one sweep.
3. Rewrite §3.2, §5.2, the prereg addendum, intro table, Discussion threshold, figures with Δ.
4. Export new real datasets to scratch first, diff, copy in only new files.
5. Export the two synthetic families; register predictions before running.
6. Full grid (71 × 16) into a new file, leaving results_extended.json as cited.
7. Widen every subsidiary sweep to the full grid: explanations, gradient truth, Taylor,
   fidelity 2×2, null, seeds (5), kernel (7 widths).
8. Query points: 20 random test points instead of the between-means line.
9. One robustness table and figure: ρ of R²_logit with the advantage by family, dimension,
   imbalance, d / n_test, seed, query-point scheme, kernel width.
10. Bookkeeping: setup section, limitations, README, FINDINGS §4.

Also: a ridge α sweep (the logit target is ~20× the scale of the probability target, so α = 1
is relatively weaker there), and make the paper's rescale-vs-clip wording match the code.

Parallelism: one dataset per process (`sweeps/parallel.py`), one BLAS thread each,
`parallel_eval=False` inside. Verified against the serial `results_taxonomy.json` on 36
configurations: max difference 1e-11, except one degenerate configuration whose R²_logit
is set by floating-point noise in clipped probabilities.
