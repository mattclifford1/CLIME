# Pre-registered prediction

Written **2026-08-07, before running the taxonomy sweep**, and deliberately not edited
afterwards. Results go in `results_taxonomy.json`; the outcome is assessed in
`FINDINGS.md` §4 and the paper's discussion.

## Why register anything

The first study found that the log-odds linearity gap Δ correlates with how much
Logit-LIME helps (Spearman ρ = 0.77, n = 28). That is a *post-hoc* correlation over seven
black boxes, and it invites the obvious objection that Δ is a curve fit to the models we
happened to try.

The underlying account makes a stronger, falsifiable claim. Logit-LIME fits a weighted
**linear** ridge to `logit f(x)`. So its achievable accuracy is governed by how close the
black box's log-odds are to a linear function of `x` — a property we can derive from each
model's definition, without running anything.

## The prediction

Five of these black boxes have not been tested at all (LDA, Gaussian naive Bayes, QDA,
decision tree, k nearest neighbours). We predict the following **ordering by benefit**,
with reasons:

| group | black boxes | log-odds in `x` | predicted outcome |
|---|---|---|---|
| **A** | logistic regression, **LDA** | **exactly linear** | Logit-LIME recovers the black box to numerical precision; largest Δ; largest benefit |
| **B** | **QDA**, **Gaussian naive Bayes** | **quadratic** | clear benefit over probability space, but *not* exact — a linear surrogate cannot fit a quadratic; intermediate Δ |
| **C** | MLP, SVM | smooth, non-polynomial | modest benefit |
| **D** | **decision tree**, random forest, **k nearest neighbours** | **piecewise constant** | no consistent benefit; Δ ≈ 0 |
| **E** | random forest + Platt, random forest + isotonic | piecewise constant, calibrated | same as D — calibration removes saturation but composes a monotone map with an unchanged step function, so it cannot create linearity |

Gradient boosting is deliberately left unassigned: it is additive *in log-odds* but built
from piecewise-constant trees, so we expect it between C and D but do not commit.

### Specific, falsifiable statements

1. **LDA lands in group A.** Its log-odds are exactly linear, so it must behave like
   logistic regression: Δ > 0.35 and a benefit of at least an order of magnitude. If LDA
   does not, the linearity account is wrong.
2. **Decision tree and kNN land in group D.** Both have Δ ≈ 0 (say < 0.15) and no
   consistent benefit (median ratio < 1.5). These are new model *families*, not variants
   of the random forest, so this is not a re-test of the same thing.
3. **QDA and Gaussian naive Bayes fall strictly between A and D** on both Δ and benefit.
   This is the risky one: the intuitive "smooth models are better" story predicts they
   join group A, whereas the linearity account predicts they cannot be exact.
4. **The group ordering A > B > C > D holds by median benefit.**

### What would falsify the account

- LDA showing no benefit (statement 1 fails), or
- decision tree / kNN showing a large benefit (statement 2 fails), or
- Δ failing to separate the groups while some other property does.

Statement 3 is the one we consider genuinely uncertain, and the most informative either
way: if QDA and naive Bayes come out as good as group A, then what matters is *smoothness*
rather than *linearity*, and the diagnostic should be redefined accordingly.

## Setup

Unchanged from the first study except where noted: 14 datasets, 12 black boxes, 3
surrogates, 2 metrics (local Brier, local KL divergence), 20 query points along the line
between class means, evaluated on locally sampled points, run serially.

Sampling is now seeded per query point (`FINDINGS.md` B10), so results are reproducible
and repeated trials with different seeds are meaningful. Every number in the first study
predates that fix and will shift slightly.
