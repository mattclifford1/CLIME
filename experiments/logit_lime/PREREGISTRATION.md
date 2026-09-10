# Pre-registered prediction

Written **2026-08-07, before running the taxonomy sweep**, and deliberately not edited
afterwards. Results go in `results/results_taxonomy.json`; the outcome is assessed in
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

---

# Second pre-registration: the explanation-targeted surrogate

Written **2026-08-14, before running `sweeps/sweep_taylor.py`**, and not edited afterwards.
Results go in `results/results_taylor.json`.

## The question

`sweep_gradient_truth.py` established that a fidelity score predicts explanation
correctness in direction but not in magnitude, and that logit space improves the
*explanation* of quadratic and smooth black boxes (groups B and C) while giving them no
*fidelity* benefit at all. The account offered for that was a trade: a surrogate can move
slope into intercept to repair its predicted probabilities averaged over the
neighbourhood, at the cost of the slope, which is what the explanation is.

If that account is right, the trade should be visible from the other side. Take the
surrogate that is explanation-optimal by construction — the first-order Taylor expansion
of the black box's log-odds at q,

    logit g(x) = logit f(q) + grad_logit f(q) . (x - q)

whose coefficients are the true local importances exactly — and measure what it costs in
fidelity. Two versions:

  **Taylor (analytic)**  grad from common/gradients.py. Needs white-box access, so it is
  an oracle and an upper bound, not a method. Its cosine to the truth is 1 BY
  CONSTRUCTION and is not evidence of anything.

  **Taylor (finite difference)**  grad estimated from 2d black-box queries. Model-agnostic
  in exactly the sense LIME is, and roughly 5,000x cheaper than LIME's 10,000 samples. Its
  cosine is an estimate, so it *is* a measurement.

## Predictions

1. **Group A: no trade.** Log-odds are exactly linear, so the Taylor expansion is the
   black box, and both Taylor surrogates should match Logit-LIME on fidelity to within
   numerical error while scoring 1.0 on explanation. If Taylor is *worse* on fidelity here,
   something is wrong with the implementation rather than with the idea.

2. **Groups B and C: a real trade, in the direction the account predicts.** The Taylor
   surrogate is exact only at q, so over a neighbourhood spanning curvature it should be
   **worse on local Brier and KL** than a fitted surrogate, while being better on
   explanation. This is the risky prediction: it says the fitted surrogates are buying
   their fidelity with explanation error, and it fails if Taylor wins on both.

3. **The finite-difference version tracks the analytic one except where the black box
   saturates.** 11.3% of query points are saturated, and a difference quotient of logit f
   is flat there. We expect its explanation accuracy to fall well below the analytic
   version's 1.0 on exactly those points, and to be close elsewhere.

4. **No prediction for groups D and E.** They have no gradient, so neither Taylor
   surrogate is defined and neither is run.

## What would falsify the account

Prediction 2 is the load-bearing one. If the Taylor surrogate is better on *both*
explanation and fidelity in groups B and C, then the fitted surrogates are simply worse at
both and there is no trade to describe — the magnitude result of the previous section
would need a different explanation.

## Setup

Unchanged: 14 datasets, the 11 differentiable black boxes, 20 query points between class
means, scored on locally sampled points with local Brier and local KL, run serially.
Finite differences use central steps of h = 1e-3 in standardised units, escalating to
1e-2 and 1e-1 where the quotient underflows to zero.
