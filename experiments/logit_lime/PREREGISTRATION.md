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

---

# Third pre-registration: what a thresholded fidelity score cannot see

Written **2026-09-18**, from the plan in `PLAN_fidelity_motivation.md`. Results go in
`results/results_instruments.json`, `results/results_null.json` and
`results/results_fidelity_extended.json`.

**This registration is not uniformly blind, and the parts differ.** Stating which is
which is the point of writing it down.

| part | status when written |
|---|---|
| E-M1, the synthetic surrogate family | predictions below are arithmetic; running it checks the implementation, not the claim |
| E-M2 on the 84 overlapping configurations | **already seen** in an ad-hoc probe. A recomputation, not a test |
| E-M2 on the 70-configuration extension | blind |
| E-M3, the null explainer | blind |
| E-M4, the worked example | a selection protocol, fixed in advance; not a prediction |

## The claim

The study reports Brier and KL where the literature, including
`clifford2023reconciling`, reports fidelity: the locality-weighted agreement between the
class predictions of surrogate and black box, both thresholded at 0.5. The draft
dismisses that instrument in two sentences. The claim it should make instead is narrower
and falsifiable:

> A threshold records one bit per evaluation point — which side of the surrogate's
> boundary it fell on. Everything the surrogate says about *how* confident it is, is
> discarded before the comparison. That is the quantity Logit-LIME changes.

Note what this does **not** say. An earlier version of this claim — "fidelity does not
track explanation quality" — is already refuted by the probe: pooled over query points,
test-data fidelity correlates with cosine-to-truth about as strongly as KL does, and when
it is not tied it names the right surrogate about as often. The defensible claim is about
what fidelity is *blind* to, how often that blindness bites, and which surrogate it
therefore favours.

## E-M1 predictions (arithmetic)

At a query point q, take the family whose explanation error is controlled exactly
(`common/surrogates.py`):

    logit g(x) = c * ( logit f(q) + s * R(theta) grad logit f(q) . (x - q) )

1. **Fidelity is exactly invariant to c.** Multiplying the whole log-odds leaves
   {g = 1/2} where it is, so not one thresholded prediction changes, however over- or
   under-confident the surrogate becomes. Exactly zero range, not approximately.
2. **Fidelity-at-f(q) is exactly invariant to s.** Its threshold moves with the
   surrogate, so {g = f(q)} = {b.(x-q) = 0} is independent of the slope's length.
3. **Both proper scoring rules respond to all three axes,** with their minimum at the
   truth (theta, s, c) = (0, 1, 1).
4. **Thresholded fidelity's response to theta decays as f(q) leaves 1/2,** because the
   black box's boundary leaves the neighbourhood, while the rank-based instruments' does
   not.

A violation of 1 or 2 is a bug, not a finding.

**Correction, recorded rather than quietly fixed.** The plan predicted that fidelity would
be flat in *the slope alone*. That is false, and the first run of E-M1 showed it: scaling
the slope while holding g(q) = f(q) slides the surrogate's class boundary, which a
threshold can see. The exact invariance needs the intercept scaled too, which is why the
family above has separate s and c. The corrected version is the stronger claim — the axis
fidelity is exactly blind to is *confidence at a fixed boundary*, which is precisely the
axis Logit-LIME differs from standard LIME on.

## E-M2 predictions, for the 70-configuration extension only

The extension runs `sweep_fidelity.py` over the five extended-grid differentiable black
boxes (Bagged Logistic, Bayes Optimal, Nearest Class Mean, Polynomial Logistic (deg 2),
RBF Logistic (Nystroem)) x 14 datasets, which have gradient ground truth but no fidelity
sweep. Registered before running:

5. **Ties.** Thresholded fidelity gives standard LIME and Logit-LIME *exactly* the same
   score at ≥ 30% of query points. KL does so at < 10%.
6. **The hard-label surrogate is fidelity's favourite.** Logistic-regression LIME has the
   best test-data fidelity in ≥ 40% of the 70 configurations, while having the lowest mean
   cosine to the truth of the three surrogates.
7. **Fidelity is not uninformative.** Its pooled rank correlation with cosine-to-truth is
   within 0.1 in magnitude of KL's. (Registered so that the section cannot quietly revert
   to the refuted claim.)

## E-M3 predictions: the null explainer

The explainer that explains nothing: g is the locality-weighted mean of f over the same
neighbourhood the surrogates are fitted on — the constant that minimises the local Brier
score — with the zero vector as its explanation.

8. **Mean local-sample fidelity ≥ 0.9** over the registered grid.
9. **It beats or ties standard LIME on local-sample fidelity at ≥ 30% of query points.**
10. **KL ranks it worst of the four at > 95% of query points.**

If 8 fails badly (< 0.8), the off-boundary argument is weaker on this grid than assumed
and the claim must be scoped to confident query points rather than stated generally.

## E-M4: selection protocol for the worked example, fixed in advance

1. Candidates come from the E-M2 join, in one of two shapes: (i) fidelity ranks standard
   LIME at or above Logit-LIME while cosine puts Logit-LIME ahead by > 0.4; (ii) the
   hard-label surrogate has the best test-data fidelity by ≥ 0.02 and the worst cosine by
   ≥ 0.4, with KL ranking it last.
2. Exclude points where ||grad logit f(q)|| < half the configuration's median: there is
   nothing to explain there.
3. Exclude points where f is non-monotone along the transect within the kernel's support.
   A surrogate that points the other way may be fitting a *different* boundary of the black
   box faithfully, which would make the figure an illustration of a non-monotone black box
   rather than of a blind instrument. Plot it; do not infer it.
4. Prefer a two-dimensional dataset, so the surface, the neighbourhood and the surrogates
   can be drawn directly.
5. Take the **median** qualifying candidate, not the most extreme, and report the base
   rate of the shape in the caption.
6. The shape must survive ≥ 4 of 5 seeds and be present at 3 kernel widths. Log every
   candidate tried, including the rejected ones.

---

# Fourth pre-registration: what a probability coefficient claims

Written **2026-09-18**, from the plan in `PLAN_probability_space_motivation.md`. Results
go in `results/results_range.json`.

**Status when written.** The ad-hoc probe in §2 of that plan has already seen
`Gaussian|Logistic`, `Gaussian|MLP`, `Breast Cancer|Logistic` and `Moons|SVM` — four of
the 168 configurations, on the quantities below. Those four are a recomputation. The other
164 are blind. The figure and table are illustrations of one configuration and are not
tests of anything.

## The claim

The paper argues for logit space on fidelity grounds, and separately asserts in
`sec:logitlime` that a log-odds coefficient is the better-posed *reading*. That assertion
is currently argued from definitions alone. It should be measurable, because the defect it
names is arithmetic:

> Standard LIME reports a probability per unit feature. Its surrogate is a linear model of
> a probability, so the surrogate is a probability only inside a slab of width
> $1/\lVert\beta\rVert$ about its own decision boundary. If that slab is narrower than the
> neighbourhood the surrogate was fitted on, then the reading the user is given expires
> inside the region it purports to describe — and no amount of fit quality repairs it.

What this does **not** say: that leaving $[0,1]$ is why Logit-LIME wins on fidelity. It is
not (the SVM column of Figure~\ref{fig:mechanism} leaves $[0,1]$ just as badly and gains
$1.4\times$). The two arguments are independent and the section must keep them apart.

## Predictions, over the 168 registered configurations $\times$ 20 query points

1. **The slab is narrow.** The median kernel-weighted fraction of the surrogate's own
   training sample on which its unclipped output is not a probability is $\ge 0.10$; the
   fraction of query points where that mass is below $0.02$ is $< 15\%$.
2. **It expires inside its own neighbourhood.** The reach — the distance from $q$ toward
   the confident side at which the standard surrogate's output leaves $[0,1]$ — is shorter
   than the locality kernel width $k$ at $\ge 2/3$ of query points.
3. **The reported size tracks saturation, not importance.** For group A (Logistic, LDA;
   exactly linear log-odds, so the true importance vector is constant along the query
   line), at non-saturated points: standard LIME's $\lVert\beta\rVert$ spans $\ge 5\times$
   between its largest and smallest value in $\ge 90\%$ of configurations, and
   Logit-LIME's spans $\le 1.5\times$ in $\ge 90\%$.
4. **The implied counterfactual degrades with confidence.** For group A at non-saturated
   points with exactly one boundary crossing along the feature: standard LIME's flip-distance
   error ratio grows with $\lvert\logit f(q)\rvert$ (pooled Spearman $\ge 0.5$), while
   Logit-LIME's flip distance is within $10\%$ of the truth at $\ge 90\%$ of points.

Whatever comes out is reported. If 3 or 4 fail on LDA but not on logistic regression — LDA
fits a larger slope and saturates sooner — the two are reported separately rather than the
claim being dropped.

Predictions 1 and 2 are about the standard surrogate alone and do not involve Logit-LIME,
so they cannot be satisfied by choosing a favourable comparison. Prediction 3 is one-sided
by construction (a constant truth), which is why it is restricted to group A: nowhere else
is the truth known to be constant.

## Robustness fixed in advance

The reach and the mass both depend on the kernel width $k$ by construction — a narrower
kernel makes the fitted chord more like a tangent and pushes the exit further out in units
of $k$. The statistics are therefore recomputed for `Gaussian|Logistic` and
`Breast Cancer|Logistic` at the extremes of `sweep_kernel.py`'s twentyfold range, and the
range is reported in the caption rather than the single default.

## Outcome (run 2026-09-18, `results/results_range.json`, 168 configurations)

| | registered | measured | |
|---|---|---|---|
| P1a | median mass ≥ 0.10 | **0.136** (quartiles 0.024–0.233) | held |
| P1b | mass < 0.02 at fewer than 15% of points | **23.7%** | **failed** |
| P2 | reach < k at ≥ 2/3 of points | **80.8%** | held |
| P3a | standard span ≥ 5× in ≥ 90% of group A | **35%** | **failed** |
| P3b | Logit-LIME span ≤ 1.5× in ≥ 90% | **100%** | held |
| P4a | standard flip error grows with confidence, ρ ≥ 0.5 | **+0.88** (n = 459) | held |
| P4b | Logit-LIME flip within 10% of truth at ≥ 90% | **85%** | **failed, narrowly** |

**P1b and P3a failed for the same reason, and it is worth more than the predictions were.**
The slab's half-width is 1/‖β‖, so it is *wide* wherever the black box never commits. Split
by the black box's own confidence at q, the median mass is 0.007 for |logit f(q)| < 1,
0.051 for 1–2, 0.090 for 2–4 and 0.235 above 4 (Spearman +0.57 over all 3,360 points). The
defect is not uniform over the grid; it concentrates exactly where a confident prediction is
the reason an explanation was wanted. The registration should have predicted the gradient,
not the average.

P3a additionally turns on two choices made in the registration itself. Eight of the 28
group A configurations are a linear model on data it cannot separate (Abalone Gender,
Circles, Credit Scoring 1, Direct Marketing), where f spans less than 0.5 over the whole
query line and no coefficient has a confidence range to track. And excluding saturated
points — done to protect Logit-LIME from `logit_ridge`'s squash bound — removes the
confident points where the standard coefficient collapses: over *all* points the group A
median spans are 10.3× (standard) against 1.01× (Logit-LIME), and the ≥ 5× fraction is 57%
rather than 35%. Both figures are reported; neither is offered as a restatement of P3a.

P4b missed by 5 points of the registered 90%, on the same saturation boundary: the flip
row is computed on the 459 unsaturated group A points with exactly one boundary crossing
along the feature (101 of 560 dropped as saturated, 0 for the crossing count). The
contrast it was testing is intact — standard LIME is within 10% of the truth at 34% of
those points against Logit-LIME's 85%, and overstates the flip distance by 1.26× at the
median and 4.50× at the ninetieth percentile.

**Kernel-width robustness** (`results/results_range_kernel.json`, scales 0.15 / 0.75 / 3.0,
Gaussian and Breast Cancer with a logistic black box): the mass stays between 0.083 and
0.282. The single cell where the effect vanishes is the two-dimensional Gaussian at the
narrowest kernel — mass 0.083, reach < k at 0% of points — which is the mechanism stated in
the registration behaving as described, not a counterexample: a narrow kernel makes the
fitted chord approach the tangent at q, whose slab is wider in units of k.

---

# Fifth pre-registration: replacing Δ, and a grid wide enough to test it

Written **2026-09-19**, from the plan in `PLAN_diagnostic_and_datasets.md`, before any of
the 42 new datasets below was run through a sweep. Results go in `results/*_full.json`,
`results/results_diagnostic_checks.json` and `results/results_ridge_alpha.json`.

## Status when written

| part | status |
|---|---|
| replacing Δ with $\Rlogit$ | **not blind.** Chosen after reading `results_taxonomy.json` and `results_extended.json` (numbers below) |
| the $\Rlogit > 0.95$ rule | **not blind.** Read off the same two files |
| everything on the 42 new datasets | blind |
| the diagnostic checks C1–C5 | blind; nothing in them has been computed |
| seeds, kernel, query points, ridge α on the full grid | blind beyond the subsets already reported |

## Why Δ is being replaced

Δ = R²_logit − R²_p was registered in the first registration and is what the paper reports.
Recomputed from the two stored grids, with the paper's own degenerate filter:

| | registered (n = 165) | extended (n = 444) |
|---|---|---|
| ρ(Δ, advantage) | 0.74 | 0.66 |
| ρ(R²_logit, advantage) | 0.76 | 0.80 |
| ρ(R²_p, advantage) | 0.36 | 0.47 |
| partial ρ(R²_p, advantage \| R²_logit) | −0.45 | −0.31 |
| AUC for advantage > 2: Δ / R²_logit | 0.94 / 0.98 | 0.90 / 0.98 |

Subtracting R²_p removes information: once R²_logit is known, a *better* probability-space
fit predicts a *smaller* advantage, so the difference is a worse predictor than its first
term. This is not a new idea bolted on: the paper's own §3.2 already says that what caps the
benefit is how close R²_logit comes to 1, and Figure 1 quotes R²_logit, not Δ.

**The registered test is not rewritten.** Statements 1 and 2 of the first registration are
phrased as Δ thresholds and are reported as written. R²_logit is a change of diagnostic,
dated here, and it is tested blind below on data it has not seen.

## A second definitional fact, found while planning

The paper says Logit-LIME rescales probabilities into $[\epsilon, 1-\epsilon]$ with
$\epsilon = 10^{-9}$. `clime/models/logit_regression.py` rescales into
$[10^{-9},\,1 - 10^{-8}]$, so the target runs from −20.7 to +18.4, not symmetrically. The
diagnostic, separately, *clips* at $10^{-9}$ on both sides. Neither is changed (either would
move every published number); both are measured in C1 and the text is corrected to match
the code.

## The new datasets (42)

| family | datasets | why |
|---|---|---|
| real, new from `toy_datasets` | Breast Cancer Prognostic, Cervical Cancer, Framingham CHD, German Credit, HCC Survival, Hepatitis | every tabular set there not already in the grid, less exact duplicates |
| real, registered in CLIME but never swept | Credit Scoring 2, Blobs, Digits 3 vs 8 | |
| **Gaussian family** (24) | $d \in \{2,5,10,30\}$ × covariance ratio $r \in \{1,3\}$ × separation $s \in \{2,4,6\}$ | geometry set by construction: the Bayes log-odds are exactly linear when $r=1$ and exactly quadratic when $r=3$; $s$ is the Euclidean distance between the means, so it does not grow with $d$ |
| **make_classification family** (9) | $d \in \{10,30,60,100,200\}$ × informative $\in \{5, d/2\}$ | where the paper's "above ~60 features both surrogates fail" begins, and whether it is $d$ or the number of informative features |

The Gaussian family is generated with numpy directly, not with `toy_datasets`'
`GaussianGenerator`: that generator re-seeds before each class, so class 1 is an exact
translate of class 0, point for point.

## Predictions

"New" means the 42 datasets above × 16 black boxes = 672 configurations. The 464 already
seen are reported alongside and never pooled into a test.

**N1. R²_logit beats Δ out of sample.** On the new configurations, ρ(R²_logit, advantage)
> ρ(Δ, advantage), and R²_logit's AUC for advantage > 2 is ≥ 0.95 and higher than Δ's.

**N2. The rule transfers.** "R²_logit > 0.95" flags an advantage > 2× with precision
≥ 0.85 on the new configurations.

**N3. Group A is better everywhere.** Logistic, LDA and Nearest Class Mean beat standard
LIME on ≥ 124 of the 126 new (dataset, model) pairs.

**N4. Group D stays at or below break-even**: median advantage < 1 on the new
configurations.

**N5. Linearity, not smoothness, within the Gaussian family.** For the black boxes whose
log-odds follow the data geometry (QDA, Gaussian naive Bayes, Bayes Optimal), the median
advantage at $r = 1$ exceeds that at $r = 3$ in ≥ 10 of the 12 $(d, s)$ cells.

**N6. The diagnostic, not saturation, within the Gaussian family.** Pooled over its 384
configurations, ρ(R²_logit, advantage) ≥ 0.6 and |ρ(saturation, advantage)| < 0.3.

**N7. Separation: no direction registered.** Larger $s$ puts fewer query points near the
boundary but sharpens the sigmoid at the ones that are. The two pull the group A advantage
opposite ways and I cannot say which wins. Reported, not tested.

**N8. Dimension is not what breaks explanations.** On the make_classification family,
where the sampling covariance is full rank at every $d$, Logit-LIME's top-1 agreement with
the gradient truth for group A is ≥ 0.9 at every $d$ up to 200, and standard LIME's is
≥ 0.5. If this holds, the paper's high-dimensional failure is a property of wide, small
datasets (Arrhythmia: 279 features, 136 test rows), not of dimension.

### Diagnostic checks (C1–C5), on the full grid

**C1. Clip sensitivity.** Recomputing R²_logit with $\epsilon \in \{10^{-3}, 10^{-6},
10^{-9}, 10^{-12}\}$: the rank correlation across non-degenerate configurations between
the $10^{-3}$ and $10^{-12}$ versions is ≥ 0.9, and ρ(R²_logit, advantage) moves by < 0.05
across the four.

**C2. Same-space curvature.** The gain in weighted R² from adding diagonal quadratic terms
to the log-odds fit has group A median < 0.01 and ρ(gain, advantage) ≤ −0.3 outside group A.

**C3. A surrogate-free measure.** For the 11 black boxes with analytic gradients, the
locality-weighted relative dispersion of $\nabla \logit f$ over the neighbourhood is 0 for
group A (arithmetic; a violation is a bug) and has ρ ≤ −0.6 with R²_logit.

**C4. In-sample advantage is the ceiling.** The ratio of the two unregularised fits'
in-sample Brier scores has ρ ≥ 0.9 with the measured advantage, higher than R²_logit's.

**C5. Its own random stream.** R²_logit from a sample with its own salt has rank
correlation ≥ 0.98 with R²_logit from the evaluation-salt sample the paper used.

### Robustness of the diagnostic on the full grid

**S. Seeds.** Over 5 seeds, ρ(R²_logit, advantage) is ≥ 0.7 at every seed and within 0.05
of the seed-42 value.

**K. Kernel width.** ρ(R²_logit, advantage) ≥ 0.6 at every kernel scale ≥ 0.3.

**Q. Query points.** With 20 random test points as query points instead of the
between-means line, group A is better on ≥ 95% of its configurations and
ρ(R²_logit, advantage) ≥ 0.6.

**R. Ridge α.** The sign of log(advantage) is unchanged between α = 0.1, 1 and 10 in ≥ 95%
of the configurations swept.

No prediction is made for datasets where the sampling covariance is rank deficient
($d \ge$ the number of test rows). They are reported as their own column.

## Outcome (run 2026-09-19, `results/*_full.json`, scored by `analysis/assess_fifth.py`)

One of the 42 new datasets (`Gauss d2 r3 s4`) went through a smoke test of every sweep
before the main run, which printed its last two configurations. Nothing else was seen.

| | registered | measured | |
|---|---|---|---|
| N1 | ρ(R²_logit) > ρ(Δ); AUC(R²_logit) ≥ 0.95 and > Δ's | ρ 0.83 vs 0.76 (difference 0.07, dataset-bootstrap 95% [0.01, 0.12]); AUC 0.956 vs 0.948 | held |
| N2 | R²_logit > 0.95 ⇒ advantage > 2×, precision ≥ 0.85 | precision 0.99 (129 flagged); recall 0.55 | held |
| N3 | group A better on ≥ 124 of 126 | 126 of 126 | held |
| N4 | group D median advantage < 1 | 0.83 | held |
| N5 | r = 1 beats r = 3 in ≥ 10 of 12 (d, s) cells | 12 of 12 | held |
| N6 | Gaussian family ρ(R²_logit) ≥ 0.6, \|ρ(saturation)\| < 0.3 | 0.83, −0.17 | held |
| N7 | (reported) group A advantage by separation | s = 2: 6.7e5×, s = 4: 57×, s = 6: 9.0× | — |
| N8 | make_classification group A: top-1 ≥ 0.9 (Logit-LIME), ≥ 0.5 (standard) at every d | minimum 0.20 / 0.05 at d = 200 | **failed** |
| C1 | clip ε: rank corr ≥ 0.9, ρ moves < 0.05 | 0.80; ρ 0.55 / 0.75 / 0.80 / 0.82 at ε = 1e-3 / 1e-6 / 1e-9 / 1e-12 | **failed** |
| C2 | curvature gain: group A median < 0.01; ρ ≤ −0.3 outside A | 0.0003; −0.15 | **failed** (second half) |
| C3 | gradient dispersion 0 for A; ρ with R²_logit ≤ −0.6 | 2.5e-30; −0.77 | held |
| C4 | in-sample ratio ρ ≥ 0.9 and > R²_logit's | 0.99 vs 0.80 | held |
| C5 | own-salt vs evaluation-salt rank corr ≥ 0.98 | 0.9996 | held |
| S | ρ ≥ 0.7 at every seed, within 0.05 of seed 42 | 0.80, 0.80, 0.81, 0.81, 0.80 | held |
| K | ρ ≥ 0.6 at every kernel scale ≥ 0.3 | 0.77–0.80 (0.35 at 0.15, not registered) | held |
| Q | random query points: group A better ≥ 95%, ρ ≥ 0.6 | 99.5% of 213, ρ 0.88 | held |
| R | sign stable across α = 0.1, 1, 10 in ≥ 95% | 168 of 168 | held |

**N1 held narrowly on AUC, clearly on ρ.** The registered grid alone could not separate the
two (difference +0.02, [−0.06, +0.10]); the extended grid and the blind new grid both can.

**N2 held on precision and exposes a recall problem the registration did not ask about.**
The rule finds 55% of the new configurations with an advantage above 2×, against 83% on
the seen ones. The misses are mostly exactly-linear black boxes on the synthetic Gaussian
data, saturated over three quarters of the neighbourhood. That is C1's failure seen from the
other side.

**C1 failed, and the failure is the finding.** R²_logit is a statement about the *clipped*
target, and at a coarse clip an exactly-linear black box has median R²_logit 0.69. But
Logit-LIME fits the same clipped target, and in those saturated cases its advantage is also
modest (1.4–10× rather than 10⁵×). So R²_logit at the surrogate's own clip measures what
Logit-LIME can fit; it is not a clean measure of the black box's geometry where it
saturates. The registration assumed the second reading.

**N8 failed, and what it shows is saturation, not dimension.** With 5 informative features,
Logit-LIME's top-1 is 1.00 at every d up to 200 (cosine ≥ 0.99). With d/2 informative it
falls to 0.20 at d = 200, and the two worst cases have 55% and 75% of query points
saturated: more informative features make the classes more separable. The claim that
"dimension is not what breaks explanations" survives; the registered form, "at every d",
did not, because the family was built with separability growing alongside d.

**C2 failed in the half that mattered.** Squared terms add nothing for group A, as they
must, but the gain barely tracks the advantage elsewhere (−0.15; pooled with group A it is
−0.47). How far the log-odds are from linear matters, not whether the departure is
quadratic, which is the registered statement 3 result again.

**Not registered, found on the way:** the paper's R² formula returns a ratio of rounding
errors when the target is constant, which caused every |Δ| > 1 exclusion (fixed by
`sweep.guarded_r2`); α = 1 costs group A an order of magnitude (median 1.1e4× at α = 0.1
against 1.4e3× at α = 1); and the lbfgs-fitted hard-label surrogate and polynomial logistic
black box are BLAS-thread-sensitive on wide data (README).
