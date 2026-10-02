# Review of the Logit-LIME thread: code, results and the Overleaf draft

*Written 2026-10-02 against CLIME `e6c53d7` and Overleaf `91421a8` (`~/Repos/Overleaf/Logit-LIME`).*

What was done: every section of the draft and every script under `experiments/logit_lime/`
was read; the analysis scripts were re-run against the stored `results/*.json` (the prose
numbers reproduce, except where noted below); the draft was recompiled; and three small
probes were run from the scratchpad (nothing in the repo was changed, nothing is
registered, 4 datasets × 8 black boxes × 20 query points, the study's own 100-point
evaluation sample). The probe reimplements the two surrogates directly and reproduces the
registered means in `results_taxonomy.json` to 0.0, so its numbers are on the same footing
as the paper's. Probe scripts: `probe.py`, `probe2.py` and the inline Stein check in the
session scratchpad; outputs in `probe.log`, `probe.json`, `probe2.log`.

Short version:

1. **Three generated table captions are corrupted** by Python string escaping, and one
   quoted correlation in the text comes from misaligned arrays. Setup text is wrong about
   the train/test split and "scikit-learn defaults". (§1)
2. **The missing baseline changes the story.** A logistic regression fitted to the black
   box's *soft* probabilities (the KL projection onto the same sigmoid-linear class) beats
   standard LIME on all 32 probe configurations, including every piecewise-constant one,
   and beats Logit-LIME on 26–27 of 32. (§2.1)
3. **Two headline numbers are properties of constants, not of black boxes.** The group-D
   "harm" disappears at ε = 10⁻³; the group-A "six orders of magnitude" moves from 3.5× to
   4×10⁵× as ε goes from 10⁻³ to 10⁻¹², and by 10× with α. (§2.2)
4. **Each surrogate estimates the kernel-averaged gradient in its own space** (Stein's
   identity for a Gaussian design). That makes the "chord, not tangent" remark exact, shows
   the Taylor "trade" was never risky, and shows that standard LIME's poor explanations of
   a *linear* black box are estimation variance, not bias. (§2.3)
5. The fidelity tie rate is half a resolution artefact of 100 evaluation points. (§2.4)

---

## 1. Things that are wrong

### 1.1 Bugs

**R1. Caption escaping in three generated tables.** In a Python source line such as
`r'... Logit-LIME''s advantage ... $\rho$ is Spearman''s ...'` the `''` does not escape an
apostrophe: it closes the raw string, concatenates an empty string, and opens a *plain*
string in which `\r` is a carriage return. Consequences, all present in the Overleaf copies
and in the compiled PDF:

- `tables/diagnostic.tex` (from `analysis/analyse_diagnostic.py:82-83`): renders as
  "Which quantity predicts Logit-LIMEs advantage ... *ho* is Spearmans rank correlation".
  The file contains a literal CR byte.
- `tables/robustness.tex` (from `analysis/analyse_robustness.py:114-118`): `''''` opens a
  triple-quoted string, so the caption contains the next two lines of Python source
  verbatim: `` ``A better is the fraction ' r'of group A configurations ... ``.
- `tables/instruments.tex` (from `analysis/table_instruments.py:63,68,69`): `\'s` inside a
  raw string reaches LaTeX as `\'s`, an acute accent: "the surrogateś class boundary",
  "Spearmanś small non-zero entries".

Fix: write those captions in double-quoted raw strings (`r"...LIME's..."`). The compiled
draft is otherwise clean: no undefined references, two overfull boxes (`tab:findings`,
`tab:main`).

**R2. The pooled fidelity–explanation correlation in the text is computed on misaligned
arrays.** `analysis/analyse_gradient_truth.py::pooled` (lines 46–49, used at 124–132)
filters the KL values and the cosines for NaN *separately* and then truncates both to a
common length. Standard LIME has 23 points with a zero coefficient vector (cosine NaN) and
Logit-LIME 27, so after the first NaN every KL is paired with a neighbouring point's cosine.
The text quotes the misaligned value, ρ = −0.42 with n = 5,948 (`06-results.tex:846-848`);
`figures/fig_fidelity_explanation.py` pairs per point and prints ρ = −0.45, n = 5,977, which
is what the figure shows. The two currently disagree on the same page. Same fix for Brier
(text: −0.38, n = 6,076). The conclusion is unchanged; the number is not.

**R3. Setup section, factual.**
- "each dataset is split 70:30" (`05-setup.tex:13-14`) is false for 8 of the 14 registered
  datasets: the sklearn toy loaders and the costcla loaders split 80:20
  (`clime/data/loaders/sklearn_toy.py:26,47,72,102`, `costcla.py:31`: Breast Cancer 454/115,
  Credit Scoring 1 901/227, Direct Marketing 301/77, Iris, Wine), and Gaussian, Moons and
  Circles draw independent train and test sets of 400 each.
- "All use scikit-learn defaults" (`05-setup.tex:21`): k-NN uses `n_neighbors=15`
  (`clime/models/log_odds_families.py:102`). `MLP_simple` passes `learning_rate='adaptive'`
  (`clime/models/MLP.py:33`), which is a no-op with the default `adam` solver, so the MLP
  *is* default; worth deleting the argument so nobody reads it as a choice.

**R4. The abstract and the README still sell the diagnostic as black-box-only.**
`00-abstract.tex:8` ("A cheap diagnostic computed from the black box alone") and
`experiments/logit_lime/README.md:12` ("measurable in advance — from the black box alone,
before any surrogate is built") contradict `04-problem.tex:117-119` ("it cannot be had from
the black box alone"). The abstract also quotes the extended grid (87/87) where the full
grid (213/213) now exists, and "held on three of four statements" refers to the Δ
taxonomy test, not to R²_logit.

**R5. Two numbers for one quantity.** The null explainer "ties or beats standard LIME on
local fidelity" at 46.0% of points in `sec:blind` and in `assess_blind.py` (P9), and at
"(registered 44%)" in `sec:robustness` (`06-results.tex:356`). If the 44% is over all 168
configurations and the 46% over the 84 with a gradient truth, say which.

**R6. "40 queries per explanation"** (`06-results.tex:910`) is the grid *average* of 2d
including step escalation (`analyse_taylor.py` prints "40 ... on average"); 2d is 4 on
Banknote and 120 on Sonar.

**R7. Fidelity for the SVM compares against `SVC.predict()`**, which is the sign of the
decision function and not `predict_proba() ≥ 0.5` (sklearn documents that the two can
disagree under Platt scaling). `clime/evaluation/faithfulness.py:176-181`. Measured on
2,000-point evaluation samples: 2.9% of points disagree on Gaussian (13.6% at one query
point), 1.1% on Breast Cancer, 1.6% on Banknote, 0.2% on Pima. A surrogate that reproduced
the SVM's probabilities exactly would score below 1, and differences the text leans on
("Wheat Seeds SVM ... fidelity 0.001 worse") are inside this. Threshold the black box's
`predict_proba` in `_get_preds`, or state the convention.

**R8. Constants that should be one constant.** Saturation is `≤ 1e-6` in
`sweeps/sweep.py:127` (the `sat.` column of `tab:main`) and `1e-9` in
`common/gradients.py:349`, `sweep_range.py`, `sweep_gradient_truth.py`; the draft uses one
word for both. The logit clip is `[1e-9, 1-1e-8]` for the surrogate, `1e-9` symmetric for
the diagnostic, `1e-12` in `common/surrogates.py`, `taylor.py`, `patches.py`. The KL metric
adds `1e-7` to both probability vectors (`faithfulness.py:40-43`) and Eq. `eq:kl` does not
say so; it caps a confident error at about 16 nats per point, which is what bounds the
hard-label surrogate's "orders of magnitude worse in the worst case".

**R9. Bayes Optimal underflows.** `clime/models/bayes_optimal.py:48-60` normalises raw
Gaussian pdfs, so far from both means `predict_proba` returns `[0, 0]`. On Sonar this
happens at 1.8% of neighbourhood points on average and 6.6% at the worst query point (0 on
Breast Cancer and Gauss d30). Compute in log space (`logsumexp`). It touches the fidelity
and gradient-truth sweeps, not the main grid.

**R10. Overleaf state.** `main.pdf` in the clone is from Sep 19 00:05 (34 pages); the
current sources compile to 40 pages. Tables are identical to the repo's; the 14 figure PDFs
that `cmp` flags differ only in their embedded timestamps. `make_paper.sh` copies but never
compiles, so push a fresh build or let Overleaf do it.

### 1.2 Numbers that are right but mean less than they read

**The group-A advantage is a function of α and ε, not of the black box.** For an exactly
linear black box Logit-LIME's residual error is pure regularisation and clipping, so the
ratio is unbounded as either goes to zero. Own sweep: median group A 1.4×10³ at α = 1,
1.1×10⁴ at α = 0.1. Probe, logistic black box, median over four datasets, local Brier:

| ε (rescale) | 10⁻³ | 10⁻⁶ | 10⁻⁹ symmetric | 10⁻⁹ / 10⁻⁸ (current) | 10⁻¹² |
|---|---|---|---|---|---|
| advantage | 3.5× | 88× | 4.3×10⁴ | 6.9×10³ | 4.2×10⁵ |

The asymmetric rescale alone costs a factor of six. "Six orders of magnitude" in the
abstract is "whatever the constants were". What is intrinsic is standard LIME's absolute
error (a chord through a sigmoid cannot go below it) and that Logit-LIME's is ≈ 0. Report
those, or cap the ratio.

**The group-D harm is an ε artefact.** C1 swept the *diagnostic's* clip; nothing swept the
*surrogate's*. Probe, median over four datasets:

| black box | Brier, current | Brier, ε = 10⁻³ | KL, current | KL, ε = 10⁻³ | wins of 4, current → 10⁻³ |
|---|---|---|---|---|---|
| Random forest | 0.82 | 1.33 | 0.61 | 1.64 | 1 → 4 |
| Decision tree | 0.87 | 1.01 | 0.43 | 0.97 | 0 → 2 |
| k-NN | 0.79 | 1.22 | 0.31 | 1.21 | 1 → 3 |

At ε = 10⁻³ the logit target is ±7 instead of ±20 and the fitted sigmoid hedges where the
step is uncertain, which Brier rewards. Registered statement 2 ("no consistent benefit")
survives; "logit space actively harms piecewise-constant black boxes" (`sec:groupresults`,
`sec:discussion`) does not survive an ε sweep. Rescale versus clip makes no difference
(0.802 vs 0.802 for the forest).

**The fidelity tie rate is half resolution.** With 100 evaluation points a locality-weighted
agreement is a coarse number. Probe, standard vs Logit-LIME, same 640 query points:

| evaluation points | 100 | 2,000 |
|---|---|---|
| identical fidelity reading | 40.5% | 20.0% |
| all three surrogates exactly 1.000 | 11.4% | 0.3% |

The arithmetic claim (exact invariance to a log-odds rescaling) is untouched and is the
right thing to lead with; "ties at 41% of points" and "perfect at one point in eight" are
mostly `samples=100` in `key_points.get_local_points`.

**The hard-label fallback is not what drives its KL result.** Checked on the registered
differentiable grid: the exact-0/1 fallback occurs at 6.7% of points; excluding them the
hard-label surrogate is still worse than the null explainer at 44.5% of the rest (44.6%
with). The overconfidence is the fitted logistic at `C=1` on near-separable local data,
which is the method; the doc's mechanism paragraph stands. Say `C=1`.

**"Local" is nearly global in the surrogate's own fit.** The effective sample size of the
10,000 kernel weights is 0.77–0.88 of n on the four probe datasets: the kernel removes a
fifth of the N(q, Σ) cloud at most. This is the same point the aLIMEgn write-up makes about
test-set fidelity (ESS 69%), and it is why everything below about *averaged* gradients
matters.

---

## 2. The approach

### 2.1 The missing baseline, and what it does to the framing

Logit-LIME changes two things at once relative to standard LIME: the link (sigmoid of a
linear function instead of a linear function) and the loss (squared error on a clipped
logit instead of squared error on the probability). The surrogate that changes only the
link is a weighted logistic regression on the black box's **soft** probabilities: minimise
Σᵢ wᵢ KL(pᵢ ‖ g(xᵢ)), which in sklearn is `LogisticRegression` on each sample duplicated
with labels 1 and 0 and weights wᵢpᵢ and wᵢ(1−pᵢ). It needs no clip, no rescale, and no ε;
saturated probabilities are just confident labels. `ALIMEGN-REVIEW.md` flagged its absence
in September; it is still absent. Probe (C = 0.5 is the ridge-α = 1 equivalent; C = 10⁴ is
near-unregularised):

| black box | standard → soft C=0.5 | standard → soft C=10⁴ | Logit-LIME (current) for reference |
|---|---|---|---|
| Logistic | 329× | 6.0×10⁴ | 6.9×10³ |
| Nearest class mean | 269× | 1.3×10⁴ | 2.4×10⁴ |
| MLP | 9.9× | 10.3× | 3.9× |
| SVM | 2.4× | 2.5× | 1.27× |
| RF + Platt | 1.63× | 1.64× | 1.22× |
| Random forest | 1.91× | 1.95× | 0.82× |
| Decision tree | 1.16× | 1.17× | 0.87× |
| k-NN | 2.87× | 2.87× | 0.79× |

Medians over four datasets, local Brier; KL is the same picture. It beats standard LIME in
32 of 32 configurations, four of four for every black box including the three
piecewise-constant ones. It beats Logit-LIME in 26 of 32 (C = 0.5) and 27 of 32 (C = 10⁴)
on both Brier and KL, by 1.2–12× on groups B, C, D and E, and by 21× to 2,400× on the two
group-A configurations that saturate (Breast Cancer with logistic regression and with
nearest class mean), which is exactly where the draft says "saturation decides how much".
It loses only on *unsaturated* group A, where ridge on an exactly linear target is
regularisation-limited (the ratios in the previous paragraph). Its explanations are as good
as Logit-LIME's or better except on the SVM (cosine to the gradient: logistic
0.993/0.999 vs 0.996; nearest class mean 0.999 vs 0.909; MLP 0.972 vs 0.968; SVM 0.77 vs
0.81).

Consequences, if this holds on the registered grid:

- "Whether logit space helps is decided by the black box" is partly a property of the
  estimator. With the KL-fitted sigmoid, the sign is positive everywhere tested and the
  question becomes *how much*. The practical recommendation in `sec:discussion` ("if
  R²_logit > 0.95 fit in logit space, otherwise stay in probability space") would change to
  "fit the sigmoid surrogate by cross-entropy; always".
- "Saturation decides how much" becomes partly "the clipped target decides how much".
- The diagnostic keeps its value for the *explanation* question (where the log-odds are not
  linear no linear surrogate is right) and loses most of it for the *fidelity* question.
- It is the first thing a reviewer will ask: why least squares on a transformed target
  rather than the likelihood of the model class you have chosen?

This has to be run properly, registered, on the registered grid before the paper's framing
is fixed. Cost: one explainer entry (`bLIMEy (soft logistic)`), one re-run of `sweep.py`,
`sweep_gradient_truth.py` and `sweep_fidelity.py`.

**Outcome, 2026-10-02 (sixth registration, `PREREGISTRATION.md`).** The probe's numbers
reproduce on its own 32 configurations, but they did not generalise. On the 137 registered configurations it had not seen, the soft-label surrogate
beats standard LIME's local Brier in 84% (registered ≥ 90%, refuted), and 94% on the full
grid. It beats Logit-LIME's KL in 78% (held) and the hard-label surrogate's in 89%
(registered ≥ 90%, missed). It does remove the group D harm (median 1.10×, better in 37/42),
and its edge over Logit-LIME tracks saturation (ρ +0.68). Its explanations are barely better
than standard LIME's (better in 59%, registered ≥ 85%, refuted), while Logit-LIME's beat
both. So the framing in §4 changes: the loss decides the fit, and Logit-LIME's loss also
carries its better explanations. Write-up: `sec:softlabel` in the Overleaf draft.

### 2.2 The estimand: what a kernel-weighted linear fit converges to

For x ~ N(q, Σ) with Gaussian kernel weights exp(−‖x−q‖²/2k²), the weighted sample is
itself Gaussian, N(q, (Σ⁻¹ + k⁻²I)⁻¹), and Stein's identity gives the population
least-squares slope of any target t as **E_w[∇t]**. So standard LIME estimates the
kernel-averaged gradient of p, and Logit-LIME the kernel-averaged gradient of logit p (of
the clipped target, where it clips). Probe 2 confirms it where the dimension is low enough
for 10,000 samples to settle: on Gaussian|SVM the standard coefficients have cosine 0.51 to
the point gradient and 0.999 to E_w[∇p]; Logit-LIME's have 0.61 to the point gradient and
1.000 to E_w[∇logit f]. Pima|QDA, Banknote|Polynomial, Pima|SVM, Wine|MLP give 0.99–1.00.
This is Garreau & von Luxburg's tabular-LIME result in the paper's own setting, and it
organises four things the draft currently treats separately:

1. **The two surrogates disagree because the estimands differ**, by the weight p(1−p)
   inside the average: E_w[p(1−p)∇logit f] vs E_w[∇logit f]. `sec:agreement` can say this
   instead of "trade off the same curvature in different ways".
2. **A fair ground truth exists for each surrogate in closed form**, and `C3` already
   computes it (`gbar` in `sweep_diagnostic_checks.py`). Score each surrogate against its
   own averaged gradient as well as against the point gradient, as `ALIMEGN-REVIEW.md`
   asked. Where the surrogate matches its estimand and the estimand differs from the point
   gradient, the "error" is a choice of neighbourhood, not a defect of the fit.
3. **Taylor prediction 2 was not risky.** The fitted surrogate is the least-squares line
   over the neighbourhood and the tangent is a different line, so the tangent's worse
   fidelity over that neighbourhood is close to guaranteed; prediction 1 is trivial. The
   section's registered-and-confirmed status overstates what it shows. What is informative
   is the *size* of the trade (1.8–3×) and a control that is missing: keep the slope at
   ∇logit f(q) and refit only the intercept. If that recovers most of the fidelity, the trade
   lives in the intercept and the "slope-for-intercept" account is literally right.
4. **For group A both estimands are parallel to β exactly**, so any direction error of
   standard LIME on a linear black box is estimation variance, not bias. Checked: on Breast
   Cancer|Logistic (d = 30) the standard surrogate's cosine to β rises from 0.67–0.79 at
   10,000 samples to 0.87–0.96 at 100,000 across the interior points, and tracks the signal
   E_w[p(1−p)] against a residual sd of about 0.2; on Gaussian and Banknote it is 1.000 at
   every point and sample size. Lowering α makes it *worse* (0.08–0.40 at 10,000), so the
   ridge is helping. The worked case (`06-results.tex:613-616`, "the slope ... is wrong
   unevenly across features") is a low signal-to-noise problem in probability space at a
   confident point, not a wrong slope. That is a sharper and more useful statement: it
   predicts the dimension and saturation dependence the full grid found (N8), it says
   standard LIME can be rescued with roughly 1/(p(1−p))² more samples, and it says Logit-LIME
   needs none of them.

### 2.3 Design choices worth revisiting

- **Query points.** The line between class means extended to the data's per-feature limits
  puts 1,647 of 3,360 registered query points at |logit f(q)| ≥ 4 (`tab:range`), many of
  them outside the data. The reading, null-explainer, tie and worked-example statistics are
  averaged over that set. The random-test-point placement exists and was used only for the
  diagnostic (ρ 0.88); use it for every claim about what a user is handed, and keep the line
  for the mechanism figures.
- **Evaluation sample.** 100 points per query point (`key_points.get_local_points`). Raise
  to 1,000–2,000 for the fidelity cells and the headline ratios; the black box's
  `predict_proba` on 2,000 points is the only cost.
- **Ridge penalty.** α = 1 on a target of scale ±20 and on a target in [0, 1] are different
  penalties. Scale α by the target variance, or report both surrogates at α → 0 as the
  reference and α = 1 as the practical setting.
- **Statistics.** The dataset-level bootstrap in `diagnostic_stats.py` is right; the prose
  still quotes configuration-level Wilcoxon p-values (3×10⁻¹⁵, 2×10⁻¹⁹, 7×10⁻⁸) over 14
  datasets × k black boxes that share each dataset's geometry.
- **The reading section against the average-marginal-effect literature.** Under §2.2 the
  standard coefficient *is* an average marginal effect of p over the neighbourhood, which is
  the quantity social-science practice prefers to log-odds coefficients (Mood 2010 argues
  that log-odds coefficients are *not* comparable across models and samples, the opposite
  direction to `sec:reading`). The section's actual claims survive (the AME cannot be
  carried linearly, its size confounds saturation with importance, the flip distance is
  wrong), but a reader from that literature will not accept "arithmetic nonsense"
  (`sweep_range.py` docstring) or "not a probability" as the headline. Engage with it in a
  sentence; the draft's prose is already more measured than its docstrings.
- **The diagnostic's role.** `sec:diagnostic` now says what R²_logit is. The honest
  competitor is then model selection by held-out local KL (C4 shows the in-sample ratio
  alone reaches ρ 0.99). Present R²_logit as the *explanation* of when logit space can
  work, and the a-priori taxonomy as the evidence; do not present a 0.95 rule with recall
  0.55 as the method.
- **Literature.** `refs.bib` has no entry for Garreau & von Luxburg (AISTATS 2020, and
  "Looking Deeper into Tabular LIME"), for Lundberg & Lee (`KernelExplainer(link="logit")`
  is a logit-space additive surrogate and the closest prior art), for Agarwal et al. (ICML
  2021, LIME/SmoothGrad convergence to an averaged gradient), or for Mood 2010. Check these
  before citing; they are from memory.

### 2.4 What is solid

The registered taxonomy and its honest reporting; the closed-form gradients with the
finite-difference validation; the instruments figure (the best figure in the draft, and the
right way to argue about fidelity); the saturation control with Platt scaling; the
dataset-level bootstrap; the resumable, shard-verified sweeps. None of §2.1–2.3 touches
the claim that exactly linear log-odds are recovered and nothing else is, or that a
threshold at 0.5 is exactly blind to confidence.

---

## 3. Experiments, in order of value per hour

1. **Soft-label logistic surrogate** on the registered grid and the gradient-truth and
   fidelity sweeps, with registered predictions written first (`PREREGISTRATION.md`, sixth).
   Predictions worth writing: beats standard LIME in ≥ 95% of configurations including group
   D; beats Logit-LIME in ≥ 75%; loses to Logit-LIME only where saturation < 5% and
   R²_logit > 0.99.
2. **ε sweep for the surrogate**, `sweep_ridge_alpha.py` style (refit on the same
   neighbourhood): ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²}, symmetric and the current asymmetric
   rescale. Settles group D and gives the group-A ratio its error bar in ε. Item 1 may make
   the transform moot; run this anyway, because the published numbers use it.
3. **Averaged-gradient ground truth** from the existing closed forms, plus the
   intercept-only Taylor control (§2.2, items 2 and 3). Rescore `results_gradient_truth`
   and `results_taylor`; no new black boxes.
4. **Variance test** for group A: cosine against sample size (10³, 10⁴, 10⁵) and against
   E_w[p(1−p)]‖β‖ / residual sd, per query point. One figure; it replaces the mechanism
   paragraph of the worked case.
5. **2,000 evaluation points** for `sweep_fidelity.py`, `sweep_null.py` and the
   instruments probe; re-read ties and the null explainer.
6. **Random test points** for `sweep_range.py`, `sweep_null.py`, `sweep_fidelity.py`.
7. **Fixes**: R1, R2, R7, R8, R9 and the setup text, then `make_paper.sh` and a compile.

Items 1–4 reuse cached black boxes and run in minutes to an hour each; the probe covered 32
configurations in five minutes.

## 4. Narrative

- **Organise the paper around the loss, not the space.** Standard LIME is (identity link,
  squared error); Logit-LIME is (logit link, squared error on the transformed target); the
  soft-label logistic is (logit link, cross-entropy); the hard-label surrogate is (logit
  link, cross-entropy on thresholded labels). The hard-label surrogate wins under the metric
  closest to its loss (fidelity, 80/154) and the KL-fitted one should win under KL; standard
  LIME does *not* win under Brier because the hypothesis class matters more than the loss
  when the black box is sigmoidal. That is the CIKM theme, training versus evaluation
  objective, carried from the sampling distribution to the loss, and it absorbs the
  group-D result instead of needing it explained away.
- **State the estimand once** (§2.2) and let `sec:reading`, `sec:agreement` and `sec:taylor`
  refer to it. It turns a remark into a mechanism, and it is citable.
- **Lead the evaluation story with the arithmetic**, which is exact, and give the tie and
  crown counts after fixing the evaluation sample. Candidate framing (ii) in
  `sec:status`, the evaluation angle, is the result that survives everything above.
- **Retire the orders-of-magnitude headline.** Report standard LIME's irreducible error and
  that the logit-space surrogates reach ≈ 0, and give the group-A ratio with its α and ε
  dependence in a footnote.
- **Keep reading and fit apart**, as the draft already does; add the AME sentence.

## 5. Smaller doc nits

- `sec:groupresults` line 137 quotes Δ ρ = 0.74; `tab:diagnostic` says 0.73; the value is
  0.735. `sec:extended` quotes ρ(R²_logit) 0.79; README and `PREREGISTRATION.md` say 0.80.
- `sec:robustness` "Group A wins everywhere" against `tab:robustness` seed 2 at 99%.
- 153 vs 154 configurations in `sec:gradienttruth` and `tab:fidelityproxy` is never
  explained (one configuration has no finite cosine for the comparison).
- `tab:fidelity` prints 440.57 beside 1.4×10³.
- `fig_reading` panel (c) uses twin axes in different units; the caption does not say so.
- The intro table's group-A row should carry the full-grid count.
- Pending in the working tree from the aLIMEgn session, not touched here: `FINDINGS.md`
  (B19), `clime/explainer/BLIMEY.py`, its README and test, and the alimegn results.
