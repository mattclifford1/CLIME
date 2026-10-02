# Review of the aLIMEgn thread (and the Logit-LIME draft it grew out of)

*Written 2026-09-11, re-checked against the repo and the Overleaf clones on 2026-09-18.*

---

# Second review, 2026-10-02: the code and the Overleaf draft

Checked: everything under `experiments/alimegn/`, the clime modules it leans on
(`clime/explainer/BLIMEY.py`, `clime/data/utils/costs.py`, `clime/evaluation/key_points.py`,
`clime/evaluation/faithfulness.py`, `clime/pipeline/make_pipeline.py`), and the Overleaf
clone (`sections/00..04`, `tables/`, `refs.bib`). The clone is level with `origin/main`, its
figures and tables are byte-identical to the repo generators' output, and all four sweeps
completed with 0 failed configurations. Every numerical claim below was recomputed from
`results/results_*.json` or from a live run; nothing was edited.

## A. Things that are wrong, most important first

### A1. `bLIMEy (cost sensitive class)` reads the TEST split's class balance, not the training set's — **DONE 2026-10-02**

> Fixed as `FINDINGS.md` B19, with two regression tests that fail on the old code. All four
> sweeps were rerun from an empty cache: the other five schemes came out bit-identical and
> the grid sweep unchanged. With the training balance (median 5:1 when undersampled) the
> global scheme now moves the surrogate both ways in the undersampled cells (better in 15/42
> and 18/42, median −0.0018 and −0.0010, local KL worse), so P7 stays confirmed and is now
> a real test. The Logit-LIME session's B20 (fidelity reads the SVM's class as
> argmax(predict_proba)) then landed the same day, and the 70 SVM configurations of the
> marginal and balance sweeps were recomputed under it: only SVM fidelity series moved, no
> verdict changed (P7's undersampled cells become 14/42 and 16/42, median −0.0012 and −0.0018). The write-up's P7
> section, summary, tables and `FINDINGS.md` §5/§7 were rewritten on the final numbers.
> The original finding follows.

`clime/explainer/BLIMEY.py:154` is `costs.weight_based_on_class_imbalance(self.test_data)`.
The registered wrapper only ever receives `train_data` through `**kwargs`, which `bLIMEy`
ignores. Verified live:

| configuration | train counts | test counts | weights from train | weights applied |
|---|---|---|---|---|
| Gaussian, undersample class 0 to 20% | [40, 200] | [200, 200] | [5.0, 1.0] | [1.0, 1.0] |
| Breast Cancer, undersample class 0 to 20% | [33, 285] | [43, 72] | [8.6, 1.0] | [1.67, 1.0] |

Consequences:

- **P7 is untested, not confirmed.** The undersampled cells of the balance sweep never
  exposed the global scheme to the imbalance it exists to use. Its weights are the natural
  test-set imbalance in all four cells, so the "global" column of `tab:balance` is the same
  scheme four times, and "global never helps, even in the cell where global imbalance is
  largest" is vacuous as run. In the stored results its fidelity is bit-identical to
  standard LIME at a median 50 to 55% of query points in both cells.
- The description in `PREREGISTRATION.md`, `README.md` and `02-setup.tex` ("y over the
  whole training set", "needs training labels") is wrong.
- The marginal and degrade rows for this scheme stand approximately, because there train
  and test balance differ only by sampling.
- This predates aLIMEgn. The comment in `clime/explainer/__init__.py` says "BLACK BOX
  TRAINING data", so the intent was train; it did not matter for CIKM because both splits
  shared a balance. It belongs in `FINDINGS.md` as B19 with a regression test.

Fix is one line (use the reference/training split), then `sweep_balance.py` (4.6 min) and
`make_writeup.sh`.

### A2. P1's "unexplained sign" is an instrument artefact, and it is resolved

The mechanism test (`analyse_marginal.py::_balance`) regresses the gap between the two
marginals on `local yhat balance`, which is kernel-weighted over the **test set**. With an
effective sample size of 69% of the test set that quantity is nearly global: its median
range along the line is 0.39, against 0.88 for `sample yhat balance`, the balance of the
surrogate's own 10,000-point sample, which is what the CIKM weights actually read.

Redone with `sample yhat balance`, pooled over the same 1,680 query points:

| one-sidedness measure | Spearman rho | p |
|---|---|---|
| `local yhat balance` (test set, as in the draft) | -0.242 | 1e-23 |
| `sample yhat balance` (the surrogate's sample) | **+0.435** | 2e-78 |

Binned, the gap is about 0 for |balance - 0.5| < 0.3, +0.05 at 0.3 to 0.4 and about +0.15
above 0.4: the published mechanism, with a threshold rather than a slope. Next step 2 of the
conclusion can be closed. `common_analysis.boundary_index` and the P5 "where does class
weighting gain" test use the same regressor and should switch too.

### A3. P3 could not have found a y-versus-yhat effect as designed

Every weighting scheme reads one thing: class frequencies. Symmetric label noise on
balanced classes leaves P(yhat) within sampling error of P(y) even while f disagrees with
the labels at nearly a third of points. Measured from the diagnostics (medians over
configurations, kernel-weighted on the test set):

| mechanism | \|y balance - yhat balance\| | local disagreement |
|---|---|---|
| clean | 0.036 | 0.064 |
| noise 5% / 10% / 20% / 30% | 0.028 / 0.043 / 0.048 / 0.062 | 0.065 / 0.086 / 0.119 / 0.176 |
| noise 40% | 0.114 | 0.298 |
| underfit | 0.114 | 0.182 |
| imbalance | 0.070 | 0.108 |

The quantity the weights read moved by at most 0.06 across most of the ladder, inside the
clean baseline, and the one lever that does move it (imbalance) was tested with the scheme
in A1 broken. On top of that the local frequencies are computed with the same near-global
kernel. "The y versus yhat axis is small" should read "the class-frequency channel for y
versus yhat is small under symmetric noise", and "stop the degradation thread" is premature
on this evidence. *(Correction, 2026-10-02: the imbalance remark above is wrong. P3's contrast
is `local y` against `local yhat`, which never go through the A1 code path; both were
bit-identical in the rerun, and P3's imbalance row is unchanged. Asymmetric noise is still the
open test.)* A cheap honest test: asymmetric noise (flip class 0 to 1 only), which
moves P(yhat) away from P(y) by construction, plus imbalance with A1 fixed.

### A4. The density-ratio surrogate is a surrogate of a different place, and the study never looks

With a logistic discriminator the ratio is exp(b'x + c). Multiplying N(q, S) by it
re-centres the Gaussian at q + S b exactly, and the locality kernel pulls it part of the
way back; the scheme is LIME fitted around a point displaced from q towards the data.
Measured on the surrogate's actual sample (shift of the weighted centroid from q, in
per-feature standard deviations; ESS of the final weights out of 10,000):

| configuration, query point | shift, kernel only | shift, CIKM | shift, ratio | ESS kernel / CIKM / ratio | cos(normal, ratio) | cos(normal, CIKM) |
|---|---|---|---|---|---|---|
| Gaussian / Logistic, end of line | 0.006 | 0.012 | 1.58 | 7707 / 7723 / 2519 | 0.999 | 0.995 |
| Moons / Random Forest, end of line | 0.013 | 0.22 | 1.46 | 7767 / 4818 / 2814 | 0.889 | 0.952 |
| Breast Cancer / Random Forest, q0 | 0.043 | 1.59 | **5.24** | 8846 / 2140 / **132** | **-0.264** | 0.735 |
| Breast Cancer / Random Forest, q3 | 0.048 | 1.80 | 3.67 | 8841 / 3627 / 1051 | 0.381 | 0.872 |
| Breast Cancer / Random Forest, q9 (boundary) | 0.055 | 0.67 | 1.36 | 8841 / 8019 / 5040 | 0.937 | 0.991 |

In 2-D the explanation direction survives; in 30-D at the ends of the line the ratio fits a
ridge on about 130 effective points five standard deviations from q and points the opposite
way to standard LIME. "Best of the six on every column" is in part "fitted where the test
data are": the covariate-shift correction doing exactly what it is for, and exactly the
reason fidelity on test data cannot be the sole judge. This is Logit-LIME's lesson arriving
from the other side, and it should be in the write-up before the ratio is promoted.

### A5. The summary quotes the conditional numbers without saying so — **DONE 2026-10-02** (summary now gives 1.9×, 61/84, then the conditional figures as such)

`00-summary.tex` leads with 0.26 vs 0.10 and 58/67. Those are the 67 configurations whose
test-set variation exceeds 0.1, which is selection on the numerator; the unconditional
figure is 1.88x (61/84) and is what `tab:predictions` reports. Per dataset the picture is
not uniform:

| | var on test | var on own marginal | ratio |
|---|---|---|---|
| Breast Cancer, Pima, Ionosphere, Banknote | 0.31 to 0.63 | 0.09 to 0.15 | 2.9 to 5.3 |
| Gaussian, Iris, Wine, Wheat Seeds, Sonar | 0.15 to 0.39 | 0.08 to 0.17 | 1.7 to 2.8 |
| Moons, Circles, Abalone Gender | 0.12 | 0.12 to 0.24 | **0.5 to 1.0** |
| Credit Scoring 1, Direct Marketing | **0.000 to 0.002** | 0.03 to 0.09 | 0 |

Test sets under 80 points give 2.3x against 1.76x for larger ones, so noise in a max-minus-
min statistic inflates the effect but does not create it. Report the heterogeneity.

### A6. Three configurations are dead cells

Credit Scoring 1 / LDA, Credit Scoring 1 / SVM and Direct Marketing / SVM have a sample
yhat balance of at most 0.02 along the entire line and fidelity identically 1 for every
scheme. They count in every "N/84". Drop or footnote them.

### A7. Pooled medians hide where the headline wins come from (KL on test data)

| dataset | ratio > normal | ratio > CIKM | CIKM > normal |
|---|---|---|---|
| wins by median in | 14/14 | 12/14 | **7/14** |
| notable losses | | Sonar (60-D, 145 train points) -0.05; Ionosphere tie 3/6 | all three 2-D synthetic sets, both costcla sets |
| largest single losses | | Circles / Logistic and / LDA, -0.59 (near-chance black box) | Credit Scoring 1 -0.15 |

The fidelity gains are the robust part of P2; the KL version of "class weighting helps" is
dataset-dependent and should be stated that way.

### A8. Smaller errors in the draft and the records — **DONE 2026-10-02** for the split, the "none uses the test set" line, "best on every column", "−0.001 to −0.001", the 210 seed repeats, the clean-label note, `\today` and the stale counts in `FINDINGS.md` and the README. Not changed: the Wilcoxon independence point and P5's verdict wording

- `02-setup.tex`: "split 70:30". It is 80:20 for Breast Cancer, Iris, Wine and the two
  costcla datasets, 70:30 for Pima, Banknote, Sonar, Ionosphere, Wheat Seeds and Abalone,
  and 50:50 for the synthetic sets (400/400).
- `02-setup.tex`: "none uses the test set". Every scheme samples from N(q, cov(test X))
  (`BLIMEY.py:117`) and the local evaluation sample uses the same covariance; plus A1.
- `03-results.tex`: "best of the six schemes on every column" is false for the own-marginal
  column (ratio -0.073 against global -0.002).
- `03-results.tex`, P7: "-0.001 to -0.001".
- `03-results.tex`: the "210 degraded configurations" include 60 seed repeats of 30
  configurations.
- `local y` in the label-noise cells reads the **clean** training labels (`degrade.py`
  flips a copy), so it is an oracle the weak-label deployer would not have. Say so; it also
  makes yhat's narrow 40% win slightly stronger than it looks.
- The 84 configurations are 14 datasets x 6 models and share splits; Wilcoxon over 84
  treats them as independent and its p-values are optimistic. Block by dataset (14) or use
  a sign test over datasets.
- P5's second half compared a two-valued weight with a continuous one by Spearman, which
  is capped structurally. "Refuted" overstates it; the registered threshold tested the
  wrong thing.
- `main.tex` dates the experiments with `\today`.
- `FINDINGS.md` section 5's opening paragraph and `README.md` still say six predictions,
  three sweeps, 332 configurations; it is eight, four, 500.
- `model_stats` keys are spelled `accurracy` (repo-wide, pre-existing).
- Checked and fine: SVM `predict` disagrees with `round(predict_proba)` on at most 0.5% of
  test points, so the fidelity ceiling is not an issue *(superseded 2026-10-02 by
  `FINDINGS.md` B20: on local samples the disagreement is 0.2–3%, more for the
  balance-trained SVM; fidelity now takes the class as argmax(predict_proba) and the aLIMEgn
  SVM rows were recomputed. Still open: bLIMEy counts class frequencies for the CIKM weights
  from `predict(X_s)` but assigns them by `round(p)`, which differ for the SVMs)*; the salts
  differ; the ratio's
  discriminator labels the reference set 1, so p/(1-p) has the right orientation; the
  rebalancing is applied to the training split only (`make_pipeline.py:51`), as claimed.

## B. Sanity check of the approach

- **The covariate-shift reading is correct, and so is the estimator.** Local fidelity on
  test data scores g against f under a distribution proportional to k(x, q) p_data(x); the
  proposal is N(q, S); the right importance weight is k p_data / N, which is exactly kernel
  times ratio. CIKM's class weights belong to the same family with f's predicted class as a
  one-bit discriminator of "which side of the data is this sample on". That one sentence
  explains both why the trick works and why its rank agreement with the ratio is only
  0.2 to 0.3, and it is a better framing than "crude proxy".
- **The exact target is available without estimation and was not run.** Fit the kernel-
  weighted ridge on the real training points, no Gaussian sample at all. If it ties the
  ratio, the result is "use real data" (Kleinlein's point) and the ratio is scaffolding. If
  the ratio wins, the synthetic sample is buying variance reduction, which is a claim worth
  making. One class in `common/weights.py`.
- **"Local" is doing less work than the words suggest.** The sampling covariance is the
  full data covariance, so the cloud spans the dataset and locality comes only from the
  kernel, which at k = 0.75 sqrt(D) is near-global for D of 8 and up. The kernel sweep in
  the conclusion is the right next step and `costs.KERNEL_WIDTH_SCALE` makes it cheap;
  `experiments/logit_lime/sweeps/sweep_kernel.py` is the template.
- **The pre-registration discipline is good and worth keeping.** Two of its verdicts need
  re-deciding (P7 for A1, P3 for A3), but the table-generated verdicts are exactly what
  makes that cheap. *P7 re-decided 2026-10-02 on the corrected scheme: still confirmed.*

## C. Experiments, in value-per-hour order

1. ~~Fix A1, rerun `sweep_balance.py`, regenerate `tables/balance.tex` and `predictions.tex`.
   P7 becomes a real test. Add the B19 regression test.~~ **DONE 2026-10-02** (B19, all four
   sweeps rerun).
2. Swap `local yhat balance` for `sample yhat balance` in `analyse_marginal.py` and
   `common_analysis.boundary_index`. Analysis only, closes next step 2.
3. Add the real-data-neighbourhood baseline, and report explanation-side measures for the
   ratio alongside fidelity: cosine to standard LIME and to the analytic gradient (closed
   forms for 11 black boxes in `experiments/logit_lime/common/gradients.py`), ESS of the
   final weights, centroid shift. About an hour on the existing marginal grid.
4. Asymmetric label noise for P3: one more `degrade.label_noise` variant, run on the six
   datasets x three bases.
5. Kernel-width sweep, {0.25, 0.5, 0.75, 1.5} x sqrt(D), on the marginal grid. It
   separates marginal mismatch from local-versus-global, which the draft already says is
   confounded.
6. Per-dataset tables and the dead cells dropped.

## D. Narratives the data support

- **The correction relocates the surrogate.** What survives of aLIMEgn is sharper than
  "trained on one marginal, judged on another": any scheme that repairs the test-set score
  does so by moving the fitted neighbourhood from q towards the data, and the ESS of the
  ratio weights at q is a ready-made out-of-distribution diagnostic for a query point. The
  question for a paper is then whether an explanation of a q that lies outside the data
  should be fitted at q, or at the nearest in-distribution region, and who should be told.
- **Fidelity cannot adjudicate that question**, which is the same conclusion Logit-LIME
  reached about training space. One paper, two axes (space and distribution), one shared
  warning.
- **The y-versus-yhat question is not dead, it was never reached.** The frequency channel
  is too narrow to carry it; if it matters anywhere it is through where the black box's
  boundary sits, which is what the imbalance manipulation changes and what the kernel's
  reach dilutes.

# First review, 2026-09-11 (re-checked 2026-09-18)

> **Read this first.** The original review was written **before** the aLIMEgn work was
> coded (commit `586711b`, 2026-09-12) and before the Logit-LIME draft was revised. Most
> of what it proposed has since been run. Each item below is marked with its status as of
> 2026-09-18:
>
> - **STANDS** — verified against the current code/tex today, still true.
> - **DONE** — has since been acted on; the outcome is recorded.
> - **STALE** — was true when written, is no longer true. Kept so the record is honest.

---

## 1. The conceptual correction to the aLIMEgn framing — **STANDS** (and is now adopted)

This was the substance of the review, and it survived. It is now recorded in
`FINDINGS.md` §5 and as the correction following `sections/01-framing.tex` in the
write-up.

The framing note ("sample `X_g` to target `P(X, ŷ)` rather than `P(X, y)`") conflates two
different things:

- **Sampling can only target a distribution over `X`.** A surrogate's training labels
  always come from `f`, so "train on `P(X, ŷ)`" is automatic for the label half in *every*
  LIME variant. The only free choices are the `X`-marginal and the sample weights. That
  answers open question 1: it is `P(X)`, not `P(X, y)`.
  - The Kleinlein et al. (BMVC'22) citation is mischaracterised in the note — they match
    `P(X)` (natural image statistics) too, not `P(X, y)`.
- **Scoring on `'test data'` already scores against `ŷ`.** `fidelity`, `Brier` and `KL`
  all compare `g` to `f`; none looks at `y`. So the `evaluation data` toggle
  (`'test data'` vs `'sample locally'`) contrasts two `X`-marginals, both labelled by `f`
  — it is *not* the `P(X,y)` vs `P(X,ŷ)` contrast. Worth measuring, but a different claim.
  - `FINDINGS.md` §5 and E2 used to state otherwise. Both are now corrected.
- **`y` vs `ŷ` can therefore only enter** through class-conditional quantities
  (`cost sensitive sampled` = ŷ vs `cost sensitive class` = y, i.e. E4) and through a
  black box degraded far enough that the two actually differ (E3).

The defensible version of the thesis: CIKM's `w_c` is a data-free stand-in for the density
ratio between the test distribution and the local sample, built from `ŷ`.

## 2. The experiments the review asked for — **DONE**, and they answered it

Run 2026-09-12/13: `experiments/alimegn/`, six registered predictions, three sweeps, 332
configurations. Write-up in `~/Repos/Overleaf/aLIMEgn/` (9 pages, pushed).

The review's ranked suggestion #4 was "combine E3 and E4, degrade `f`, compare ŷ-derived
against y-derived class weights, and add a learned density ratio as the principled
reference." That is what was run. Findings:

| What was tested | Outcome |
|---|---|
| The CIKM collapse vs evaluation marginal | Belongs to the marginal: fidelity varies 0.26 on test data vs 0.10 on its own marginal (58/67); registered as ≥5×, measured 2.6× — confirmed in direction, not magnitude |
| Class weighting as a marginal correction | +0.034 on test data vs +0.004 on own marginal; worst query point 0.72 → 0.92. Every scheme that repairs one score worsens the other — the signature of a marginal correction |
| Explicit density ratio (the "principled reference") | **Beats the class trick**, using no labels: better local KL in 69/84; worst point 0.94. Rank agreement with the class weighting is only ρ ≈ 0.22–0.30 |
| The `y` vs `ŷ` axis | Small: ~3% in KL at 40% label noise, nothing under underfitting or imbalance, sign seed-unstable in 11/36 repeats; concentrated in piecewise-constant black boxes (RF +0.037, logistic +0.001) |
| Degrading the black box (P6) | **Refuted, informatively.** Degrading *shrinks* the effect (variation 0.35 → 0.12, ρ = −0.44, p = 2e-06; class-weight gain +0.111 → +0.003). What is corrected is `f`'s own confident one-sidedness, which a badly fitted `f` does not produce |
| E4 (P7) | **Settled.** Local imbalance from ŷ helps in all four cells of the 2×2; global imbalance from `y` in none — 19/210 on degraded black boxes, the only one of six schemes that never helps. **STALE:** run with the test split's balance (A1, B19). The 2026-10-02 rerun still confirms P7; the global scheme is better in 29/210 on degraded black boxes and negative by median. |
| Truth objective (P4) | No reversal: ŷ-derived weights also agree better with true labels (129/210 vs 58/210) |
| Methodological note | At `k = 0.75√D` a "local" score over the test set has an effective sample size of 69% of it (median over 14 datasets, up to 91%). CIKM's own instrument is closer to global than its name suggests |

**Net:** the strong aLIMEgn framing does not survive. What survives is "trained on one
`X`-marginal, judged on another" — i.e. covariate shift — and the density ratio is the
better method, with CIKM's class trick as a crude proxy for it.

## 3. What is still open on aLIMEgn

From the write-up's conclusion, plus gaps I noted:

1. **Promote the density ratio from control to method.** It wins every test-data column (not the own-marginal one) and was only
   a reference point. Needs a kernel-width sweep (ratio and kernel may be doing each
   other's work), estimators beyond a logistic discriminator, and — the real gap — a look
   at what it does to the *explanation*. The study never inspects a coefficient.
2. **Explain the sign in P1's mechanism test.** The gap between marginals is larger where
   the neighbourhood is evenly split, the opposite of the published account. Either the
   account or the measure needs amending.
3. **Fix the locality instrument** (the 69% point above): repeat the marginal sweep at
   several kernel widths to separate marginal mismatch from local-vs-global.
4. **Stop the degradation thread.** Three mechanisms, 210 configurations, one surviving
   3% signal confined to label noise and piecewise-constant black boxes.
5. ~~**E4's other half is still unrun**~~ — **STALE**: `sweep_balance.py` crosses normal
   with balanced training, which is that comparison (P7, P8).
6. **Open question 3 is still open** — `y ∈ {±1}` vs `ŷ ∈ [0,1]`. It is the same
   probability-vs-label distinction that motivates Logit-LIME, which is the argument for
   one paper rather than two.
7. **E5 grid heatmaps** remain the strongest unused figure: `evaluation points: grid`
   exists and has never produced one.

**On scope:** both write-ups now reach the same conclusion independently — this is one
paper, "what is a surrogate trained on, and what is it scored against", with Logit-LIME
asking it about the *space* and aLIMEgn about the *distribution*. On its own the aLIMEgn
material is a section, not a paper.

---

## 4. The Logit-LIME critique, item by item, with today's status

### Still standing (verified 2026-09-18)

- **Δ is close to circular.** `experiments/logit_lime/sweeps/sweep.py:76-89` computes it as
  the unregularised weighted least-squares R² of the two surrogates' own fits. It is not
  really "computable before building any surrogate" — computing it amounts to fitting both.
  The reviewer's question is "why not fit both and pick by held-out local KL?" You need to
  show Δ beats that, or present Δ as an *explanation* rather than a *diagnostic*. The
  non-circular evidence is the a-priori taxonomy; lean on that.
- **Δ's sample overlaps the evaluation sample.** `sweep.py:80` draws 2000 points via
  `get_local_points`, which uses salt `'local evaluation sample'` — the same salt as the
  100 points each surrogate is scored on, so the evaluation points are a prefix of Δ's
  draw.
- **The obvious baseline is missing: weighted logistic regression on the soft
  probabilities.** `AVAILABLE_EXPLAINERS` has `bLIMEy (logit)` (ridge on rescaled logits)
  and `bLIMEy (logistic regression)` (ridge... on *rounded* labels) but nothing that fits
  the KL projection directly. Duplicate each sample with labels 0 and 1, weighted `1−p`
  and `p`: no clipping, and it minimises weighted KL by construction. Both current
  variants are handicapped relative to it.
- **The ε is asymmetric and does not match the paper.** `sections/03-problem.tex:67` says
  rescaled into `[ε, 1−ε]` with `ε = 1e-9`; `clime/models/logit_regression.py:13` has
  `MAX_P = 0.99999999`, i.e. `1 − 1e-8`. The diagnostic clips symmetrically at 1e-9
  (`sweep.py:84`). Three different constants. The group-D result ("logit space *harms*
  trees, forests, kNN") is the one most likely to be an ε artefact, because at saturated
  points the rescale value *is* the target — an ε sweep would settle it.
- **The hard-label variant's KL catastrophe is partly an artefact.**
  `clime/models/logistic_regression.py:24-28` falls back to *exactly* 0/1 probabilities in
  a one-class neighbourhood, and sklearn's default `C = 1` is applied to nearly separable
  local data. That is at least as good an explanation as "hard labels are overconfident".
- **Only 100 evaluation points per query point** (`clime/evaluation/key_points.py:127`).
  Ratios like 1.5e11× are ratios of tiny errors estimated from 100 points.
- **Local Brier on the local sample is standard LIME's own training objective** — weighted
  squared error in probability space on the same distribution. That *strengthens* the logit
  wins (you beat LIME on its home turf) and is the CIKM train/eval-objective theme again,
  but the paper never says so.
- **The gradient ground truth is a point gradient; LIME targets a kernel-weighted slope.**
  `∇ logit f(q)` at `q` is not what the kernel-weighted fit estimates. The kernel-average
  of the gradient is the fairer target and is cheap given the existing closed forms. Report
  both the averaged `∇f` and averaged `∇ logit f` — they are not parallel. This matters for
  the B/C "better explanation, no better fidelity" result and for the Taylor comparison,
  which currently wins by construction on the pointwise target.

### Since fixed, or overtaken — **STALE**

- Typos "AgnosticSurrogate", "prediciting", "probabilites" — gone.
- "The intro's contribution list omits gradient truth / fidelity-as-proxy / Taylor" — the
  intro now carries a full claims table covering all of them, with registered/refuted
  status per row.
- "Δ ≳ 0.35 is asserted as a rule" — `sections/06-discussion.tex:27` now says explicitly
  that the threshold is descriptive, not derived. *Worth one re-check*: `04-setup.tex:106`
  still defines group A by `Δ > 0.35` while the extended grid's group-A median Δ was 0.282.
- Number drift between `FINDINGS.md` and the paper — the numbers were reconciled in commit
  `505b34e`. The underlying recommendation stands: generate prose numbers from the results
  JSON rather than typing them.

### Unrechecked (were true in the pre-revision draft)

- The cross-entropy-floor paragraph is still commented out at
  `sections/02-background.tex:104-105`; check nothing in the intro still promises it.
- "Six orders of magnitude" (abstract) vs "sixteen orders" (Taylor caption) — both still
  present, but in different scopes, so possibly fine now.
- "40 black-box queries" is `2d`, which equals 40 only at `d = 20`.
- Abstract length (~450 words in the old draft).

---

## 5. Cheap experiments that would most protect the claims

Hours, not days, and all reuse `experiments/logit_lime/sweeps/sweep.py`:

1. **Soft-label logistic surrogate** (the missing KL-projection baseline), plus an **ε
   sweep** for the logit ridge and a **C sweep** for the hard-label variant.
2. **Kernel-averaged gradient ground truth** from the existing closed forms in
   `experiments/logit_lime/common/gradients.py`, reporting both averaged targets.
3. **Δ against "fit both, choose by held-out local KL"** — the direct answer to the
   circularity objection.
4. **More evaluation points** (1000 rather than 100) for the headline ratios.
