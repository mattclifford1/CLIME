# Plan: motivating Brier/KL over thresholded fidelity

Written 2026-09-18 as a handover.

## STATUS: executed 2026-09-18. This file is kept as the record of what was
## planned; where the execution departed from it, the departure is the finding.

Everything below was carried out. What changed on contact with the data:

- **§3's scale-invariance claim was stated wrongly here, and E-M1 caught it.** Fidelity
  is *not* flat in the slope: scaling the slope while keeping `g(q) = f(q)` slides the
  surrogate's class boundary, which a threshold can see. The exact invariance is under
  scaling the *whole log-odds* — confidence at a fixed boundary — which is the stronger
  claim and the axis Logit-LIME actually moves on. The probe now has three axes, not two.
- **E-M2's registered prediction 7 failed**, in fidelity's favour: its level correlation
  with cosine is *higher* than KL's (+0.44 vs −0.21 on the blind 70), not within 0.1.
  The section says so.
- **E-M4's case 1 turned out to select saturated points** where Brier is blind too, so a
  case 3 was added after the fact and both are reported — case 1's as the honest caveat,
  case 3's as the motivating figure. The addition is recorded in
  `analysis/select_example.py` and `PREREGISTRATION.md` rather than smoothed over.
- **The monotonicity filter rejected every case-2 candidate and every SVM**, which is a
  result: the dramatic "fidelity crowns a backwards explanation" points on this grid are
  all non-monotone black boxes, not blind instruments.
- **The strict worked-example shape is rare** (≈1% of points at the loosest threshold,
  0.07% at the strictest), so the figure illustrates a mechanism and the table carries the
  frequency. The systematic result — fidelity crowning the hard-label surrogate in half of
  all configurations — is the load-bearing one.

Delivered: `sweeps/sweep_instruments.py`, `sweeps/sweep_null.py`, `--models` on
`sweeps/sweep_fidelity.py`, `common/surrogates.py`, `analysis/analyse_fidelity_explanation.py`,
`analysis/assess_blind.py`, `analysis/select_example.py`, `analysis/check_example.py`,
`analysis/table_instruments.py`, `analysis/table_fidelity_proxy.py`,
`figures/fig_instruments.py`, `figures/fig_blind.py`; `sec:metrics` rewritten and
`sec:blind` added in the Overleaf draft; third registration in `PREREGISTRATION.md`;
`FINDINGS.md` §4.

---

The original plan follows unchanged.

## 1. What the section has to do, and what it must not claim

The Logit-LIME draft currently dismisses thresholded fidelity in two sentences of
`sections/02-background.tex` (`sec:metrics`) and one results subsection
(`sec:fidelity`, the 2×2 protocol table). The user wants a short motivation, with
illustrative examples, that (a) shows *why* agreement-at-0.5 is the wrong instrument
for this study and (b) shows the effect on the *explanations*, not only on the
probabilities. It goes in / next to `sec:metrics` ("Evaluating a surrogate" — the
"replicating the model's probability output" section), with the numbers in the
results.

**The probes in §2 change what the section can honestly say.** The naive story —
"fidelity does not track explanation quality" — is *refuted* by data already on disk:
pooled over points, thresholded fidelity on the test set correlates with cosine-to-truth
about as strongly as KL does (ρ ≈ +0.39 vs −0.33), and when it is not tied it names the
right winner as often as KL. So the section must make the narrower, defensible claim:

> A threshold at 0.5 records one bit per point: which side of the surrogate's boundary
> the point falls on. It is therefore (i) exactly invariant to the *scale* of the
> surrogate's coefficients, (ii) exactly invariant to *everything* about the surrogate
> when the black box's boundary is outside the neighbourhood, and (iii) unable to
> penalise overconfidence. The consequences on our grid: it ties on ~40% of paired
> comparisons, gives every surrogate a perfect score at ~20% of query points where the
> surrogates' explanations still differ, compresses four-order-of-magnitude
> differences into a few percentage points, and ranks the hard-label surrogate — the
> one with the *worst* explanations — first more often than any other. Proper scoring
> rules use the discarded information. They are not a ground truth either
> (Section `sec:fidelitypredicts`), which is why the paper scores explanations directly
> wherever a truth exists.

Every experiment below serves one clause of that paragraph. If a result contradicts a
clause, drop the clause; do not tune the experiment.

## 2. What already exists (reuse, do not rebuild)

| asset | what it gives us |
|---|---|
| `results/results_fidelity.json` | per-query-point `fidelity (local)` and `Brier (local)` for the 3 surrogates × 168 registered configs × {local sample, test data}. 20 points each, `between_class_means` |
| `results/results_gradient_truth.json` | per-query-point cosine / rank ρ / top-1 to the analytic ∇logit f, plus Brier and KL, for the same 3 surrogates on 154 configs (11 differentiable black boxes) |
| the join of the two | **free**. Same seeds, same 20 points; probe verified Brier|local agrees to 0.0 between the files. 84 configs overlap (14 datasets × Logistic, LDA, QDA, GNB, MLP, SVM). The other 5 gradient-truth models are extended-grid only and have no fidelity sweep |
| `figures/fig_justification.py` | the 1-D transect + explanation bar-chart figure for Breast Cancer|Logistic pt 10; template for the worked example |
| `analysis/analyse_fidelity.py`, `table_fidelity.py` | the existing 2×2 analysis; extend rather than duplicate |
| `analysis/analyse_gradient_truth.py::load` | row loader with degenerate-point handling; reuse for the join |
| `common/gradients.py` | analytic ∇logit f — the ground truth for E-M1 and E-M4 |
| `common/style.py` | palette (`INK`, `BLUE`, `ORANGE`, `MUTED`…). Do not pick colours by eye |
| registry metrics | `'fidelity (local)'`, `'fidelity (local query probs)'` (threshold at f(q) instead of 0.5), `'spearman'`, `'Brier score (local)'`, `'KL divergence (local)'` — the whole "instrument ladder" is already implemented |
| `sections/05-results.tex` `sec:hardlabel`, `sec:fidelity`, `sec:fidelitypredicts` | already say: hard-label variant is overconfident; fidelity compresses the effect ~10³×; KL/Brier rank but do not size. **Cross-reference, do not repeat** (the write-up is a lean working doc — see memory) |

### Probe results (2026-09-18, ad hoc; recompute in E-M2)

Join of the two result files, 84 configs, 1,680 query points, standard vs Logit-LIME
paired:

| quantity | fidelity, local sample | fidelity, test data | KL | Brier |
|---|---|---|---|---|
| level: ρ(instrument, cosine), pooled | +0.23 | +0.39 | −0.33 | −0.30 |
| exact ties between the two surrogates | 40% of points | 38% | 2% | ~0% |
| right winner when not tied (\|Δcos\| > 1e-3) | 68% | 74% | 64% | — |
| magnitude: ρ(gain, Δcosine), paired | +0.04 | +0.13 | −0.16 | −0.15 |
| configs where instrument says hard-label surrogate is best | 43/84 | 44/84 | 21/84 | 22/84 |

Mean cosine to truth: standard 0.861, Logit-LIME 0.893, **hard-label 0.827** (top-1:
0.71 / 0.79 / 0.67). Points where *all three* surrogates score fidelity = 1.000 on the
local sample: 353 of 1,680 (21%); across those points the spread of cosine between the
three surrogates has median 0.08 and exceeds 0.2 at 38% of them.

Per group, test-data fidelity vs KL, right winner when not tied: A 97% vs 83%,
B 59% vs 46%, C 73% vs 69%. **KL is a coin flip on group B.** Say so.

Two cautions found while probing, both of which must survive into the write-up:

1. **Points where all instruments fail exist.** Moons|SVM pt 1: standard cos 0.15,
   logit 0.99, fidelity identical (0.925 / 0.995), and KL *slightly prefers standard*
   (5.6e-2 vs 6.0e-2). Proper scores are better instruments, not oracles.
2. **A high-fidelity surrogate pointing the wrong way may be telling the truth about a
   different boundary.** Gaussian|SVM pt 5: hard-label surrogate fidelity 0.986 (best),
   cosine **−1.00**; the true gradient there is not small (‖∇‖ = 3.15 vs config median
   1.68). An RBF SVM's decision function decays back toward its bias far from the data,
   so the neighbourhood may contain f's *flip-back* boundary, which the hard-label fit
   captures faithfully while the gradient at q points the other way. Before using any
   such point as an illustration, plot f along the transect through q and check f is
   monotone across the kernel's support. If it is not, the example shows a non-monotone
   black box, not a blind instrument. (The fallback for a one-class neighbourhood is
   zero coefficients → cosine NaN, so −1.00 is a genuine fit, not the B12 fallback.)

## 3. The precise claim (write this out once, in the paper, in ≤ 6 lines)

For a binary surrogate g and black box f, locality weights w over evaluation points,

    fidelity = Σ w_i 1[ sign(g_i − ½) = sign(f_i − ½) ] / Σ w_i .

- **Scale invariance.** For a linear surrogate the set {g = ½} is unchanged by
  (β, β₀−½) → s(β, β₀−½), s > 0. So fidelity cannot see ‖β‖. In logit space ‖β‖ *is*
  the reading the paper argues for ("a unit increase multiplies the odds by e^β",
  `sec:problem`).
- **Off-boundary invariance.** If sign(f − ½) is constant on the support of w, fidelity
  is 1 for every g with the same sign there, regardless of β's direction. This is most
  of LIME's use: query points are usually confident predictions, not boundary cases.
- **Near the boundary it *does* constrain direction** (the wedge between the two
  hyperplanes carries kernel mass) — which is why group A scores 97% above, and why the
  section must not overclaim. Sensitivity to direction → 0 as the boundary leaves the
  neighbourhood; sensitivity to scale is 0 everywhere.
- Rank-based instruments (Spearman of probabilities, AUC, threshold-at-f(q)) restore
  direction sensitivity off the boundary but remain scale-blind. Proper scoring rules
  are sensitive to both. This pre-empts the "just use AUC" review.

## 4. Experiments

Each has: goal → registered prediction → method → acceptance → what to do if it fails.
Costs are wall-clock guesses. Everything runs with `parallel_eval=False`.

### E-M1 — instrument response of a synthetic surrogate family (illustration; ~2 h)

**Goal.** Show, with no fitting in the loop, what each instrument can and cannot see.
This is the "back it up" figure for §3 and the natural first panel of the motivation.

**Method.** 2-D Gaussian data, logistic black box (truth ∇logit f exact). At query
point q build the family

    logit g_{θ,s}(x) = logit f(q) + s · R(θ) ∇logit f(q) · (x − q),

with rotation θ ∈ [−180°, 180°] and scale s ∈ [0.1, 10] (log-spaced); cosine to the
truth is cos θ by construction, so the x axis *is* the explanation error. Wrap it in a
small class exposing `predict`, `predict_proba`, `get_explanation` (put it in
`common/`, **not** the explainer registry — a registry entry is auto-offered in the
notebook and auto-tested). Then every registry metric runs unchanged on it.

Instruments, in ladder order: `fidelity (local)` on the local sample; `fidelity (local)`
on test data; `fidelity (local query probs)`; `spearman`; `Brier score (local)`;
`KL divergence (local)`. Query points: three from the 20 on the between-means line with
f(q) ≈ 0.5, 0.9, 0.99 (read the indices off `get_points_between_class_means`).

Figure: rows = query points (boundary → confident), columns = instruments, each panel a
curve over θ at s = 1 and a curve over s at θ = 0 (or a small θ×s heatmap). Normalise each
instrument to its own range so the *shape* is what the reader compares.

**Prediction.** Fidelity: flat in s at every q (exact); a V in θ at f(q) ≈ 0.5, flattening
to a constant 1 by f(q) ≈ 0.99. Query-probs fidelity and Spearman: V in θ at every q,
flat in s. Brier and KL: minimum at (θ = 0, s = 1) at every q, sharpest near the
boundary.

**Acceptance.** The figure shows those shapes. Failure is only possible for the
near-boundary "V" (a sampling artefact would show a noisy floor): if so, increase
`samples` in `get_local_points` for the evaluation set, and say so in the caption.

**Second panel (recommended).** Repeat on Breast Cancer|Logistic (30-D) rotating in a
random plane that contains the truth vector. Same shapes expected; it shows the effect is
not a 2-D toy. Keep only if it adds nothing surprising; if it does, that is a result.

### E-M2 — thresholded fidelity as a proxy for a correct explanation (analysis; ~3 h + optional 1–2 h sweep)

**Goal.** Put the §2 probe numbers on the same footing as `sec:fidelitypredicts`
(which does level / direction / magnitude for KL and Brier), adding the two fidelity
cells, tie rates, the all-perfect-score points, and the best-by-instrument table.

**Method.** New `analysis/analyse_fidelity_explanation.py` joining
`results_fidelity.json` and `results_gradient_truth.json` per (config, point). Reuse
`analyse_gradient_truth.load` for the degenerate-point handling and clip KL at 1e-16
before logs (the probe hit log(0)). Report:

1. level ρ, direction (config level, as the paper does — mean over 20 points — *and*
   point level), magnitude ρ, for all four instruments; per group.
2. tie rate per instrument; fraction of points where every surrogate scores 1.000;
   cosine spread among surrogates at those points.
3. best surrogate by instrument (counts) alongside each surrogate's mean cosine and
   top-1. This is the hard-label story: the instrument's favourite has the worst
   explanations.
4. compression: the existing "10³× Brier is 2.4 pp fidelity" number, restated per
   point rather than per config if it reads better.

Write `tables/fidelity-proxy.tex` (extend `table_fidelity.py` or add
`table_fidelity_proxy.py`; register in `make_paper.sh`).

**Registration.** The 84-config part has been *seen* (§2). Say so in
`PREREGISTRATION.md` — it is a recomputation, not a blind test. The blind part is the
extension: run `sweep_fidelity.py` for the five extended-grid differentiable models
(Bagged Logistic, Bayes Optimal, Nearest Class Mean, Polynomial Logistic (deg 2), RBF
Logistic (Nystroem); 14 × 5 = 70 configs; add `--models` filtering as
`sweep_gradient_truth.py` has, or a separate output file merged at analysis time so
`results_fidelity.json` stays the registered 168). Register before running:
tie rate ≥ 30%; hard-label best-by-test-fidelity in ≥ 40% of the 70 and its mean cosine
lowest of the three; fidelity's level ρ within ±0.1 of KL's in magnitude.

**Acceptance.** Whatever comes out is reported. The clause "fidelity is uninformative"
is already dead; if the extension also kills "ties ≥ 30%" or "hard-label is the
favourite", those clauses go too and the section rests on E-M1 + E-M3 + compression.

**Optional hypothesis to check, not to build on.** Each surrogate wins under the
instrument closest to its own training loss (hard-label ↔ fidelity holds: 44/84;
standard LIME ↔ Brier does *not*: 12 vs Logit-LIME's 50). One sentence at most if it
survives.

### E-M3 — the explainer that explains nothing (base rate; ~1 h)

**Goal.** A number a reader remembers: the fidelity a zero-coefficient surrogate gets.

**Method.** Constant surrogate g ≡ kernel-weighted mean of f over the training
neighbourhood (the Brier-optimal constant; KL stays finite), explanation = zero vector.
Same wrapper class as E-M1. For every registered config and query point compute its
fidelity on the local sample and on test data, its Brier and KL, and compare with the
three fitted surrogates. No new black boxes: `run_pipeline` caches them; only the
scoring loop is new, so this is minutes of compute. Consider adding it as a fourth
column of E-M2's table rather than a separate table.

**Prediction.** Mean local-sample fidelity of the null explainer ≥ 0.9; it *beats or
ties* standard LIME on local-sample fidelity at ≥ 30% of points; under KL it is worst at
> 95% of points.

**Acceptance.** As predicted → one sentence + a table column. If it scores badly under
fidelity (< 0.8), the off-boundary clause of §3 is weaker than assumed on this grid and
the claim must be scoped to confident query points — split by f(q).

### E-M4 — the worked example on explanations (illustration; ~4 h including selection)

**Goal.** One figure a reader can look at and see: same fidelity reading, different
explanation; or better fidelity reading, worse explanation — with the proper scores
alongside catching what the threshold missed.

**Selection protocol (fix before looking further).**

1. Candidate pool = E-M2's join. Two shapes:
   - **case 1**: fidelity (both cells) says standard ≥ logit; cosine says logit ≫
     standard (Δcos > 0.4).
   - **case 2**: hard-label surrogate has the best test-data fidelity by ≥ 0.02 and the
     worst cosine by ≥ 0.4, and KL ranks it last.
2. Exclude points where ‖∇logit f(q)‖ < ½ × the config's median (nothing to explain).
3. Exclude points where f is non-monotone along the transect within the kernel's support
   (§2 caution 2). Plot it; do not infer it.
4. Prefer a 2-D dataset (Gaussian, Moons) so the black box's surface, the neighbourhood,
   the three boundaries and the four gradient arrows (truth + 3 surrogates) can all be
   drawn in one panel. Fall back to the 1-D transect + bar chart of
   `fig_justification.py` for a tabular dataset.
5. Pick the *median* qualifying point, not the most extreme, and state the base rate of
   the shape in the caption (E-M2 counts: how many of 1,680 points / 84 configs).
6. Robustness of the chosen point: rerun that one config under the five seeds of
   `sweep_seeds.py` (its `results_seed*.json` only carry Brier/KL, so this is a small
   new run recording fidelity + cosine) and under 3 kernel widths from
   `sweep_kernel.py`'s range. The shape must hold in ≥ 4/5 seeds. If it does not, the
   point is noise; go to the next candidate. Log every candidate tried.

Candidates from the probe, pending steps 2–3 and 6: Gaussian|SVM pt 5 and Moons|SVM pt 0
(case 2, drawable), Wheat Seeds|SVM pt 19 (case 2, tabular), Moons|SVM pt 1 and
Breast Cancer|MLP pt 12 (case 1). Breast Cancer|Logistic pt 10 (already in the paper) is
the *compression* example, not a blindness example: fidelity 0.979 vs 1.000 for cosine
0.71 vs 1.00 and a 26,000× KL gap; reuse it in the caption rather than redraw.

**Figure.** Two query points of one config (one at the boundary, one confident), each
with the picture and a readout strip: fidelity (test) / fidelity (local) / Brier / KL /
cosine, for standard, Logit-LIME, hard-label and the null explainer of E-M3. The strip is
the argument; the picture is why.

**Acceptance.** A point that passes steps 2, 3 and 6 exists. If none does, the
"instrument failure" illustration is replaced by the compression example plus E-M1, and
the section says the blind cases on this grid are mostly non-monotone or flat black
boxes — which is itself worth a sentence.

### E-M5 — the hard-label surrogate under fidelity (analysis only; folded into E-M2 item 3)

Already implied by `sec:hardlabel` (best Brier, poor KL). The new fact is that it is also
the *fidelity* favourite while having the worst explanations. One table column and one
sentence; cross-reference `sec:hardlabel` for the overconfidence mechanism.

## 5. Robustness checks to run before writing numbers

- **Local vs test fidelity.** Both cells everywhere. The local cell is the blinder one
  (already in the paper: "a particularly blind combination"); CIKM'23's protocol is the
  test-data cell, so that is the one the motivation quotes first.
- **Saturation.** Report the E-M2 numbers with and without configs flagged saturated /
  `truth_is_constant` in `results_gradient_truth.json`.
- **Seeds and kernel width** for the worked example only (E-M4 step 6); the grid-level
  numbers are counts over 1,680 points and do not need reseeding.
- **Alignment check** at the top of the analysis script: assert Brier|local from the two
  result files agrees per point (probe: max difference 0.0). If a re-run ever breaks
  that, the join is invalid.

## 6. Writing plan (keep it lean — the write-up is a working doc)

- `sections/02-background.tex`, `sec:metrics`: replace the two-sentence dismissal with
  §3 above in ≤ 6 lines, one figure reference (E-M1 + E-M4 as one two-row figure, or two
  figures if the 2-D panel needs the space), and forward references to `sec:fidelity`
  and `sec:fidelitypredicts`. Do not restate numbers that live in captions.
- `sections/05-results.tex`, `sec:fidelity`: add the E-M2/E-M3 table and ~10 lines:
  ties, all-perfect points, hard-label favourite, null explainer, per-group direction
  including the honest "KL is a coin flip on group B". Rename the subsection if it no
  longer only asks "would the usual evaluation have seen this?".
- `sections/01-intro.tex` `tab:findings`: one row, status *illustration* (E-M1, E-M4)
  and one row *observation* (E-M2/E-M3 counts).
- `sections/04-setup.tex` `tab:experiments`: rows for the new sweep/analysis scripts.
- `sections/06-discussion.tex`: one limitation bullet — proper scores rank surrogates
  but are not a truth; where a gradient exists, score the explanation.
- Repo: `README.md` sweep table, `make_paper.sh`, `rerun_all.sh` (only if a sweep is
  added), `PREREGISTRATION.md` (third registration: E-M2 extension + E-M3 predictions,
  with the note that the 84-config numbers were seen first), `FINDINGS.md` §4 (a
  subsection under "Does fidelity predict a correct explanation?"), and this file's
  status.

## 7. Traps and things easy to miss

- **Do not claim fidelity is uncorrelated with explanation quality.** It is not (§2).
- **Ties are the headline, magnitude blindness is the theory, overconfidence is the
  mechanism.** Keep those three separate in the prose.
- **Non-monotone black boxes** (§2 caution 2): the gradient at q and the nearest
  boundary can disagree for RBF SVMs and MLPs. The paper's ground truth is the gradient
  by definition; an illustration must not pick a point where that definition is doing
  the work.
- **The B12 fallback** (one-class neighbourhood → zero coefficients) makes cosine NaN,
  not 0. Count NaNs; do not `nan_to_num` them into the spread statistic silently.
- **`fidelity (local query probs)`** exists (B13 fixed its zero-weight bug). Use it as
  the "threshold at f(q)" rung of the ladder; it is the cheapest fix a reviewer will
  propose and the ladder shows it is still scale-blind.
- **KL's ε = 1e-7 floor.** A surrogate that outputs exactly 0/1 gets ~16 nats per point
  where f disagrees. Fine for ranking; do not quote such a KL as a ratio.
- **New draw sites need their own salt** (`rng_from_point`). E-M1/E-M3/E-M4 should not
  need new sampling — reuse `get_local_points` — but if one does, do not reuse
  `'local evaluation sample'`.
- **`run_pipeline` is cached in-process; `cache_clear()` after changing module
  constants** (kernel width in E-M4 step 6).
- **Registry side effects.** A new explainer class in the registry is auto-tested
  (`test_explainer_builds_far_from_the_boundary`) and appears in the notebook. Keep the
  synthetic-family and null explainers in `experiments/logit_lime/common/`.
- **Colours** from `common/style.py` only; the four-surrogate readout needs a fourth
  colour — take the next validated palette slot, not an eyeballed grey.
- **Binary only.** Everything here is two-class; say so once.
- **Do not overwrite `results_fidelity.json`** with the extended models; keep the
  registered 168 intact and merge at analysis time.

## 8. Order and budget

1. E-M1 (synthetic family, 2 h). Cannot fail; gives the theory figure immediately.
2. E-M2 analysis on the existing 84 configs (3 h). Write the table script as you go.
3. E-M3 null explainer (1 h) — one more column in the E-M2 table.
4. Register E-M2's extension + E-M3's predictions, then run the 70-config fidelity
   extension (1–2 h wall clock, unattended).
5. E-M4 worked example, with the selection log (4 h).
6. Write-up and regenerate (`make_paper.sh`), FINDINGS/README/PREREGISTRATION (2 h).

About two working days. Steps 1–3 need no new sweeps and can be done in one session.
