# Plan: a motivation section — what a probability-space coefficient claims

Written 2026-09-18 as a handover.

## STATUS: executed 2026-09-18. This file is kept as the record of what was
## planned; where the execution departed from it, the departure is the finding.

Everything below was carried out. What changed on contact with the data:

- **The registered predictions were about averages when they should have been about a
  gradient.** P1b and P3a both failed, and for the same reason: the slab's half-width is
  $1/\lVert\beta\rVert$, so it is *wide* wherever the black box never commits. Median mass
  by the black box's confidence is 0.007 / 0.051 / 0.090 / **0.235** across
  $|\logit f(q)|$ bands (Spearman +0.57). The defect concentrates exactly where a confident
  prediction is the reason an explanation was wanted. That is a better claim than the one
  registered, and the section makes it.
- **P3a also turned on two choices the registration itself made.** Eight of the 28 group A
  configurations are a linear model on data it cannot separate, where $f$ spans less than
  0.5 over the whole query line. And excluding saturated points — done to protect
  Logit-LIME from its squash bound — removes the confident points where the standard
  coefficient collapses (all-points medians 10.3× vs 1.01×, against 2.7× vs 1.01×
  unsaturated). Both are reported; neither is offered as a restatement that passes.
- **P4b missed by five points** (85% against a registered 90%), on the same saturation
  boundary. The contrast it tested is intact.
- **Panel (c)'s saturation band is empty on the illustration configuration**, so the figure
  carries none. The logit coefficient's drift there is 7%, below the 10% shading threshold.
- **The worked example's flip row needed a monotonicity guard after all**, exactly as §2
  warned: the flip is computed only where the black box crosses its own boundary once
  along the feature, and the count of exclusions is reported (101 of 560 saturated, 0 for
  the crossing count on group A).
- **Panel (a) was redrawn** from a hatched overlay to a white slab on grey, so that grey
  means "not a probability" in both (a) and (b). The hatching swamped the panel.

Delivered: `sweeps/sweep_range.py` (with `--scales` for the kernel robustness),
`analysis/analyse_range.py`, `analysis/table_reading.py`, `figures/fig_reading.py`;
`sections/03-motivation.tex` added and `03-problem`…`07-conclusion` renumbered to
`04`…`08`; the duplicated "What a logit coefficient means" paragraph moved out of
`sec:logitlime`; four rows in `tab:findings`, one in `tab:experiments`, one recommendation
sentence and one limitation bullet in the discussion; fourth registration and its outcome
in `PREREGISTRATION.md`; `FINDINGS.md` §4; `README.md`, `make_paper.sh`, `rerun_all.sh`.

---

The original plan follows unchanged.

## 0. The brief, in one paragraph

Add a section to the Logit-LIME write-up that motivates *not* fitting the surrogate in
probability space, from the point of view of the number a user is handed. Standard LIME's
coefficient is "probability per unit feature". The section should show, with one simple
illustration and one table, what happens when a reader carries that reading across the
neighbourhood the surrogate was fitted on: the claimed probability leaves $[0,1]$ within a
fraction of the kernel width, the same coefficient asserts different things at different
starting points, the coefficient vanishes exactly where the user is most likely to ask
(a confident prediction), and the counterfactual it implies ("how far to flip the
decision") is wrong by a growing factor. A log-odds coefficient has none of these defects
*as a reading*, whatever the quality of the fit. The fit quality is the rest of the paper;
this section is about the meaning of the reported number and must say so explicitly.

The write-up is a lean working doc (memory: `logit-lime-writeup-purpose`). Target: about
one page of prose, one figure, one table, one short grid-level backing table or paragraph.

## 1. What the section must claim, and what it must not

**Claims (each backed by a panel, a table row or a sweep number):**

1. **The reading expires inside its own neighbourhood.** For a linear surrogate of $p$,
   the set where $g \in [0,1]$ is a slab of width $1/\lVert\beta\rVert$ around the
   boundary. Carrying the coefficient a distance $r_{\text{exit}} = (1-g(q))/\lVert\beta\rVert$
   (or $g(q)/\lVert\beta\rVert$ on the other side) from $q$ produces a number that is not a
   probability. On the Gaussian data that distance is *shorter than the kernel width*
   ($k = 0.75\sqrt{d} = 1.06$) at every query point with $f(q) \ge 0.58$, and the
   kernel-weighted training mass on which the unclipped surrogate is outside $[0,1]$ is
   15–35% (probe, §2). Clipping does not rescue the reading: on that mass the clipped
   surrogate has zero derivative, contradicting its own coefficient.
2. **The reading has no fixed meaning.** The standard coefficient is neither the derivative
   of $p$ at $q$ (the chord over a neighbourhood as wide as the sigmoid is less than half
   the tangent: $0.43$ vs $0.96$ at the boundary point) nor a finite difference over any
   particular step. It also starts from the wrong place: $g(q) \neq f(q)$ by up to $0.17$
   on the Gaussian data ($0.68$ vs $0.85$ at point 11), so "$p$ rises by $\beta$" has no
   agreed base value. A log-odds coefficient is one thing at every point: the odds
   multiply by $e^{\beta_j}$ per unit of $x_j$, from any base, and for a linear-logit black
   box it *is* the true slope ($3.941$ vs $3.942$ at all 14 non-saturated points).
3. **The coefficient conflates importance with saturation.** Along the query line the
   standard $\lVert\beta\rVert$ falls from $0.43$ at the boundary to $0.001$ at the ends
   while the black box's log-odds slope is constant. A user shown the number cannot tell
   "this feature does not matter" from "the prediction is saturated". In logit space the
   slope carries the first and the intercept the second. This is the point a reviewer will
   push on ("but that *is* the derivative of $p$") — see §7.
4. **The counterfactual it implies is wrong by a growing factor.** Distance along the top
   feature to $g = \tfrac12$: standard LIME $-2.60$ vs true $-1.54$ at $f(q) = 0.989$, $-5.0$
   vs $-2.0$ at $0.997$, $-10.6$ vs $-2.5$ at $0.999$; Logit-LIME matches the truth to two
   decimals on the logistic black box and to within ${\sim}7\%$ on the MLP. (Do not extend
   this to the SVM; §7.)

**Must not claim:**

- that leaving $[0,1]$ is *why* Logit-LIME wins on fidelity. Figure 1's SVM column shows
  the standard surrogate leaving $[0,1]$ just as badly with a $1.4\times$ benefit. The
  existing sentence in `sec:logitlime` — "an argument for logit space independent of every
  fidelity measurement" — is the right framing; keep it, and cross-reference
  `sec:problem`.
- that the log-odds *fit* is right. The reading is well posed everywhere; the fit is only
  as good as the log-odds are linear (`sec:groupresults`). Say this once.
- anything about multi-class. Binary throughout; say so once.

## 2. Probe results (2026-09-18, ad hoc; the sweep in §4 recomputes them)

Direct construction of both surrogates at the 20 registered query points, Gaussian data,
$k = 1.06$. `r+` is the distance along $\beta$ until the unclipped standard surrogate hits
$1$; `mass` is the kernel-weighted fraction of the surrogate's own 10,000-point training
sample where its unclipped output is outside $[0,1]$; `flip` is the signed step along
feature 1 (the true top feature) to reach $g = \tfrac12$, against the black box's actual
crossing on the same axis.

| pt | $f(q)$ | $g_{\text{std}}(q)$ | $\lVert\beta_{\text{std}}\rVert$ | $\lVert\beta_{\text{logit}}\rVert$ | $\lVert\nabla\logit f\rVert$ | $p(1{-}p)\lVert\nabla\rVert$ | r+ | r+/k | mass | flip std | flip logit | flip true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian\|Logistic | | | | | | | | | | | | |
| 10 | 0.579 | 0.534 | 0.430 | 3.941 | 3.942 | 0.961 | 1.08 | 1.02 | 0.155 | −0.10 | −0.11 | −0.11 |
| 11 | 0.847 | 0.680 | 0.389 | 3.941 | 3.942 | 0.510 | 0.82 | 0.78 | 0.167 | −0.62 | −0.58 | −0.58 |
| 12 | 0.957 | 0.805 | 0.296 | 3.941 | 3.942 | 0.161 | 0.66 | 0.62 | 0.205 | −1.39 | −1.06 | −1.06 |
| 13 | 0.989 | 0.890 | 0.202 | 3.941 | 3.942 | 0.043 | 0.55 | 0.51 | 0.253 | −2.60 | −1.53 | −1.54 |
| 14 | 0.997 | 0.946 | 0.119 | 3.939 | 3.942 | 0.011 | 0.46 | 0.43 | 0.286 | −5.03 | −2.01 | −2.01 |
| 15 | 0.999 | 0.976 | 0.060 | 3.936 | 3.942 | 0.003 | 0.41 | 0.39 | 0.315 | −10.6 | −2.48 | −2.49 |
| 19 | 1.000 | 1.000 | 0.001 | 3.684 | 3.942 | 0.000 | 0.28 | 0.27 | 0.354 | −634 | −4.64 | −4.38 |
| Gaussian\|MLP | | | | | | | | | | | | |
| 11 | 0.906 | 0.688 | 0.399 | 3.950 | 4.711 | 0.403 | 0.78 | 0.74 | 0.181 | −0.63 | −0.58 | −0.58 |
| 13 | 0.990 | 0.898 | 0.196 | 3.318 | 2.719 | 0.026 | 0.52 | 0.49 | 0.263 | −2.72 | −1.69 | −1.58 |
| 15 | 0.999 | 0.979 | 0.053 | 2.865 | 2.698 | 0.004 | 0.40 | 0.38 | 0.318 | −11.9 | −2.89 | −2.61 |
| Breast Cancer\|Logistic ($d=30$, $k=4.11$) | | | | | | | | | | | | |
| 13 | 0.950 | 0.662 | 0.215 | 3.722 | 3.770 | 0.178 | 1.57 | 0.38 | 0.205 | 2.73 | 2.45 | 2.44 |
| 15 | 0.998 | 0.809 | 0.212 | 3.612 | 3.770 | 0.007 | 0.90 | 0.22 | 0.235 | 4.63 | 5.26 | 5.16 |
| 17 | 1.000 | 0.908 | 0.159 | 3.412 | 3.770 | 0.000 | 0.58 | 0.14 | 0.284 | 17.4 | 8.63 | 7.88 |

Points 0–9 mirror 10–19 on the class-0 side (mass below $0$ instead of above $1$).

Things the probe taught that shape the plan:

- **Point 11 is the illustration point.** $f(q) = 0.85$, the same `QUERY_INDEX = 11` as
  `fig_mechanism.py`, so the new figure and Figure 1 describe the same neighbourhood.
  Walking $+1, +2, +3$ sd along feature 1 from $q$, standard LIME claims
  $0.68 \to 0.96 \to 1.23 \to 1.51$; the black box (and Logit-LIME, which recovers it) gives
  $0.85 \to 0.989 \to 0.9993 \to 0.99995$. The standard reading crosses $1$ between the
  first and second step, i.e. inside the kernel width.
- **The flip story is clearest at points 13–15**, not 11 (where all three agree). The
  table, not the figure, carries the flip; the figure shows the walk.
- **Logit-LIME's coefficient drifts at the saturated ends** ($3.94 \to 3.68$ at point 19)
  because `logit_ridge` squashes $p$ into $[10^{-9}, 1-10^{-8}]$ with an affine map. Mark
  saturated points (flag in `results_gradient_truth.json`) and cross-reference
  `sec:saturation`; exclude them from any registered ratio.
- **The MLP's truth is not constant** ($\lVert\nabla\rVert$ 2.7–5.7), and Logit-LIME
  tracks it only loosely. The illustration is the logistic black box; the MLP is one
  sentence ("the same shapes; the flip is within 7%").
- **Moons|SVM is unusable for the flip.** The true crossing along an axis is at $0.01$ from
  points 8–10 while both surrogates say $0.3$–$8$; the RBF boundary is curved inside the
  neighbourhood. Same trap as `PLAN_fidelity_motivation.md` §2 caution 2. Restrict any
  flip analysis to group A, where the truth is analytic and monotone by construction.
- **In 30-D the exit radius is a smaller fraction of $k$** ($0.1$–$0.6$) because $k$ grows
  with $\sqrt{d}$ while the sigmoid's width does not. The claim "expires inside its own
  neighbourhood" is *stronger* on the tabular grid, but distances of 5+ sd along one
  feature are outside the data for everyone; say so when quoting Breast Cancer flips.

## 3. What already exists (reuse, do not rebuild)

| asset | what it gives us |
|---|---|
| `sections/03-problem.tex`, `sec:logitlime`, paragraphs "What a logit coefficient means" and "Why the change of units is not the effect we measure" | the prose argument already exists in two paragraphs. **Move** the first into the new section (do not duplicate); leave the second where it is and cross-reference |
| `figures/fig_mechanism.py` | query point 11, transect construction, feature-space wash, "outside $[0,1]$" shading, unclipped standard output via `surrogate_model.predict`. Template for panels (a) and (b) |
| `figures/fig_justification.py` | the kernel-under-the-curve device (`fill_between` of the locality weight) and the three-panel layout at `figsize=(7.0, 2.6)` |
| `common/gradients.py::grad_logit` | the truth $\nabla\logit f(q)$ for panel (c) and the table |
| `common/surrogates.py::null_explainer` | shows how to regenerate the surrogate's own training neighbourhood with the `'surrogate training sample'` salt — deliberately the same draw site, not a new one |
| `sweeps/sweep.py::opts, DATASETS, MODEL_GROUPS` | the registered grid and the `opts` dict for `run_pipeline` (black boxes come back cached) |
| `results/results_gradient_truth.json` | per-point saturation flags and $\lVert\nabla\logit f\rVert$ for the 154 differentiable configs |
| `sections/05-results.tex` `sec:kernel`, `sec:saturation`, `sec:spatial` | kernel-width dependence of the chord, saturation, and "the gain is near the boundary" — cross-reference, do not restate |
| `common/style.py` | palette. Hatching for the "not a probability" region; no new colours needed (three series: black box, standard, logit; the tangent slope in panel (c) is `MUTED`) |

## 4. Work items

### W1 — the illustration figure (`figures/fig_reading.py` → `figs/fig_reading.pdf`; ~3 h)

Gaussian data, logistic black box, $q$ = query point 11 (assert $f(q) \approx 0.85$ at
runtime rather than trusting the index). One row, three panels, `figsize=(7.0, 2.6)`.

**(a) Where the surrogate is a probability.** Feature space in raw $x_1, x_2$ (the data
are symmetric, no need to rotate). Black box wash as in Figure 1; $q$ in `AQUA`; the
locality kernel as two iso-weight circles ($w = 0.5$ and $0.1$, radii
$k\sqrt{-2\ln w}$ — derive from Eq. `eq:kernel`, do not eyeball). The standard surrogate's
$g = 0$ and $g = 1$ lines (parallel to its $g = \tfrac12$ boundary, $1/\lVert\beta\rVert$
apart) with the two outer regions hatched and labelled "surrogate: $p > 1$" / "$p < 0$",
each annotated with its kernel-weighted training mass ($0.150$ and $0.016$ at point 11).
Arrow from $q$ along $\beta_{\text{std}}$ of length $r_{\text{exit}} = 0.82$, labelled as
the distance the coefficient can be carried. Logit-LIME's $p = 0.99$ and $0.01$ isolines
dashed in `ORANGE` for contrast: there is no edge to fall off.

**(b) Carrying the coefficient.** Horizontal axis: step along feature 1 from $q$, in sd,
from $-3$ to $+3$. Curves: black box (`INK`, thick), standard surrogate unclipped
(`BLUE`, dotted outside $[0,1]$, solid inside, plus the clipped flat segment), Logit-LIME
(`ORANGE`). Grey bands outside $[0,1]$ as in Figure 1. Markers at integer steps with the
claimed value printed for the standard surrogate ($0.68, 0.96, 1.23, 1.51$) and, smaller,
for the black box. Annotate "starts at $0.68$, not $0.85$" at step 0. Kernel weight along
the axis drawn under the curves as in `fig_justification.py` so the reader sees that the
$>1$ region carries weight. Mark the three $g = \tfrac12$ crossings on the left
($-0.62$ / $-0.58$ / $-0.58$); at this point they agree, which is honest and sets up the
table where they do not.

**(c) The number reported, as the query point moves.** Horizontal axis: the 20 query
points along the class-means line (position in sd from the boundary crossing, with $f(q)$
as a secondary tick row at $0.5, 0.9, 0.99$). Vertical: $\lVert\beta\rVert$ for standard
(`BLUE`), Logit-LIME (`ORANGE`), the truth $\lVert\nabla\logit f\rVert$ (`INK`, dashed,
constant $3.94$) and the tangent slope $p(1-p)\lVert\nabla\logit f\rVert$ (`MUTED`, thin).
The two units cannot share an axis: use a twin axis (probability per sd on the left in
`BLUE`, log-odds per sd on the right in `ORANGE`), or scale each series to unit maximum
and say so in the caption. The shape is the message: standard falls to zero at the ends;
Logit-LIME is flat on the truth and only bends in the saturated band (shade it, citing
`sec:saturation`); the standard chord is below the tangent even at the boundary.

If (c) crowds the figure, split it off as `fig_reading_scale.pdf`; do not drop it — it is
the panel that makes claim 3.

Print the numbers the caption quotes (masses, $r_{\text{exit}}$, the walk values, the
$g(q)$ vs $f(q)$ gap) at the end of the script, as the other figure scripts do.

### W2 — the table (`analysis/table_reading.py` → `tables/reading.tex`; ~1 h)

Gaussian|Logistic, feature 1, four query points (10, 11, 13, 15: $f(q) = 0.58, 0.85, 0.99,
0.999$). Rows:

- $f(q)$; $g(q)$ standard / Logit-LIME
- coefficient on feature 1: standard (probability per sd), Logit-LIME (log-odds per sd),
  truth (log-odds per sd); Logit-LIME's odds ratio $e^{\beta_1}$
- claimed $p$ after $+1$ sd and $+2$ sd: standard / Logit-LIME / black box
- $r_{\text{exit}}$ in sd and as a fraction of $k$; kernel mass outside $[0,1]$
- distance to the flip along feature 1: standard / Logit-LIME / black box

Take `--dataset`, `--model`, `--points` arguments as `table_example_explanation.py` does so
the MLP version can be printed for the one-sentence comparison. Register in
`make_paper.sh`. Caption says: standardised units, one unit = one sd of the feature on the
training split; steps of one sd are not small.

### W3 — grid-level backing (`sweeps/sweep_range.py` → `results/results_range.json`, `analysis/analyse_range.py`; ~3 h + ~1 h wall clock)

The illustration is one configuration. The section needs three numbers from the
registered grid so it is not a toy. For every registered config (168) and its 20 query
points, build the standard and Logit-LIME surrogates directly (black boxes via
`run_pipeline` with `opts()`, so they are cached; `parallel_eval=False`) and record:

- $f(q)$, $g_{\text{std}}(q)$ (unclipped), $g_{\text{logit}}(q)$
- $\lVert\beta_{\text{std}}\rVert$, $\lVert\beta_{\text{logit}}\rVert$
- $r_{\text{exit}}$ toward the confident side ($=(1-g(q))/\lVert\beta\rVert$ if
  $g(q) \ge \tfrac12$, else $g(q)/\lVert\beta\rVert$), and $r_{\text{exit}}/k$
- kernel-weighted mass of the training neighbourhood with unclipped $g > 1$, and $< 0$
  (regenerate the neighbourhood with the `'surrogate training sample'` salt, exactly as
  `null_explainer` does)
- for group A only (Logistic, LDA; 28 configs): the flip distance along the true top
  feature for standard, Logit-LIME and the analytic truth
  $-\logit f(q) / \partial_j \logit f(q)$
- the saturation flag joined from `results_gradient_truth.json` where it exists

Resume from the output file like the other sweeps. Cost: $168 \times 20 \times 2$ ridge
fits on 10k samples; under an hour, dominated by the forest and kNN black boxes.

`analyse_range.py` prints, overall and per group, with and without saturated points:

1. median and quartiles of the outside-$[0,1]$ mass; fraction of points with mass $< 0.02$
2. fraction of points with $r_{\text{exit}} < k$
3. median $|g_{\text{std}}(q) - f(q)|$ vs $|g_{\text{logit}}(q) - f(q)|$
4. per config, the span $\max\lVert\beta\rVert / \min\lVert\beta\rVert$ over the 20
   points, for each surrogate (non-saturated points only)
5. group A: flip error ratio $|{\text{flip}_{\text{surrogate}}} / {\text{flip}_{\text{true}}}|$ vs $|\logit f(q)|$
   (Spearman), and the fraction of points within 10% of the truth, per surrogate

Write `tables/range.tex`: one row per group, columns = items 1, 2, 4 (both surrogates);
group A gets item 5 in a second small block or the caption.

**Register before running** (fourth registration in `PREREGISTRATION.md`; the Gaussian
numbers in §2 have been seen, say so):

- P1. Median outside-$[0,1]$ mass over all 3,360 points $\ge 0.10$; mass $< 0.02$ at fewer
  than 15% of points.
- P2. $r_{\text{exit}} < k$ at $\ge 2/3$ of points.
- P3. Group A, non-saturated points: the standard surrogate's $\lVert\beta\rVert$ span is
  $\ge 5\times$ in $\ge 90\%$ of configs; Logit-LIME's is $\le 1.5\times$ in $\ge 90\%$.
- P4. Group A, non-saturated points: standard LIME's flip error ratio grows with
  $|\logit f(q)|$ (pooled Spearman $\ge 0.5$); Logit-LIME's is within 10% of the truth at
  $\ge 90\%$ of points.

Whatever comes out is reported. If P3 or P4 fail on LDA (whose log-odds are exactly linear
but whose fitted slope may be large, pushing more points into saturation), report Logistic
and LDA separately rather than dropping the claim.

**Kernel-width robustness.** Items 1–2 depend on $k$ by construction (narrower kernel →
tangent-like chord → farther exit). Rerun the range statistics for Gaussian|Logistic and
Breast Cancer|Logistic at the two extremes of `sweep_kernel.py`'s range by setting
`costs.KERNEL_WIDTH_SCALE` (the explainers are built directly, so the module global is read
at fit time; call `run_pipeline.cache_clear()` anyway if anything goes through the
pipeline). One sentence in the caption: "the mass is $x$–$y$ over a $20\times$ range of
$k$". Expect it to stay well above zero, because the sigmoid width is fixed by the black
box and the kernel must be narrower than that before the chord becomes a tangent.

### W4 — write-up (~2 h)

- **New file `sections/03-motivation.tex`**, `\section{Why not probability space? What a
  coefficient claims}`, label `sec:reading`, input in `main.tex` between background and
  the mismatch section; rename `03-problem` … `07-conclusion` to `04` … `08` (`git mv`;
  `main.tex` is the only list; `make_paper.sh` touches `figs/` and `tables/` only). The
  memory file `logit-lime-writeup-purpose` names `sections/01-intro.tex` and
  `04-setup.tex` — update it after the rename.
- Structure of the section, in this order and no longer than this:
  1. Two lines stating the two candidate readings (a linear model of $p$; a linear model
     of $\logit p$ inverted through a sigmoid). No method definition here — that stays in
     `sec:logitlime`, which the section forward-references.
  2. The walk (Figure, panel b) and what it claims: claim 1.
  3. Claim 2 in three sentences (chord not tangent; wrong base; one meaning in logit
     space). Move the "What a logit coefficient means" paragraph here, trimmed; replace it
     in `sec:logitlime` with one sentence and `\ref{sec:reading}`.
  4. Claim 3 with panel (c). Pre-empt the "that is the derivative" objection in one
     sentence (§7).
  5. Claim 4 with the table's flip row.
  6. One paragraph: the grid (W3 numbers, per group); the kernel-width sentence; binary
     only; "this is an argument about the reported number, independent of the fidelity
     results, and the SVM column of Figure~\ref{fig:mechanism} is the reminder that
     repairing the range is not where the fidelity benefit comes from".
- `sections/01-intro.tex` `tab:findings`: a new block *Interpretation: what a coefficient
  claims* with two rows — the illustration (status *illustration*) and the grid counts
  (status *registered (4th)*).
- `sections/04-setup.tex` `tab:experiments`: one row for `sweep_range.py`.
- `sections/06-discussion.tex`: one sentence in the practical recommendation — the
  log-odds reading is the one to report even where $\Delta$ says not to switch the *fit*,
  because a probability coefficient cannot be carried anywhere; and one limitation
  bullet — the interpretable-domain case (§7, item on binarised features) is argued, not
  measured.
- Repo: `README.md` sweep table, `make_paper.sh` (`fig_reading.py`, `table_reading.py`),
  `rerun_all.sh` (the new sweep), `PREREGISTRATION.md` (4th registration), `FINDINGS.md`
  §4 (a subsection "What a probability coefficient claims"), this file's status header.

## 5. Order and budget

1. W1 figure (3 h). Cannot fail; all numbers are in §2.
2. W2 table (1 h).
3. Register P1–P4, then run W3 (3 h + ~1 h unattended).
4. W4 write-up and regenerate with `make_paper.sh` (2 h).

About one working day. W1 and W2 need no sweep and can be done in one session.

## 6. Things easy to miss

- **Unclipped output.** `bLIMEy.predict_proba` clips to $[0,1]$; the unclipped standard
  surrogate is `surrogate_model.predict(X)[:, 1]` (Figure 1 already does this). For the
  logit surrogate `surrogate_model.predict` returns *sigmoided* probabilities
  (`logit_ridge.predict`); its log-odds are `intercept_ + X @ coef_`.
- **`get_explanation` returns the class-1 row** for every surrogate (B15). Coefficients
  are in standardised units; "one unit" means one sd of the feature on the training split.
- **The logit squash is affine, not a clip** (`logit_regression.py::logits`), so saturated
  probabilities become $\logit \approx \pm 18$–$20$ and the coefficient at fully saturated
  points drifts by ${\sim}7\%$. Flag, do not hide.
- **Regenerating the training neighbourhood** uses the `'surrogate training sample'` salt
  on purpose (same draw site as bLIMEy). Anything scored on a *new* sample needs its own
  salt; never `'local evaluation sample'`.
- **The flip along an axis** needs a monotone black box on that axis within the range
  searched. Group A is monotone by construction; MLP was fine in the probe; RBF SVM is
  not (§2). Search a bounded range ($\pm 8$ sd), take the crossing nearest $q$, and record
  `nan` when there is none — count the `nan`s.
- **Both classes.** "Beyond probability 1" on the class-1 side is "below 0" on the class-0
  side; the sweep records both masses and the text says so once.
- **`run_pipeline` is cached in-process.** The black box comes back cached; the explainers
  are rebuilt each time. Kernel-width reruns change `costs.KERNEL_WIDTH_SCALE` at module
  level; restore it afterwards in the same process.
- **Nothing goes in the explainer registry** — a registry entry is auto-offered in the
  notebook and auto-tested. The figure builds surrogates directly.
- **Colours from `common/style.py` only.** Hatching (`hatch='///'`, edge `MUTED`, no
  face) for the non-probability regions; it survives greyscale printing, which a fourth
  colour would not.
- **Do not overwrite** any existing `results/*.json`; `results_range.json` is new.
- **Doc is lean.** No restated numbers between prose and captions; the caption carries the
  point-11 numbers, the table the flips, the text the grid counts.

## 7. The objections to pre-empt (one sentence each in the section)

- *"The probability coefficient is the derivative of $p$, and $p$ is what the user cares
  about."* It is not the derivative (chord, §1 claim 2), and even the derivative
  $p(1-p)\nabla\logit p$ mixes a property of the feature (the slope) with a property of
  the location (the saturation); the user is handed the product and cannot recover either
  factor. Logit space reports them separately.
- *"Just clip."* Clipping makes the surrogate's derivative zero on 15–35% of the
  neighbourhood's mass, so the surrogate's prediction and its explanation disagree on its
  own training sample.
- *"The neighbourhood is small, the reading is local."* The kernel width is
  $0.75\sqrt{d}$ in standardised units — $1.06$ in 2-D, $4.1$ in 30-D — and the reading
  expires within a fraction of it at two thirds of query points (P2).
- *"Logit-LIME's coefficient is also wrong when the log-odds are not linear."* Yes: the
  reading is well posed everywhere, the fit is not, and `sec:groupresults` measures the
  fit. Keep the two separate.
- *"Real LIME uses binarised interpretable features, where a coefficient is a whole-step
  effect, so the unit-step reading is not what users do."* That makes it worse: the
  additive attributions users sum ("$+0.3$ from this feature, $+0.4$ from that") leave
  $[0,1]$ directly, with no derivative to hide behind. The write-up works in feature space
  without an interpretable domain; note in the discussion that the argument transfers and
  sharpens there, and that it is argued, not measured (the patch-domain result in
  `sec:patches` is the nearest measurement).
- *Related work to check before writing* (15 min, `refs.bib` and a search): the SHAP
  documentation's "log-odds vs probability" discussion and Lundberg & Lee 2017 make the
  same additivity argument for attributions; any prior "LIME in logit space" proposal
  should be cited if one exists. `refs.bib` currently has no SHAP entry.

## Appendix — the probe script (ad hoc; `sweep_range.py` supersedes it)

```python
import sys, os, warnings
sys.path.insert(0, '/home/matt/projects/CLIME/experiments/logit_lime')
warnings.filterwarnings('ignore')
import numpy as np, clime
from sweeps.sweep import opts, METRICS
from common import gradients
from clime.data.utils import costs
from clime.evaluation.key_points import get_points_between_class_means

def logit(p):
    p = np.clip(p, 1e-12, 1-1e-12); return np.log(p/(1-p))

for DATASET, MODEL in [('Gaussian', 'Logistic'), ('Gaussian', 'MLP'),
                       ('Breast Cancer', 'Logistic'), ('Moons', 'SVM')]:
    r = clime.pipeline.run_pipeline(opts(DATASET, MODEL, 'bLIMEy (normal)', METRICS[0]),
                                    parallel_eval=False)
    clf, train, test = r['clf'], r['train_data'], r['test_data']
    qs, _ = get_points_between_class_means(test); qs = np.asarray(qs, float)
    d = qs.shape[1]; k = 0.75*np.sqrt(d)
    for i, q in enumerate(qs):
        E = clime.explainer.AVAILABLE_EXPLAINERS
        es = E['bLIMEy (normal)'](clf, query_point=q, train_data=train, test_data=test)
        el = E['bLIMEy (logit)'](clf, query_point=q, train_data=train, test_data=test)
        bs = np.asarray(es.get_explanation(), float); bl = np.asarray(el.get_explanation(), float)
        fq = clf.predict_proba(q[None])[0, 1]
        gq = es.surrogate_model.predict(q[None])[0, 1]          # unclipped
        grad = gradients.grad_logit(clf, MODEL, q)[0]
        nb = np.linalg.norm(bs)
        rp, rm = (1-gq)/nb, gq/nb
        rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
        X = rng.multivariate_normal(q, np.cov(test['X'].T), 10000)
        w = costs.weights_based_on_distance(q, X); g = es.surrogate_model.predict(X)[:, 1]
        m1, m0 = np.sum(w*(g > 1))/w.sum(), np.sum(w*(g < 0))/w.sum()
        j = int(np.argmax(np.abs(grad)))
        flip_s = (0.5-gq)/bs[j]
        gl_q = el.surrogate_model.predict(q[None])[0, 1]
        flip_l = -logit(gl_q)/bl[j]
        ts = np.linspace(-8, 8, 3201); L = np.repeat(q[None], len(ts), 0); L[:, j] += ts
        fl = clf.predict_proba(L)[:, 1]; s = np.sign(fl-0.5); cross = np.where(np.diff(s) != 0)[0]
        flip_t = ts[cross[np.argmin(np.abs(ts[cross]))]] if len(cross) else np.nan
        print(i, fq, gq, nb, np.linalg.norm(bl), np.linalg.norm(grad), rp, rp/k, m1, m0,
              flip_s, flip_l, flip_t)
```
