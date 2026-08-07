# CLIME — state of the research

Written 2026-08-07 by reconstructing the repo, the notebook outputs and the Overleaf
projects. Last code commit was **2024-06-06**; the work has been dormant since.

---

## 1. Where the project stands

Three threads, in order of maturity:

| Thread | Status | Overleaf | Code |
|---|---|---|---|
| **CIKM'23 — location-agnostic surrogates** | Published, DOI `10.1145/3583780.3615284` | `~/Repos/Overleaf/CIKM-2023-camera-ready` | complete, figures reproducible |
| **Logit-LIME** | Sketch (~1 page) + first real results, see §4 | `~/Repos/Overleaf/Logit-LIME` | implemented, working, operating regime now characterised |
| **aLIMEgn / aLIMEgnment** | Framing note (~1 page, 3 open questions) | `~/Repos/Overleaf/aLIMEgn` | not started |

`~/Repos/Overleaf/CLIME/` is **empty** — that clone has no content. The paper you're
thinking of is `CIKM-2023-camera-ready`. `pre-cut-CIKM-2023-camera-ready` is *not* a
longer version: diffing the two shows only formatting, ORCIDs, copyright block and
light copy-editing. Two substantive things did change:

- pre-cut said the surrogate used *"ridge regression with lasso regularisation to
  enforce sparsity"*; camera-ready correctly says just **ridge regression**. The code
  agrees with camera-ready (`BLIMEY.py` uses `sklearn.linear_model.Ridge`, with a
  commented-out `Lasso` alternative).
- pre-cut noted that the query-point line *"remains linear in the first two principal
  components since PCA is a linear transformation"* — cut for space, worth restoring if
  the work is ever extended, because it justifies the 2D visualisation.

There is no unpublished material sitting in the pre-cut draft. If you want a longer
paper you are writing new content, not restoring cut content.

---

## 2. The setup

One configurable pipeline, one `opts` dict, string-keyed registries for every stage
(see `CLAUDE.md` for the mechanics). The scientific object of study is the gap between
two distributions:

- **X_s** — the data the surrogate is *trained* on: Gaussian samples drawn around the
  query point `q`, labelled by the black box `f(X_s)`, weighted by an exponential
  distance kernel `w_x`.
- **X_test** — the data the surrogate is *evaluated* on: the real test set, which
  follows the distribution `f` was trained on.

Standard LIME implicitly assumes these agree locally. The project's claim is that they
systematically don't, and that where they disagree is predictable from where `q` sits on
the decision surface.

The pipeline makes the mismatch controllable from both ends:

- `evaluation points` chooses **where** explainers are built — `between_class_means` (a
  line crossing the boundary, used for all paper figures), `grid` (a 20×20 PCA grid,
  400 explainers, added Dec 2023), `class_means`, `data_limits`, `all_test_points`.
- `evaluation data` chooses **what they are scored against** — `test data` (the paper's
  setting, the mismatched one) or `sample locally` (Gaussian samples around `q`, i.e.
  scoring the surrogate on its own training distribution).

That second switch is the experimental knob that makes the mismatch visible. It was
added late and has barely been used.

---

## 3. Published findings (CIKM'23)

**Finding 1 — fidelity depends on query-point location, and the dependence is
systematic.** Walking `q` along the line between class means, local fidelity of standard
LIME is high near the decision boundary and **degrades as `q` moves away from the
boundary into low-density regions of `X_test`** — specifically where `f` predicts one
class but nearby high-density test data belongs to the other. On Breast Cancer the drop
is from ~0.97 to ~0.54 (`experiments/figs/sampling/combined-Breast Cancer.png`).

This is counterintuitive in a way worth keeping in the pitch: those are exactly the
high-confidence regions where a user would *most* trust the explanation.

**Finding 2 — the cause is a train/eval distribution mismatch, not a modelling failure.**
Away from the boundary, `f(X_s)` is heavily class-imbalanced (nearly everything is
predicted as `q`'s class), while `X_test` in the same locality is not. The surrogate
optimises the wrong thing.

**Finding 3 — class-balanced sample weights recover most of the loss.** Replacing `w_x`
with `w_xc = w_x · w_c` where `w_c ∝ 1/n_c` over `f(X_s)` mostly flattens the fidelity
curve. Crucially this needs **no access to `X_test`** — the imbalance in the black box's
own predictions is enough signal to re-align the distributions.

**Finding 4 — when the classes are genuinely far apart, balancing costs nothing.** For
well-separated Gaussians (`μ = ±3`) the opposite class falls outside the locality kernel
entirely, `X_s` and `X_test` already agree, `w_c ≈ 1`, and the two methods coincide.
So the fix is safe to apply by default. This is a stronger result than the paper leans
on and could carry more weight in a longer write-up.

**The magnitude of the drop tracks local test-set density across the boundary.** Breast
Cancer shows the asymmetry clearly: the denser side of the boundary suffers less.

Not-yet-written observations sitting in the repo:

- `pics/random forest on moons showing cost sensitive sampling improves local and normal
  fidelity.png` — the fix helps **global** fidelity too, not only local. Never reported.
- `pics/rf balanced, sampled cost helps but class cost hurts slightly.png` — on a
  *balanced* random forest, sample-based costs (`w_c` from `f(X_s)`) help but
  data-based costs (`w_c` from the black box's training data) **hurt slightly**. This is
  the distinction between `'bLIMEy (cost sensitive sampled)'` and `'bLIMEy (cost
  sensitive class)'` and it is a genuinely interesting negative result: it says the
  useful signal is *local* class imbalance, not *global* class imbalance. `notes.txt`
  flags this as a research idea ("show other cost sensitive trained models where you
  need to do other things for LIME") and it was never followed up.

---

## 4. Thread 2 — Logit-LIME

**The argument** (from `Logit-LIME/main.tex`): LIME regresses on the black box's output
*probabilities* with a linear model. A linear model is unbounded, so it can predict
p < 0 or p > 1. It is the wrong hypothesis class for a probability. Train a properly
probabilistic surrogate instead.

**What is implemented.** Two surrogates, both selectable in the pipeline:

- `clime/models/logit_regression.py::logit_ridge` — ridge regression **in logit space**.
  Squashes `p` into `[1e-9, 1-1e-8]`, takes `log(p/(1-p))`, fits ridge, inverts with a
  sigmoid at predict time. Registry keys: `'bLIMEy (logit)'`, `'bLIMEy (logit and
  sample weights)'`.
- `clime/models/logistic_regression.py::logistic_regression` — sklearn logistic
  regression fitted on **rounded** probabilities, i.e. on the black box's hard class
  labels. Registry keys: `'bLIMEy (logistic regression)'`, `'bLIMEy (logistic regression
  and sample weights)'`. This is the "use the classes as input instead of the
  probabilities" variant the tex file suggests.

Supporting evaluation metrics were added at the same time and are the right ones for
this thread: `'Brier score'`, `'Brier score (local)'`, `'log loss'`, `'log loss
(local)'`. Accuracy-style fidelity can't distinguish a well-calibrated surrogate from a
badly-calibrated one that happens to threshold the same way, so these matter.

**Where it got to.** The last commit, `a895667` "logitLIME working" (2024-06-05), fixed
the inverse transform — it had been applying softmax over the wrong axis and stacking
the columns backwards; it now uses a sigmoid with `[1-p, p]` ordering. So the June 2024
state is: *the method finally produces correct probabilities*. The very next thing was a
comparison run, and then work stopped.

**The last run** is still saved in `experiments.ipynb`: Gaussian data (`μ = ±1`,
identity covariance, 200 per class, standardised), random forest (92% test accuracy),
three explainers — normal / cost-sensitive-sampled / logit — 400 grid query points,
**Brier score (local)**, evaluated on **locally sampled data**.

It is inconclusive, and for a mundane reason: all three score ≈ 0.02 while every plot
axis is hard-coded to `[0, 1]`. The bar chart shows three identical stubs and the
heatmaps are three flat purple squares. **The experiment ran; the plotting hid the
result.** Fixing the axis scaling (see §6) and re-running that exact cached config is
the cheapest next step in the whole repo — the numbers may well already separate.

**Known gap (now fixed).** `bLIMEy.get_explanation()` used to raise `IndexError` for the
logit surrogate (§6, B4). Fidelity metrics don't touch it, so the pipeline ran clean, but
feature importances — the actual explanation — could not be extracted. Fixed and tested.

### When is Logit-LIME actually better? (answered 2026-08-07)

Full study: 4 datasets x 7 black boxes x 3 surrogates x 2 metrics, 20 query points each,
run serially. **Written up as a paper in `~/Repos/Overleaf/Logit-LIME/` (10 pages,
compiles clean).** Headline: the June 2024 comparison used a random forest, which is close
to the worst possible choice.

**The decisive check.** If the black box *is* a logistic regression its log-odds are
exactly linear in `x`, so a ridge fit in logit space should recover it while a line fitted
to the sigmoid cannot. It does — by **12 797x** on Gaussian, 287x on Banknote, 17x on
Breast Cancer. The implementation is sound.

**The benefit is a property of the black box, not the explainer.**

| black box family | Logit-LIME better | median Brier ratio |
|---|---|---|
| logistic regression, MLP | **8/8** | **13.7x** |
| SVM, gradient boosting | 8/8 | 1.28x |
| random forest (raw, Platt, isotonic) | 6/12 | 0.93x |

**The predictor.** Fit a locality-weighted linear model to the black box's log-odds and to
its probabilities on the locally sampled points, and take the difference in weighted R².
This "log-odds linearity gap" Δ predicts the benefit: **Spearman ρ = 0.77, p = 2e-06,
n = 28**. Every configuration with Δ > 0.35 is a logistic regression or MLP, and all six
show a benefit (3.6x to 1.3e4). Δ needs only the black box, so it can be computed *before*
choosing a surrogate.

Caveat worth keeping: a large Δ implies a large benefit, but not the converse. Pima
Indian Diabetes + logistic regression has Δ = 0.16 yet a huge measured ratio, because
Logit-LIME recovers that black box essentially exactly (3e-09) and the ratio's denominator
collapses. Δ measures how much *worse* probability space is, not how good logit space can
get.

**Saturation is NOT the mechanism** — the discriminating experiment. Platt-calibrating a
random forest takes saturation from **65.9% to 0.0%** while leaving Δ at ~0.04 and the
benefit at 1.5x (slightly *worse* than uncalibrated). Across the sweep saturation has no
relationship with benefit at all (**ρ = −0.13, p = 0.5**). Platt scaling composes a sigmoid
with the forest's unchanged piecewise-constant score, so `logit p` stays a step function
of `x`. This is the cleanest result in the study.

**Where the benefit lives.** Concentrated in the **transition region near the decision
boundary**, zero in the tails — the opposite of my initial guess. Saturated tails are easy
for both surrogates because the black box is locally flat there; the difficulty is where
the probability surface actually curves.

**Unexpected: the two proper scoring rules disagree, informatively.** The hard-label
variant (`bLIMEy (logistic regression)`, which thresholds the black box's probabilities)
attains the **best local Brier score in 16/28** configurations — more than either other
surrogate, median 24% better. But it is best on **KL in only 9/28**, with a median
divergence **4.2x** the better of the others and a worst case eight orders of magnitude
out. It is *well ranked and badly calibrated*: fitting to hard labels throws away the
black box's uncertainty, squared error forgives confident-and-right, KL punishes
confident-and-wrong without limit. Practical upshot: **report Brier and KL together — a
disagreement between their rankings is a cheap detector of an overconfident surrogate.**

**What this means for the paper.** Not "our surrogate is better" — that dies on random
forests. It is *"the right surrogate depends on the black box's local log-odds geometry,
and here is a cheap diagnostic that tells you which to use"*. That framing explains the
random forest result rather than being embarrassed by it, mirrors the CIKM contribution
("don't be location agnostic" → "don't be black-box agnostic"), and connects to aLIMEgn,
which is likewise about aligning to properties of the black box rather than the data.

---

## 5. Thread 3 — aLIMEgn

The most recent thinking (July 2024) and the most promising direction, but no code.

**The reframing.** The CIKM paper's contribution generalises. Sampling `X_g` should not
target `P(X, y)` — the distribution that trained the black box — it should target
`P(X, ŷ)`, the distribution of the black box's *own* predictions. Those coincide only
when `f` has learned the data well. They diverge under weak labels, under-training,
distribution shift, or plain underfitting, and in exactly those cases sampling from
`P(X, y)` (the standard advice, e.g. Kleinlein et al. BMVC'22) gives you a surrogate
aligned to the wrong target. The CIKM class-balancing trick is then reinterpreted as one
concrete way of using `ŷ` to align `X_g` with `P(X, ŷ)` without access to either
distribution.

That's a cleaner and more general story than the CIKM framing, and it subsumes it.

**Matt's own open questions**, verbatim from the tex, all still open:

1. Is `X_g` aligned to `P(X, y)` or just `P(X)`?
2. Should `g` be evaluated on test data drawn from `P(X, ŷ)` rather than `P(X, y)`?
3. `y` is a label in `{±1}` but `ŷ` is a probability in `[0, 1]` — does that asymmetry
   matter to the reasoning?

Question 2 is **already answerable with the existing code** and is the strongest
experimental hook in the repo. `evaluation data` toggles precisely that: `'test data'`
scores against `P(X, y)`, `'sample locally'` scores against a proxy for `P(X, ŷ)`. Nobody
has run the comparison systematically. The prediction the framing makes is sharp: for a
*well-fit* black box the two evaluations should agree; as the black box degrades they
should diverge, and the class-balanced explainer should track the `P(X, ŷ)` evaluation
while standard LIME tracks neither.

Question 3 also connects the two open threads: it is the same probability-vs-label
distinction that motivates Logit-LIME. **Logit-LIME and aLIMEgn are arguably one paper**
— "what should the surrogate be trained on, and against what should it be scored" —
rather than two.

---

## 6. Bugs and rough edges

All verified by running the code, not by reading it. Ordered by how much they'd cost you.

> **B1–B9 and B11–B14 were fixed on 2026-08-07** and each has a regression test. B10 was
> found while verifying those fixes and is **still open** — it needs a decision from you, because
> fixing it re-bases every number in the repo.
>
> The fixes were checked against a before/after snapshot of the paper's configurations
> run in serial mode. **All four Gaussian configurations are bit-identical**, and both
> Breast Cancer configurations produce the *same set* of query points. See "Effect on
> published results" at the end of this section.

### B1 — the package doesn't install. `setup.py`, `clime/data/*` — **FIXED**

Three files are named `__init__,py` — **comma instead of dot**:

```
clime/data/loaders/__init__,py
clime/data/utils/__init__,py
clime/data/tests/__init__,py
```

Editable installs survive by accident (PEP 420 namespace packages). `find_packages()`
does not, so `clime.data.loaders` and `clime.data.utils` are **absent from a built
wheel** — confirmed by building one. Anyone doing a real install gets
`ModuleNotFoundError` on `import clime`.

This means **the Colab badge in `README.md` is dead**, because cell 1 of the notebook
runs `pip install git+https://github.com/mattclifford1/CLIME` on Colab. It has been
broken since the loaders were split out (`7482111`, June 2023) — i.e. for the entire
period after the paper was submitted.

**Fixed** by renaming to `__init__.py`. `find_packages()` now reports
`clime.data.loaders`, `clime.data.utils` and `clime.data.tests`, and they appear in a
built wheel. The Colab badge is still commented out in the README until the fix is
pushed to GitHub, since Colab installs from the remote.

### B2 — `'Logistic balanced training'` trains anti-balanced. `clime/data/utils/costs.py` — **FIXED**

`get_instance_class_weights` builds `Y = concat(y, 1-y)`, putting the class-1 indicator
in column 0, then dots it with `class_weights = [w_class0, w_class1]`. The weights come
out **swapped**. Measured on 8:2 data:

```
class weights [w0, w1] = [1.0, 4.0]
instance weight for y=0  -> 4.0     # majority gets the minority's weight
instance weight for y=1  -> 1.0
```

So the model up-weights the majority class. Only `logistic_balanced_training` reaches
this path; `random_forest_balanced_training` and `SVM_balanced_training` pass sklearn's
`class_weight='balanced'` instead and are correct — note `random_forest.train()` has a
`balanced_training` branch that is dead code, since its wrapper never sets the flag.
Inconsistent and misleading either way.

**Fixed** by rewriting `get_instance_class_weights` on top of sklearn's
`compute_sample_weight('balanced', y)`, which removes the hand-rolled label matrix
entirely. All three balanced-training wrappers now go through the same
`balanced_training=True` path; because the weights are numerically identical to
sklearn's `class_weight='balanced'`, Random Forest and SVM predictions are unchanged
(verified). The visible effect is on `'Logistic balanced training'`: on 300:60 Gaussian
data the black box's **balanced accuracy rises from 0.833 to 0.905** while overall
accuracy is flat (0.944 → 0.942) — exactly the trade balancing is meant to make.
Tests: `clime/data/tests/test_costs.py`.

### B3 — class-balanced fidelity metrics truncate their weights. `clime/evaluation/faithfulness.py` — **FIXED**

`weights = data['y'].copy()` inherits `int64` from the labels, then float weights are
assigned into it and silently floor. Measured on 7:3 data: minority weight `2.333` is
stored as `2`.

Affects `'fidelity (class balanced)'` and `'fidelity (local and balanced)'`. Not used in
any paper figure (those use `'fidelity (local)'`), but any result using them is wrong.

**Fixed** — weights are now built in a fresh float array. On imbalanced (300:60) data
`'fidelity (class balanced)'` moves from 0.7255 to 0.6638. On balanced data there is no
change, since every weight is 1.0 either way.
Test: `test_class_weights_are_not_truncated`.

### B4 — `get_explanation()` fails for the logit surrogate. `clime/explainer/BLIMEY.py` — **FIXED**

```
normal        -> [-0.187, -0.209]
logit         -> IndexError: too many indices for array: array is 1-dimensional
logistic reg  -> [2.764, 2.888]
```

`logit_ridge.fit` targets a 1-D `y[:, 1]`, so `coef_` is 1-D and `coef_[0, :]` blows up.
Blocks all feature-importance work on Logit-LIME.

**Fixed** with `np.atleast_2d(self.surrogate_model.coef_)[0, :]`. All explainers now
return one finite importance per feature. Note the surrogates' coefficients live on
**different scales** (probability, logit, log-odds) — they are not directly comparable
and a longer write-up needs to say how it normalises them.
Test: `clime/explainer/test_explanations.py`.

### B5 — every plot is hard-clipped to [0, 1]. `clime/utils/plots.py` — **FIXED**

`_heatmap_interpolate` clips `zi` to `[0,1]` with `vmin=0, vmax=1`;
`plot_multiple_bar_dicts` and `plot_line_graphs` default `ylims=[0,1]` and only ever
expand. Correct for fidelity/accuracy, actively destructive for Brier score (~0.02,
invisible) and log loss (unbounded, clipped away). **This is what made the June 2024
Logit-LIME comparison look like a null result.**

**Fixed** by adding `clime.evaluation.METRIC_RANGES`, which declares a fixed display
range for bounded metrics (fidelity → `(0, 1)`, spearman → `(-1, 1)`) and `None` for
unbounded ones (Brier, log loss, KL). Line, bar and heatmap plots look the metric up and
either pin the axis or scale it to the data. Subplots showing the *same* metric still
share a scale, so comparisons between explainers stay honest.

The mutable `ylims=[0, 1]` defaults were a second, worse bug: Python reuses that list
across calls, so one plot with a score of 5.0 permanently rescaled every later plot in
the session. All such defaults are now `None`.
Tests: `clime/utils/test_utils.py`.

### B6 — a metric key is unreachable. `clime/evaluation/__init__.py` — **FIXED**

```python
'fidelity (query probs)':       query_probs_fidelity,
'fidelity (local query probs)': query_probs_fidelity,   # should be query_probs_local_fidelity
```

`query_probs_local_fidelity` is defined, exported, and never callable from the pipeline.
Selecting the "local" variant silently runs the global one. These metrics implement the
"classes = above/below the query point's probability" idea (also in
`'bLIMEy (cost sensitive sampled - probs)'`, via
`costs.weights_based_on_class_either_side_of_prob`) — an unwritten-up variant that
redefines the class split relative to `q` instead of at 0.5. Worth revisiting under the
aLIMEgn framing.

**Fixed** — the key now points at `query_probs_local_fidelity`. Selecting it changes the
score materially (0.854 → 0.680 on the Gaussian config), confirming the two metrics were
never the same thing.
Test: `test_local_query_probs_metric_is_actually_local`.

### B7 — `get_points_between_class_means` normalises by the wrong quantity. `clime/evaluation/key_points.py` — **FIXED**

`gradients /= np.sum(gradients)` divides the mean-difference vector by the **sum of its
components**, not its norm. If the components roughly cancel — which happens whenever the
class means differ in opposite directions across features — the denominator approaches
zero and the query-point line's scale explodes. Constructed a case with exactly
cancelling components: **every query point comes back `nan`**.

**Fixed** with `np.linalg.norm(gradients)`, plus an explicit exception when the two class
means coincide. Two things worth knowing:

- The normalisation constant **cancels out** of the resulting query points (scaling
  `gradients` by `c` scales the `min_`/`max_` solve by `1/c`), so the line is unchanged
  wherever the old code didn't blow up. Verified to 1e-16.
- `np.sum(gradients)` could be **negative**, which silently *reversed the direction* of
  the line. The orientation is now canonical: always class 0 → class 1. Breast Cancer is
  one of the datasets whose line was reversed — see "Effect on published results".

Tests: `test_between_class_means_survives_cancelling_means`, `..._line_spans_the_data`,
`..._identical_means_raises`.

### B8 — dead / stale code — **FIXED**

- `clime/pipeline/multiple_runs.py::get_avg` reads `opts['class samples']`; the key has
  been `opts['data params']['class_samples']` since the June 2023 refactor. It raises
  `KeyError` on any current config. `clime/main.py` calls it, so **`clime/main.py` is
  broken**. `todo.txt` has flagged this for over two years.
- `clime/evaluation/average_score.py::get_avg_score` is imported but never registered;
  its `explainer_generator(clf, data_dict, query_point=...)` call signature no longer
  matches `explainer_generator.__call__(clf, train_data, test_data, query_point)`. It
  would fail if wired back in.
- `clime/explainer/LIME.py::LIME` (the reference implementation) references an
  unimported `clime` and a commented-out `LimeTabularExplainer` import.
- `clime/models/QDA.py` is commented out of the registry.
- `Untitled.ipynb` is a scratchpad from the July 2023 logit derivation. Superseded by
  `logit_regression.py`. Safe to delete.
- `.ipynb_checkpoints/` and `__pycache__/` are on disk but correctly gitignored — they
  are stale build artefacts from 2023 and can be cleared.

**Fixed** — deleted `multiple_runs.py`, `average_score.py`, `main.py`, `Untitled.ipynb`,
the unfinished reference `LIME` class and `.ipynb_checkpoints/`, and dropped the now-dead
imports from `clime/pipeline/__init__.py` and `clime/evaluation/__init__.py`. `QDA.py` is
left in place — it is a real model that just needs re-enabling, not dead code.

### B9 — `freezeargs` mutates its caller's dict — **FIXED**

`recursive_freeze` assigns `value[k] = recursive_freeze(v)` into the original dict before
wrapping it. Callers' `opts` come back with `frozendict`/`tuple` values. Combined with
`@cache`, this makes stale results after a code edit easy to get in a notebook (restart
the kernel when comparing before/after a change).

**Fixed** — `recursive_freeze` now builds new containers instead of freezing in place, so
callers keep their plain dicts. The cache still cannot see code changes, so restarting the
kernel when comparing before/after an edit is still necessary.
Tests: `test_freezeargs_does_not_mutate_the_caller`, `test_freezeargs_result_is_hashable`.

### B10 — results are not reproducible with `parallel_eval=True` — **OPEN**

Found while verifying the fixes above. Running the *same* configuration with the *same*
code twice gives different numbers:

```
Gaussian | bLIMEy (cost sensitive sampled)        max|Δ| = 2.2e-03
Breast Cancer | bLIMEy (normal)                   max|Δ| = 5.6e-03
logistic-balanced-training                        max|Δ| = 3.6e-03
logit-explainer                                   max|Δ| = 6.3e-04
```

`bLIMEy._sample_locally` draws its 10 000 samples from the **global** numpy RNG, which is
seeded once at `import clime`. Results therefore depend on how many draws happened
before — and under `multiprocessing`, on which worker handles which query point.

A single serial sweep is deterministic, but the scope is narrower than that sounds: the
same configuration run from a differently structured script drifts at the third decimal,
because the preceding draws differ. `key_points.get_local_points` has the same problem, so
`evaluation data: 'sample locally'` **and the Logit-LIME diagnostic** are both affected
(measured: saturation 65.9% vs 65.6%, gap 0.044 vs 0.042 for the same configuration).

The effect is small (~1e-3, well below the effects the paper reports, which are ~0.1–0.4)
so **no conclusion in the paper is at risk**. But the exact figures cannot be regenerated,
and it makes small effects — precisely the regime the Logit-LIME comparison lives in —
impossible to distinguish from noise.

Not fixed, because the fix re-bases every number in the repo and that should be your call.
The fix is to give each explainer its own seeded generator derived deterministically from
the query point, so the result is independent of scheduling and call order:

```python
# clime/explainer/BLIMEY.py, in _sample_locally
seed = int.from_bytes(hashlib.sha256(np.ascontiguousarray(self.query_point).tobytes()).digest()[:4], 'big')
rng = np.random.default_rng(clime.RANDOM_SEED + seed)
sampled_data['X'] = rng.multivariate_normal(self.query_point, cov, self.samples)
```

`key_points.get_local_points` has the same problem and needs the same treatment for
`evaluation data: 'sample locally'`. Use a stable hash (`hashlib`, not `hash()`, which is
salted per process).

### B11 — `log loss` is cross-entropy, not KL — **FIXED**

`log_loss_score` computes the cross entropy `H(y, p)` between the black box's
probabilities and the surrogate's. Its minimum is `H(y)`, not zero, and that floor is
often most of the number. Measured on the logistic black box: floor **0.111 of a reported
0.132 — 84%**, compressing a genuine 24 500x difference between surrogates into an
apparent 1.19x.

**Fixed** by adding proper KL divergence metrics — `'KL divergence'` and
`'KL divergence (local)'` — which are zero exactly when the surrogate reproduces the black
box. `'log loss'` is kept (the cross entropy is sometimes what you want) but its docstring
now states the floor. The pre-existing `'KL'` key was renamed
`'mutual information (RBIG)'`, which is what `rbig_kl` actually computes — it estimates
mutual information via RBIG, not a divergence, and having it sit next to a real KL metric
under the name "KL" was misleading.

### B12 — the logistic-regression surrogate crashes far from the boundary — **FIXED**

`bLIMEy (logistic regression)` thresholds the black box's probabilities and fits sklearn's
`LogisticRegression`. Away from the decision boundary the black box predicts a single
class over the whole neighbourhood, and sklearn raises
`ValueError: This solver needs samples of at least 2 classes`. This **took down the entire
experiment sweep** partway through.

The failure regime is exactly the one this project studies: high-confidence regions far
from the boundary. `pytest` missed it because the existing fixtures happen to place every
query point where both classes appear locally.

**Fixed** — the surrogate now detects the single-class case and falls back to a constant
predictor (zero coefficients, constant probability) instead of raising.
Test: `test_explainer_builds_far_from_the_boundary`, parameterised over every registered
explainer.

### B13 — query-probability weighting produces all-zero weights — **FIXED**

Found by the test written for B12. `bLIMEy (cost sensitive sampled - probs)` weights
samples by which side of the *query point's* probability they fall on. In a saturated
neighbourhood every sample sits at the same probability as the query point, so nothing
lies to either side, every weight comes out zero, and the ridge fit dies with
`ZeroDivisionError: Weights sum to zero`.

(The proximate cause is that `np.round(0.5)` is `0` under banker's rounding, so exact ties
are assigned to *neither* side rather than one.)

**Fixed** — `weights_based_on_class_either_side_of_prob` now detects the degenerate case,
warns, and falls back to uniform weights. The tie-assignment rule is left alone
deliberately: changing it would alter the semantics of an existing metric.

### B14 — the SVM black box is badly misconfigured — **FIXED**

`clime/models/svm.py` hard-coded `gamma=2`, an extremely wide RBF kernel for standardised
data with more than a couple of features. Measured test accuracy:

```
Breast Cancer   0.626      (majority class rate ~0.63)
Pima Diabetes   0.654      (majority class rate ~0.65)
Banknote        0.998
Gaussian        0.920
```

So on the two higher-dimensional datasets the SVM was at chance, emitting a near-constant
probability (~0.63, s.d. 0.0005 within a neighbourhood). Any result using the SVM on those
datasets was meaningless — both surrogates trivially fit a constant.

**Fixed** to sklearn's default `gamma='scale'`. Accuracies become 0.925 / 0.957 / 1.000 /
0.779. Does not affect any published figure: the CIKM paper uses a random forest.

### Effect on published results

Before/after snapshots of the paper's configurations, run in **serial** mode so the
comparison isn't confounded by B10:

| configuration | before | after | verdict |
|---|---|---|---|
| Gaussian μ=±1, standard `w_x` | 0.949500 | 0.949500 | identical |
| Gaussian μ=±1, class balanced `w_xc` | 0.976112 | 0.976112 | identical |
| Gaussian μ=±3, standard `w_x` | 0.999992 | 0.999992 | identical |
| Gaussian μ=±3, class balanced `w_xc` | 0.999992 | 0.999992 | identical |
| Breast Cancer, standard `w_x` | 0.839371 | 0.834625 | same query points, order reversed |
| Breast Cancer, class balanced `w_xc` | 0.962554 | 0.963615 | same query points, order reversed |

All four synthetic configurations are bit-identical. For Breast Cancer the *set* of query
points is identical (max difference 0.0) but B7 reversed the traversal direction, so
query point 0 is now the one that used to be 19. Because the surrogate at each point draws
from the shared global RNG (B10), visiting them in the other order shifts each score by
~5e-3 — the same magnitude as B10's run-to-run noise, and two orders of magnitude below
the ~0.4 fidelity drop the paper reports.

**If you regenerate Figure 3(a) (Breast Cancer), its x-axis will be mirrored** relative to
the published version. The curve's content is unchanged. The published PNGs in
`experiments/figs/sampling/` have not been touched.

---

## 7. Suggested next experiments

Roughly in cost order. The first two are hours, not days.

**E1 — Logit-LIME's operating regime.** ***Done — see §4, and the paper in
`~/Repos/Overleaf/Logit-LIME/`.*** Remaining follow-ups, in order of value:
(a) test the Δ diagnostic on black boxes outside the seven used here, and establish a
threshold rather than the eyeballed 0.35;
(b) sweep the locality kernel width `k`, which defines "local" for both the surrogate's
training data and the evaluation metric and so partly determines the effect being
measured — this is the obvious reviewer question;
(c) test whether the effect survives an interpretable-domain transform, where the
surrogate operates on binary indicators rather than raw features;
(d) more seeds and splits — everything so far is a single split per configuration.

**E2 — the aLIMEgn evaluation-target sweep.** Fix the black box and the query points,
toggle `evaluation data` between `'test data'` and `'sample locally'`, and sweep across
the `between_class_means` line. This directly answers aLIMEgn open question 2. Expected
signature: the two evaluations agree near the boundary and diverge in the tails, with
standard LIME collapsing only under `'test data'` — which would demonstrate that the
CIKM finding *is* a `P(X,y)` vs `P(X,ŷ)` misalignment, and not a property of LIME.
Everything needed is already implemented.

**E3 — degrade the black box on purpose.** aLIMEgn's whole premise is `P(X,ŷ) ≠ P(X,y)`.
Force the gap: label noise on the training set, an underfit model (`MLP_simple` with few
iterations, or a depth-limited forest), or class-imbalanced training data. Then repeat
E2. Prediction: the worse `f` is, the more the two evaluation targets diverge, and the
more the `ŷ`-derived class weights beat `y`-derived ones. This is the experiment that
turns the framing note into a paper, and it needs one new dataset wrapper (label noise)
plus one model config.

**E4 — settle sampled vs data class costs.** Turn
`pics/rf balanced, sampled cost helps but class cost hurts slightly.png` into a real
result: `'bLIMEy (cost sensitive sampled)'` vs `'bLIMEy (cost sensitive class)'` vs
`'bLIMEy (normal)'`, across all UCI datasets, both balanced and imbalanced black boxes.
B2 is fixed, so the balanced-training models now actually balance (this changed Logistic
balanced training's balanced accuracy from 0.833 to 0.905 on 300:60 data), making this
comparison meaningful for the first time. A clean "local imbalance is the signal, global imbalance is not" result strengthens both
threads.

**E5 — grid heatmaps for the paper's datasets.** `evaluation points: grid` exists and was
never used for a figure; every published plot is the 1-D line. A 2-D fidelity map over
the PCA plane, overlaid on the decision surface, would show *where* the mismatch lives
rather than just that it exists along one transect. 400 explainers per config, so
budget compute — but the `@cache` and `parallel_eval` machinery is already there. This is
the strongest new *figure* available for a longer version of the CIKM paper.

**E6 — beyond the exponential kernel.** The `k = 0.75·√|D|` kernel width is inherited
from LIME's defaults and is doing a lot of unexamined work: it defines "local" for both
`X_s` and the evaluation metric simultaneously, so it partly determines the effect being
measured. Sweeping it would answer the obvious reviewer question ("isn't this just a
badly chosen kernel width?"). There is a `todo.txt` entry for a "param for weight of
LIME balancer" that never got done — same family of ablation.

**E7 — a non-tabular domain.** Everything is tabular with no interpretable-domain
transform, which the paper explicitly notes as a simplification. Whether the location
effect survives an interpretable domain (superpixels, bag-of-words) is the biggest
open generality question, and the most work.

---

## 8. Suggested code changes

**Done on 2026-08-07** (see §6 for detail):

1. ~~B1 — rename the three `__init__,py` files~~ — done, wheel verified.
2. ~~B5 — metric-aware plot limits~~ — done via `clime.evaluation.METRIC_RANGES`.
3. ~~B4 — `get_explanation()` for logit surrogates~~ — done.
4. ~~B2, B3, B6, B7 — correctness fixes with regression tests~~ — done. The suite went
   from 7 tests that only assert runs *complete* to 50, including numerical assertions on
   known inputs. B2 and B3 would both have been caught by these.
5. ~~Delete dead code~~ — done: `main.py`, `multiple_runs.py`, `average_score.py`, the
   reference `LIME` class, `Untitled.ipynb`, `.ipynb_checkpoints/`.

Also done since: B11 (KL divergence metrics), B12 (logistic-regression surrogate crashing
far from the boundary), B13 (all-zero query-probability weights), B14 (SVM `gamma=2`),
plus three new models in the registry — gradient boosting and Platt/isotonic-calibrated
random forests — which the calibration control needed.

**Still open**, roughly in order of value:

1. **B10 — deterministic sampling.** See §6. Needs your decision because it re-bases every
   number. Until then, use `parallel_eval=False` for anything where the effect is small.
2. **Persist results.** `run_pipeline` caches in-process only; every notebook restart
   re-runs everything, and a 400-point grid sweep is expensive. A `joblib.Memory` cache
   keyed on the frozen `opts` would make E5 comfortable and make results reproducible
   across sessions. Worth doing *after* B10, since caching nondeterministic results
   freezes whichever draw you happened to get.
3. **A headless experiment runner.** `experiments/*.py` each hard-code a full `opts` dict
   and duplicate ~40 lines. A YAML/JSON config plus one runner script would make the
   sweeps in §7 tractable and, more importantly, make the exact configuration behind
   each figure recoverable — right now the only record of the last experiment is
   widget state serialised inside a notebook.
4. **Pin the environment properly.** `requirements.txt` pins only sklearn; numpy,
   scipy and matplotlib float. Given the known sklearn 1.2.2 breakage, a lockfile (or at
   least an `environment.yml` capturing the working conda env) would protect the
   published results. Also: the sklearn incompatibility itself is worth 30 minutes to
   diagnose — being stuck on a 2022 release will get more painful, not less.
5. **Re-enable `QDA`.** It is implemented and commented out of the registry with no
   recorded reason.
6. **Push B1 and un-comment the Colab badge.** The badge installs from GitHub, so it
   stays broken until the rename is pushed.
