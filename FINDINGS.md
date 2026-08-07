# CLIME — state of the research

Written 2026-08-07 by reconstructing the repo, the notebook outputs and the Overleaf
projects. Last code commit was **2024-06-06**; the work has been dormant since.

---

## 1. Where the project stands

Three threads, in order of maturity:

| Thread | Status | Overleaf | Code |
|---|---|---|---|
| **CIKM'23 — location-agnostic surrogates** | Published, DOI `10.1145/3583780.3615284` | `~/Repos/Overleaf/CIKM-2023-camera-ready` | complete, figures reproducible |
| **Logit-LIME** | Sketch only (~1 page of prose, no results) | `~/Repos/Overleaf/Logit-LIME` | implemented and running, never written up |
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

**Known gap.** `bLIMEy.get_explanation()` raises `IndexError` for the logit surrogate
(§6). Fidelity metrics don't touch it, so the pipeline runs clean, but you cannot
currently extract feature importances from a Logit-LIME explainer — which is the actual
explanation. This must be fixed before any qualitative or feature-agreement result.

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

### B1 — the package doesn't install. `setup.py`, `clime/data/*`

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

Fix: rename to `__init__.py` (three `git mv`s), then rebuild and check the wheel.

### B2 — `'Logistic balanced training'` trains anti-balanced. `clime/data/utils/costs.py:40`

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

Fix: `Y = concat(1-y, y)` (or index-based construction), unify all three wrappers onto
one mechanism, add a test asserting the minority class gets the larger weight.

### B3 — class-balanced fidelity metrics truncate their weights. `clime/evaluation/faithfulness.py:170`

`weights = data['y'].copy()` inherits `int64` from the labels, then float weights are
assigned into it and silently floor. Measured on 7:3 data: minority weight `2.333` is
stored as `2`.

Affects `'fidelity (class balanced)'` and `'fidelity (local and balanced)'`. Not used in
any paper figure (those use `'fidelity (local)'`), but any result using them is wrong.

Fix: `weights = np.ones(len(data['y']), dtype=float)`.

### B4 — `get_explanation()` fails for the logit surrogate. `clime/explainer/BLIMEY.py:63`

```
normal        -> [-0.187, -0.209]
logit         -> IndexError: too many indices for array: array is 1-dimensional
logistic reg  -> [2.764, 2.888]
```

`logit_ridge.fit` targets a 1-D `y[:, 1]`, so `coef_` is 1-D and `coef_[0, :]` blows up.
Blocks all feature-importance work on Logit-LIME.

Fix: `np.atleast_2d(self.surrogate_model.coef_)[0, :]`, or give `logit_ridge` a
`get_explanation`. Note the three surrogates' coefficients live on **different scales**
(probability, logit, log-odds) — they are not directly comparable and a longer write-up
needs to say how it normalises them.

### B5 — every plot is hard-clipped to [0, 1]. `clime/utils/plots.py`

`_heatmap_interpolate` clips `zi` to `[0,1]` with `vmin=0, vmax=1`;
`plot_multiple_bar_dicts` and `plot_line_graphs` default `ylims=[0,1]` and only ever
expand. Correct for fidelity/accuracy, actively destructive for Brier score (~0.02,
invisible) and log loss (unbounded, clipped away). **This is what made the June 2024
Logit-LIME comparison look like a null result.**

Fix: derive limits from the data, or attach a `range` / `higher_is_better` attribute to
each metric in `AVAILABLE_EVALUATION_METRICS` and let the plotters read it. The mutable
default `ylims=[0, 1]` in those signatures is also a latent bug — Python reuses the list
across calls, so limits leak between plots within one session.

### B6 — a metric key is unreachable. `clime/evaluation/__init__.py`

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
aLIMEgn framing; fix the key first.

### B7 — `get_points_between_class_means` normalises by the wrong quantity. `clime/evaluation/key_points.py:76`

`gradients /= np.sum(gradients)` divides the mean-difference vector by the **sum of its
components**, not its norm. If the components roughly cancel — which happens whenever the
class means differ in opposite directions across features — the denominator approaches
zero and the query-point line's scale explodes. It works on the paper's datasets by
luck. Should be `np.linalg.norm(gradients)`.

### B8 — dead / stale code

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

### B9 — `freezeargs` mutates its caller's dict

`recursive_freeze` assigns `value[k] = recursive_freeze(v)` into the original dict before
wrapping it. Callers' `opts` come back with `frozendict`/`tuple` values. Combined with
`@cache`, this makes stale results after a code edit easy to get in a notebook (restart
the kernel when comparing before/after a change). Copy before freezing.

---

## 7. Suggested next experiments

Roughly in cost order. The first two are hours, not days.

**E1 — re-run the June 2024 Logit-LIME comparison with working axes.** Fix B5, rerun the
saved notebook config (Gaussian μ=±1, RF, grid, local Brier). The result may already be
there. Add log loss alongside Brier — they disagree about tail calibration, which is
exactly what logit-space regression should change. Cheap, and it either unblocks the
Logit-LIME paper or kills it.

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
Requires B2 fixed first, since the balanced-training models are currently anti-balanced.
A clean "local imbalance is the signal, global imbalance is not" result strengthens both
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

Ordered so each is independently useful.

1. **B1** — rename the three `__init__,py` files, rebuild the wheel, verify the Colab
   badge. Unblocks anyone else running this.
2. **B5** — metric-aware plot limits. Highest ratio of insight-unblocked to effort.
3. **B4** — `get_explanation()` for logit surrogates.
4. **B2, B3, B6, B7** — the correctness fixes, each with a regression test. `pytest`
   currently only asserts that runs *complete* (`isinstance(score, np.float64)`); it
   never asserts a value. A handful of numerical assertions on known inputs would have
   caught B2 and B3.
5. **Delete B8's dead code** — `main.py`/`get_avg`, `average_score.py`, the reference
   `LIME` class, `Untitled.ipynb`, and untrack `__pycache__`/`.ipynb_checkpoints`.
6. **Persist results.** `run_pipeline` caches in-process only; every notebook restart
   re-runs everything, and a 400-point grid sweep is expensive. A `joblib.Memory` cache
   keyed on the frozen `opts` would make E5 comfortable and make results reproducible
   across sessions.
7. **A headless experiment runner.** `experiments/*.py` each hard-code a full `opts` dict
   and duplicate ~40 lines. A YAML/JSON config plus one runner script would make the
   sweeps in §7 tractable and, more importantly, make the exact configuration behind
   each figure recoverable — right now the only record of the last experiment is
   widget state serialised inside a notebook.
8. **Pin the environment properly.** `requirements.txt` pins only sklearn; numpy,
   scipy and matplotlib float. Given the known sklearn 1.2.2 breakage, a lockfile (or at
   least an `environment.yml` capturing the working conda env) would protect the
   published results. Also: the sklearn incompatibility itself is worth 30 minutes to
   diagnose — being stuck on a 2022 release will get more painful, not less.
