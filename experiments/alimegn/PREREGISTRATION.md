# aLIMEgn — predictions registered before running

Written 2026-09-12, before any sweep in this directory was run. The repo's convention
(`experiments/logit_lime/PREREGISTRATION.md`) is to fix falsifiable statements first, so
that a taxonomy or an account cannot be quietly fitted to whatever came out.

## The claim being tested

From `~/Repos/Overleaf/aLIMEgn/main.tex` (June 2024): a surrogate's training sample should
target `P(X, ŷ)` — the distribution of the black box's *own* predictions — rather than
`P(X, y)`, the distribution that trained the black box. The two coincide when `f` has
learned the data well and diverge when it has not (weak labels, underfitting, shift).
The CIKM'23 class-balancing trick is then one way of using `ŷ` to align the sample.

## A correction to the framing, fixed before running

The note's own open questions 1 and 2 are answered by inspection of the pipeline, not by
experiment, and the answers change what there is to measure:

1. **Sampling can only choose a marginal over `X`.** The labels of a surrogate's training
   set always come from `f`, in every LIME variant. So "train on `P(X, ŷ)`" is automatic
   for the label half; the only free choices are the `X`-marginal and the sample weights.
2. **Every fidelity metric in this repo already scores against `ŷ`.** `fidelity`, `Brier`
   and `KL` all compare `g` to `f`, never to `y`. So the `evaluation data` switch
   (`'test data'` vs `'sample locally'`) does **not** contrast `P(X, y)` with `P(X, ŷ)`:
   it contrasts two `X`-marginals, both labelled by `f`. `FINDINGS.md` §5/E2 says
   otherwise and is wrong on that point.

`y` versus `ŷ` therefore enters in exactly two places, and those are what the sweeps here
manipulate: **class-conditional weighting** (class frequencies taken from `y` or from `ŷ`
over the same points) and **a deliberately degraded `f`** (which is what makes the two
sources differ at all).

## The weighting schemes compared

All six are real methods — none needs the test set or any label the deployer would not
have. Each supplies `sample_weight` for the surrogate's ridge, on top of (or instead of)
the locality kernel.

| key | class frequencies from | needs |
|---|---|---|
| `bLIMEy (normal)` | — (locality kernel only) | nothing |
| `bLIMEy (cost sensitive sampled)` | `ŷ` on the surrogate's own sample | nothing (CIKM'23) |
| `bLIMEy (cost sensitive class)` | `y` over the whole training set | training labels |
| `bLIMEy (local y)` | `y` on nearby training points | training data + labels |
| `bLIMEy (local yhat)` | `ŷ` on those same nearby training points | training data, no labels |
| `bLIMEy (density ratio)` | — (estimated `p_train(x)/p_sample(x)`) | training data, no labels |

`local y` against `local yhat` is the `y`-versus-`ŷ` contrast proper: identical points,
identical mechanism, only the source of the labels differs.

## Predictions

**P1 — the CIKM collapse is a property of the evaluation marginal, not of LIME.**
Standard LIME's local fidelity falls away from the decision boundary when scored on
`'test data'`, and that fall is at least 5× smaller when the same surrogates are scored on
`'sample locally'` — because the local sample *is* the surrogate's training marginal.
*Falsified if* the drop is comparable under both.

**P2 — class weighting is a correction to that marginal mismatch.** `cost sensitive
sampled` beats `normal` under `'test data'`, and its advantage shrinks under `'sample
locally'`. *Falsified if* the advantage is the same size under both, or larger locally.

**P3 — the aLIMEgn claim: as `f` degrades, `ŷ`-derived weights beat `y`-derived ones.**
`local yhat` beats `local y` on fidelity to `f`, and the gap grows monotonically with the
divergence between `P(ŷ|x)` and `P(y|x)` (swept via label noise on the training set, an
underfit `f`, and class-imbalanced training data). At zero degradation the two are within
seed noise. *Falsified if* the two are indistinguishable at every degradation level — in
which case `y` versus `ŷ` is not a real axis — or if `local y` wins as `f` worsens.

**P4 — the ordering reverses when the target is the truth rather than `f`.** Scored on
agreement with the *true* test labels near `q` instead of with `f`, `local y` beats
`local yhat`. This is the sharp version of the framing: aligning to `P(X, ŷ)` is right
only because the thing being explained is `f`. *Falsified if* `ŷ`-weights win on both
targets, which would mean the two objectives are not actually in tension.

**P5 — the covariate-shift reading.** `density ratio` — which estimates
`p_train(x)/p_sample(x)` directly and uses no labels at all — beats every class-weighting
scheme under `'test data'` evaluation. And CIKM's `ŷ` class weights correlate positively
with the estimated density ratio across sampled points (Spearman ρ > 0.3 at query points
away from the boundary), i.e. the class-balancing trick is a crude density-ratio
correction. *Falsified if* the density ratio does not help, which would mean the effect is
not covariate shift; or if the correlation is absent, which would mean the class trick
works for some other reason.

**P6 — the two axes are independent.** Degrading `f` does not change the `X`-marginal
mismatch, so P1's effect size stays roughly constant across the noise ladder while P3's
grows. *Falsified if* the marginal effect itself scales with degradation.

## Deliberately not predicted

Which *degradation mechanism* (label noise, underfitting, imbalance) produces the largest
`y`/`ŷ` divergence per unit of lost accuracy. Recorded and reported, not predicted.

---

## Second registration, 2026-09-13: local versus global class imbalance

Registered after the first six were run and analysed, before `sweep_balance.py` ran. This
is `FINDINGS.md` E4, which has been open since the CIKM work: the claim there is that
*local* class imbalance is the useful signal and *global* class imbalance is not, resting
on one unreported PNG. The grid crosses two manipulations — how imbalanced the black box's
**training data** are, and whether the black box **corrects for that imbalance during
training** — over the same six weighting schemes.

**P7 — local imbalance is the signal.** `cost sensitive sampled` (frequencies from `ŷ` on
the surrogate's own sample) beats `cost sensitive class` (frequencies from `y` over the
whole training set) on test-set fidelity in all four cells of {natural, undersampled
training data} × {normal, balanced training}, and the global scheme does not beat standard
LIME in any cell. *Falsified if* the global scheme wins in the undersampled cell — which is
where global imbalance is largest, and so where it has the best chance of being the signal.

**P8 — balanced training shrinks the gain, for the same reason degradation did.** P6 found
that the thing class weighting corrects is the black box's own confident one-sidedness away
from the boundary. Training the black box with balanced class weights moves its boundary
towards the minority class and makes its predictions less one-sided in exactly that region,
so the gain from `cost sensitive sampled` should be **smaller** against a balance-trained
black box than against the same model trained normally, on the same data. *Falsified if*
the gain is equal or larger under balanced training — which would mean P6's mechanism was
a property of label noise rather than of one-sidedness as such.

P8 is the informative one: it is the same mechanism reached by a manipulation that has
nothing to do with corrupting labels.
