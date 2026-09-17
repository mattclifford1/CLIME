'''
The interpretable-domain transform: surrogates fitted on binary patch indicators.

Everything else in this study fits the surrogate in the raw feature space, which the paper
says explicitly isolates the effect being studied from the biases an interpretable-domain
transform introduces. This module is that transform, and exists to answer the obvious
follow-up: does the choice of space still matter once the surrogate no longer sees the
features at all?

This is what LIME actually does on images. The image is cut into patches, and the surrogate
sees one binary indicator per patch - 1 for "present", 0 for "replaced by a baseline" -
rather than pixels:

    x(z) = b + sum_j z_j (q_j - b_j)      restricted to the pixels of patch j

The reason the question is answerable here is that the ground truth survives the transform
in closed form. For a black box whose log-odds are linear in x with gradient beta,

    logit f(x(z)) = logit f(b) + sum_j z_j * gamma_j,     gamma_j = sum_{i in patch j} beta_i (q_i - b_i)

so the black box IS exactly linear in z, and gamma is the true importance of each patch -
not an approximation of one. Two consequences worth being explicit about, because they cut
in opposite directions:

  - Logit-LIME's hypothesis class contains the truth exactly, so it should recover gamma
    up to ridge shrinkage. This is not a fair fight, and it is not meant to be: it is the
    group A argument carried into the interpretable domain, where the claim to test is
    whether the advantage survives at all, not whether it is a surprise.
  - Standard LIME is once again fitting a line to a sigmoid, now over the hypercube.

`validate_linearity` checks the identity numerically rather than trusting the algebra. For
a black box whose log-odds are NOT linear in x the identity simply fails, and there is no
closed-form patch truth - the same boundary as the rest of this study, drawn in a
different space.

Saturation is a live constraint here rather than a footnote. Turning patches off moves the
image a long way, so the log-odds swing across the hypercube is large - about 40 for LDA
on this data - and float64 pins a probability to exactly 0 or 1 well before that, capping
any measurable logit near +-27.6. The identity still holds exactly where it can be seen -
2e-12 for logistic regression, and ~6e-5 for LDA and Nearest Class Mean, which is
predict_proba's own rounding rather than deviation (see validate_linearity). The fraction
of z on which it cannot be seen is recorded per configuration, because the surrogates are
fitted on those same clipped probabilities.

The baseline b is the dataset mean, which is the zero vector in the standardised space the
pipeline hands over. That is the "grey out the patch" baseline of image LIME, and it makes
gamma_j the effect of revealing patch j against an average image.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn
import clime
from clime.data.utils import costs

SIDE = 8            # digits are 8x8
PATCH = 2           # 2x2 patches -> 16 indicators
SAMPLES = 10000     # as everywhere else in this study
EPS = 1e-12

# a new draw site needs its own salt, or two sites return identical points and a surrogate
# is scored on its own training sample (CLAUDE.md, clime/utils/seeding.py)
TRAIN_SALT = 'patch surrogate training sample'
EVAL_SALT = 'patch local evaluation sample'


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p/(1 - p))


def patch_indices(side=SIDE, patch=PATCH):
    '''pixel indices of each patch, in row-major grid order'''
    if side % patch:
        raise ValueError(f'patch {patch} does not tile a {side}x{side} grid')
    out = []
    for r in range(0, side, patch):
        for c in range(0, side, patch):
            out.append(np.array([(r + a)*side + (c + b)
                                 for a in range(patch) for b in range(patch)]))
    return out


def compose(Z, q, baseline, patches):
    '''binary patch vectors -> images: start from the baseline, reveal the patches that are on'''
    Z = np.atleast_2d(np.asarray(Z))
    X = np.repeat(np.asarray(baseline, dtype=np.float64)[None, :], Z.shape[0], axis=0)
    for j, idx in enumerate(patches):
        X[np.ix_(Z[:, j] == 1, idx)] = q[idx]
    return X


def true_patch_coefficients(beta, q, baseline, patches):
    '''
    the exact importance of each patch, for a black box with linear log-odds

    gamma_j is the change in log-odds from revealing patch j, which is the sum of the
    black box's pixel weights over that patch times how far those pixels move from the
    baseline. Note it depends on q as well as beta: the same patch matters more in an
    image whose pixels there are far from average.
    '''
    beta = np.asarray(beta, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    return np.array([beta[idx] @ (q[idx] - baseline[idx]) for idx in patches])


def validate_linearity(clf, gamma, q, baseline, patches, n=400, seed=0):
    '''
    check the identity numerically, and say how much of z it could not be checked on

    Returns (residual, saturated_fraction). If the identity holds then
    logit f(x(z)) - gamma.z is CONSTANT in z, so the residual is the largest deviation
    from its own mean - which avoids needing any particular reference point to be
    measurable.

    Measured only where f has not saturated, and this is not a technicality. float64
    cannot represent a probability closer to 0 or 1 than about 1e-16, so logit f is capped
    near +-27.6 however large the true log-odds are. A black box whose log-odds swing by
    40 across the hypercube - LDA and Nearest Class Mean both do here - therefore breaks
    the identity *as measured* while satisfying it exactly. Checking on saturated z
    reports a residual of ~10 and means nothing; checking on unsaturated z reports ~1e-12
    and means what it says. The saturated fraction is returned alongside because the
    surrogates are fitted on those same clipped probabilities, so it is a property of the
    experiment rather than of this check.

    The residual that remains on unsaturated z is rounding, not deviation. On LDA it is
    ~6e-5 measured this way; recomputed from the model's exact decision_function, which is
    the log-odds and is never squashed into a probability, it is 8e-15. The difference is
    where the probability gets close to 1: at p = 1 - 1.15e-12 float64 has about four
    significant digits left in (1 - p), and logit reads them all.
    '''
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, size=(n, len(patches)))
    p = clf.predict_proba(compose(Z, q, baseline, patches))[:, 1]
    saturated = (p <= EPS) | (p >= 1 - EPS)
    frac = float(np.mean(saturated))
    if (~saturated).sum() < 2:
        return float('nan'), frac
    c = logit(p[~saturated]) - Z[~saturated] @ gamma
    return float(np.max(np.abs(c - np.mean(c)))), frac


def sample_Z(n_patches, q, samples=SAMPLES, salt=TRAIN_SALT):
    '''
    Bernoulli(1/2) over the patches, with the query's own representation first.

    z = 1 is the query image itself, which is the point being explained and carries the
    largest locality weight; LIME includes it for the same reason.
    '''
    rng = clime.utils.rng_from_point(q, salt=salt)
    Z = rng.integers(0, 2, size=(samples, n_patches))
    Z[0, :] = 1
    return Z


def kernel_weights(Z):
    '''
    the same exponential kernel as everywhere else, with distance measured to the all-ones
    vector - the query point's representation in the interpretable domain
    '''
    return costs.weights_based_on_distance(np.ones(Z.shape[1]), Z)


class PatchLIME:
    '''
    a bLIMEy surrogate fitted in the interpretable domain

    Mirrors clime.explainer.BLIMEY: same ridge, same alpha, same locality kernel, same
    class-1 convention for the coefficients. The only change is what the surrogate sees -
    binary patch indicators instead of pixels - which is the whole point.
    '''

    def __init__(self, black_box_model, query_point, patches, baseline=None,
                 samples=SAMPLES, train_logits=False):
        self.clf = black_box_model
        self.q = np.asarray(query_point, dtype=np.float64).ravel()
        self.patches = patches
        self.baseline = (np.zeros_like(self.q) if baseline is None
                         else np.asarray(baseline, dtype=np.float64))
        self.train_logits = train_logits

        Z = sample_Z(len(patches), self.q, samples=samples)
        probs = self.clf.predict_proba(compose(Z, self.q, self.baseline, patches))
        weights = kernel_weights(Z)

        if train_logits:
            self.surrogate_model = clime.models.logit_ridge(
                alpha=1, fit_intercept=True, random_state=clime.RANDOM_SEED)
        else:
            self.surrogate_model = sklearn.linear_model.Ridge(
                alpha=1, fit_intercept=True, random_state=clime.RANDOM_SEED)
        self.surrogate_model.fit(Z, probs, sample_weight=weights)

    def get_explanation(self):
        '''class 1, taking the last coefficient row - see BLIMEY.get_explanation (B15)'''
        return np.atleast_2d(self.surrogate_model.coef_)[-1, :]

    def predict_proba(self, Z):
        p = self.surrogate_model.predict(np.atleast_2d(np.asarray(Z)))
        return np.clip(p, 0, 1) if not self.train_logits else p

    def predict(self, Z):
        return (self.predict_proba(Z)[:, 1] >= 0.5).astype(np.int64)
