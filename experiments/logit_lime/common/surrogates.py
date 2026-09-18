'''
Surrogates whose behaviour is prescribed rather than fitted, for probing the instruments.

Everything the study measures is a property of a fitted surrogate, which makes it hard to
say what an instrument is responding to: a fit changes its direction, its scale and its
intercept at once. These classes fix all three by hand, so an instrument can be moved
along one axis at a time.

  LinearLogitSurrogate   logit g(x) = a + b.(x - q), with a and b supplied.  The base
                         class of common/taylor.py's two Taylor surrogates, which are the
                         special case b = grad logit f(q).

  rotate / family        the synthetic family used in figures/fig_instruments.py,

                             logit g_{theta,s}(x) = logit f(q)
                                                    + s * R(theta) grad logit f(q) . (x-q)

                         whose cosine to the explanation ground truth is cos(theta) by
                         construction and whose coefficient norm is s times the truth's.
                         The x axis of that figure IS the explanation error, with no fit
                         in the loop to confound it.

  ConstantSurrogate      the explainer that explains nothing: g is the locality weighted
                         mean of f over the neighbourhood (the constant that minimises the
                         Brier score there), and its explanation is the zero vector.  It
                         is the base rate an instrument has to beat to be worth reading.

None of these is registered in clime.explainer.AVAILABLE_EXPLAINERS, deliberately: a
registry entry is offered in the notebook UI and swept by the test suite as if it were a
method anyone should use, and none of these is one.  They expose the explainer contract -
predict, predict_proba, get_explanation - so every registered metric scores them unchanged.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np

import clime
from clime.data.utils import costs

EPS = 1e-12


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p/(1 - p))


class LinearLogitSurrogate:
    '''logit g(x) = a + b.(x - q), with b fixed rather than fitted'''

    def __init__(self, query_point, intercept=0.0, coef=None):
        self.q = np.asarray(query_point, dtype=np.float64).ravel()
        self.intercept = float(intercept)
        self.coef = (np.zeros_like(self.q) if coef is None
                     else np.asarray(coef, dtype=np.float64).ravel())
        self.degenerate = bool(np.linalg.norm(self.coef) == 0
                               or not np.all(np.isfinite(self.coef)))
        if not np.all(np.isfinite(self.coef)):
            self.coef = np.zeros_like(self.q)

    def get_explanation(self):
        return self.coef

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        z = self.intercept + (X - self.q) @ self.coef
        p = 1.0/(1.0 + np.exp(-np.clip(z, -700, 700)))
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        # the same convention as bLIMEy.predict: the boundary itself is class 1
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


class ConstantSurrogate(LinearLogitSurrogate):
    '''
    g(x) = p_bar everywhere, explanation = 0.

    p_bar is the locality weighted mean of f over the neighbourhood, which is the constant
    minimising the local Brier score, so this is the strongest possible surrogate that
    says nothing about any feature - not a straw man.
    '''

    def __init__(self, query_point, p_bar):
        super().__init__(query_point, intercept=_logit(p_bar), coef=None)
        self.p_bar = float(p_bar)


def null_explainer(black_box_model, query_point, test_data, samples=10000):
    '''
    ConstantSurrogate fitted on the neighbourhood bLIMEy would have trained on.

    The salt is bLIMEy's own rather than a new one: this is deliberately the SAME draw
    site - the surrogate's training sample - so that the null explainer and the fitted
    surrogates see the same points.  (A new draw site would need its own salt; see
    clime/utils/seeding.py.)
    '''
    q = np.asarray(query_point, dtype=np.float64).ravel()
    cov = np.cov(test_data['X'].T)
    rng = clime.utils.rng_from_point(q, salt='surrogate training sample')
    X = rng.multivariate_normal(q, cov, samples)
    p = np.asarray(black_box_model.predict_proba(X), dtype=np.float64)[:, 1]
    w = costs.weights_based_on_distance(q, X)
    return ConstantSurrogate(q, float(np.sum(p*w)/np.sum(w)))


# ------------------------------------------------------- the synthetic surrogate family

def orthonormal_partner(v, seed=0):
    '''
    a unit vector orthogonal to v, spanning the plane the rotation happens in.

    In two dimensions the plane is the whole space and the partner is determined up to
    sign, so it is taken canonically and the figure is reproducible without a seed.  In
    higher dimensions the plane is a choice; it is drawn from a seeded generator and the
    seed is reported, since a different plane is a different (equally valid) slice.
    '''
    v = np.asarray(v, dtype=np.float64).ravel()
    u = v/np.linalg.norm(v)
    if u.size == 2:
        return np.array([-u[1], u[0]])
    rng = np.random.default_rng(seed)
    for _ in range(16):
        w = rng.standard_normal(u.size)
        w -= (w @ u)*u
        n = np.linalg.norm(w)
        if n > 1e-8:
            return w/n
    raise RuntimeError('could not find a direction orthogonal to the truth')


def rotate(v, theta, partner=None, seed=0):
    '''rotate v by theta within the plane it spans with `partner`, preserving its norm'''
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    u = v/n
    w = orthonormal_partner(v, seed=seed) if partner is None else np.asarray(partner)
    return n*(np.cos(theta)*u + np.sin(theta)*w)


def family_member(query_point, intercept, truth, theta=0.0, scale=1.0, sharpen=1.0,
                  partner=None, seed=0):
    '''
    one member of the family:

        logit g(x) = sharpen * ( logit f(q) + scale * R(theta) grad logit f(q) . (x - q) )

    cos(theta) is exactly the cosine similarity of the explanation to the truth and
    scale*sharpen exactly the ratio of the norms, so both are error by construction.

    `scale` and `sharpen` both multiply the coefficient vector, but they are not the same
    probe, and the difference is the whole point of the figure:

      scale    multiplies the slope alone, leaving g(q) = f(q).  The surrogate's class
               boundary {g = 1/2} is the set b.(x-q) = -logit f(q)/s, which MOVES as s
               changes - so a threshold instrument responds to this, but only by watching
               the boundary slide past, not by seeing the slope.
      sharpen  multiplies the whole log-odds, so the boundary {g = 1/2} is fixed and only
               the steepness of the sigmoid across it changes.  This is the axis on which
               a surrogate is over- or under-confident, and it is the axis a threshold at
               0.5 is exactly blind to.
    '''
    b = scale*sharpen*rotate(truth, theta, partner=partner, seed=seed)
    return LinearLogitSurrogate(query_point, intercept=sharpen*intercept, coef=b)
