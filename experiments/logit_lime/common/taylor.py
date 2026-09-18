'''
The explanation-targeted surrogate: a first-order Taylor expansion of the log-odds at q.

Every surrogate elsewhere in this study is fitted - it minimises a locality-weighted error
over a sampled neighbourhood, and its coefficients are whatever that fit produces. This
one goes the other way. It sets its coefficients to the black box's local behaviour and
accepts whatever fidelity follows:

    logit g(x) = logit f(q) + grad_logit f(q) . (x - q)

which is the surrogate whose explanation is correct by construction. Its purpose is to
measure the price of that, and so to test the account offered for why logit space improves
the explanation of a group B or C black box without improving its fidelity: that a fitted
surrogate trades slope for intercept to repair its probabilities over the neighbourhood,
and the slope is the explanation. See PREREGISTRATION.md, second registration.

Two versions, and the difference between them matters:

  AnalyticTaylor          takes the gradient from common/gradients.py. This needs the
                          black box's parameters, so it is an oracle, not a method. Its
                          cosine to the ground truth is 1 BY CONSTRUCTION - it is the same
                          vector - and must never be reported as a finding.

  FiniteDifferenceTaylor  estimates the gradient from 2d black-box queries. This needs
                          exactly what LIME needs: the ability to call predict_proba. It
                          is a real explainer, its explanation accuracy is a real
                          measurement, and it costs 2d queries against LIME's 10,000.

Both expose the explainer contract - predict, predict_proba, get_explanation - so the
existing metrics score them unchanged.  Both are the special case b = grad logit f(q) of
common/surrogates.py's LinearLogitSurrogate, which is where that contract is implemented.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np

from common import gradients
from common.surrogates import LinearLogitSurrogate

EPS = 1e-12
# escalating step sizes: a difference quotient of logit f underflows to zero wherever the
# black box saturates, and a wider step can still straddle the transition. Same shape of
# fallback as QDA's regularisation - climb until it works rather than guess one value.
FD_STEPS = (1e-3, 1e-2, 1e-1)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p/(1 - p))


class _TaylorBase(LinearLogitSurrogate):
    '''the expansion at q: the intercept is read off f, the slope from _gradient'''

    def __init__(self, black_box_model, query_point, **kwargs):
        self.clf = black_box_model
        q = np.asarray(query_point, dtype=np.float64).ravel()
        self.q = q
        intercept = float(_logit(
            np.asarray(self.clf.predict_proba(q[None, :]))[0, 1]))
        super().__init__(q, intercept=intercept, coef=self._gradient())
        if self.degenerate:
            self.coef = np.zeros_like(self.q)

    def _gradient(self):
        raise NotImplementedError


class AnalyticTaylor(_TaylorBase):
    '''oracle: the gradient in closed form from the fitted black box's parameters'''

    def __init__(self, black_box_model, query_point, model_name, **kwargs):
        self.model_name = model_name
        super().__init__(black_box_model, query_point, **kwargs)

    def _gradient(self):
        return np.asarray(
            gradients.grad_logit(self.clf, self.model_name, self.q[None, :])[0],
            dtype=np.float64)


class FiniteDifferenceTaylor(_TaylorBase):
    '''
    model-agnostic: central differences of logit f, using only predict_proba.

    Records which step size was needed, and whether every step underflowed - which is what
    happens in a neighbourhood the black box treats as entirely one class, and is a real
    failure mode of the method rather than of the implementation.
    '''

    def __init__(self, black_box_model, query_point, steps=FD_STEPS, **kwargs):
        self.steps = steps
        self.step_used = np.nan
        self.n_queries = 0
        super().__init__(black_box_model, query_point, **kwargs)

    def _gradient(self):
        d = self.q.size
        eye = np.eye(d)
        for h in self.steps:
            plus = _logit(np.asarray(
                self.clf.predict_proba(self.q[None, :] + h*eye))[:, 1])
            minus = _logit(np.asarray(
                self.clf.predict_proba(self.q[None, :] - h*eye))[:, 1])
            self.n_queries += 2*d
            g = (plus - minus)/(2*h)
            if np.any(g != 0) and np.all(np.isfinite(g)):
                self.step_used = h
                return g
        self.step_used = np.nan
        return np.zeros(d)
