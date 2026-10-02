'''
logistic regression fitted to the black box's probabilities, not to its classes

The two logit-space surrogates already here fit the same model class as this one - a
sigmoid of a linear function, read as log-odds per unit feature - and differ only in how:

    logit_ridge            least squares on logit p, after rescaling p into
                           [1e-9, 1 - 1e-8] so that saturated probabilities have a log-odds
    logistic_regression    cross-entropy against np.round(p): the black box's classes, so
                           its confidence is thrown away before the fit
    soft_logistic_regression (this)
                           cross-entropy against p itself:
                               min  - sum_i w_i [ p_i log g(x_i) + (1 - p_i) log(1 - g(x_i)) ]
                                    + ||beta||^2 / (2C)
                           i.e. the locality-weighted KL projection of the black box onto the
                           model class. No clipping or rescaling: a saturated p is a
                           confident label, not a log-odds of +-20.

sklearn's LogisticRegression takes class labels only, so each sampled point enters twice,
once as class 1 with weight w*p and once as class 0 with weight w*(1-p). The weighted
log-likelihood of those 2n rows is exactly the soft cross-entropy above. Rows whose weight
is zero are dropped, which halves the work wherever the black box saturates.

C = 0.5 puts the same penalty on ||beta||^2 as Ridge(alpha=1) does in the other surrogates
(sklearn minimises C*sum(loss) + ||beta||^2/2), against a data term on a different scale -
a nominal match, not an exact one. The intercept is not penalised, as in Ridge.
tol = 1e-8 rather than sklearn's 1e-4: at 1e-4 lbfgs stops ~3e-4 short of the optimum
(checked against a direct minimiser in test_explanations.py), for a few extra
iterations, and a fit run to convergence does not depend on where an iteration cap
falls, which is what made the hard-label surrogate BLAS-thread-sensitive (README).

Composition rather than subclassing an sklearn estimator: subclasses that take arguments
sklearn does not know about are what broke under scikit-learn 1.2 (FINDINGS.md B17).
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn.linear_model


class soft_logistic_regression:
    def __init__(self, C=0.5, tol=1e-8, max_iter=10000, random_state=None):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.constant = None          # set when the neighbourhood has one class only

    def fit(self, X, y, sample_weight=None):
        '''
        y is the black box's probability for class 1, or its (n, 2) predict_proba output -
        the same target bLIMEy hands every surrogate
        '''
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = y[:, 1] if y.ndim == 2 else y
        w = (np.ones(len(p)) if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float64))
        w1, w0 = w*p, w*(1 - p)
        if not (w1.sum() > 0 and w0.sum() > 0):
            # the black box gives every weighted point the same certain class: no boundary
            # to fit and the likelihood has no finite optimum. Predict that class, with no
            # feature importances - the same fallback as logistic_regression (B12)
            self.constant = float(np.sum(w1)/np.sum(w))
            self.coef_ = np.zeros((1, X.shape[1]))
            self.intercept_ = np.array([np.inf if self.constant >= 0.5 else -np.inf])
            self.n_iter_ = np.array([0])
            return self
        self.constant = None
        XX = np.vstack([X, X])
        yy = np.r_[np.ones(len(p)), np.zeros(len(p))]
        ww = np.r_[w1, w0]
        keep = ww > 0
        self.model = sklearn.linear_model.LogisticRegression(
            C=self.C, tol=self.tol, max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(XX[keep], yy[keep], sample_weight=ww[keep])
        # class 1 is classes_[1], so coef_ is already the class 1 row (B15)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.n_iter_ = self.model.n_iter_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.constant is not None:
            p = np.full(X.shape[0], self.constant)
            return np.stack([1 - p, p], axis=1)
        return self.model.predict_proba(X)

    def predict(self, X):
        '''probabilities, as the other surrogates' predict returns - see bLIMEy.predict_proba'''
        return self.predict_proba(X)

    def decision_function(self, X):
        '''the surrogate's log-odds, the quantity its coefficients are a gradient of'''
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_[-1, :] + float(self.intercept_[-1])
