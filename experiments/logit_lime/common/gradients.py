'''
Analytic gradients of the black box's log-odds, for use as explanation ground truth.

The coefficient ground truth of sweep_ground_truth.py only exists for black boxes whose
log-odds are exactly linear, where coef_ IS the local importance vector everywhere. The
generalisation is the local one: for any differentiable black box the vector a local
linear surrogate should recover at q is

    g(q) = d/dx logit f(x) |_{x=q}

For the exactly linear families g(q) = coef_ at every q, so this is a strict extension of
the existing instrument rather than a different one - the sweep checks exactly that.

Both surrogates aim at the same target. Standard LIME regresses probabilities, so its
target is the gradient of p rather than of logit p, but

    grad p = p(1-p) grad logit p

and p(1-p) > 0, so the two differ by a positive scalar and point the same way. Cosine
similarity and rank measures are invariant to that factor, so scoring both surrogates
against g gives neither an advantage from the choice of space.

Finite differences will not do here. logit f is computed from clipped probabilities, so
wherever the black box saturates - which is most of a neighbourhood for QDA or naive
Bayes on real data - the difference quotient collapses to zero and the point is lost.
Every gradient below is instead derived from the fitted parameters, in closed form,
which is defined whether or not the probability underflows.

There is deliberately no entry for the piecewise constant families (tree, forest, kNN,
gradient boosting, the calibrated forests). Their gradient is zero almost everywhere and
undefined on the splits: no local linear ground truth exists, which is a result rather
than a gap. `has_gradient` reports this.

usage:
    from common import gradients
    g = gradients.grad_logit(clf, 'QDA', X)        # (n, d)
    gradients.has_gradient('Decision Tree')        # False
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np


def unwrap(clf, attr):
    '''
    the pipeline wraps the estimator twice - the clime model wrapper, then the model
    balancer - and the constructed black boxes add an sklearn Pipeline on top, so the
    attribute has to be looked for down the chain rather than on the object handed in
    '''
    seen = 0
    while clf is not None and seen < 6:
        if hasattr(clf, attr):
            return clf
        clf = getattr(clf, 'model', None)
        seen += 1
    raise AttributeError(f'no {attr} found by unwrapping .model')


def _pipeline_steps(clf):
    '''the (name, estimator) steps of the sklearn Pipeline a constructed model wraps'''
    return dict(unwrap(clf, 'steps').steps)


def _as_2d(X):
    X = np.asarray(X, dtype=np.float64)
    return X[None, :] if X.ndim == 1 else X


# ---------------------------------------------------------------- linear log-odds

def _linear_coef(clf):
    '''class 1 weights of a model whose decision_function is the log-odds'''
    return np.atleast_2d(unwrap(clf, 'coef_').coef_)[-1, :]


def grad_linear(clf, X):
    '''logistic regression, LDA: log-odds are w.x + b, so the gradient is w everywhere'''
    X = _as_2d(X)
    return np.tile(_linear_coef(clf), (X.shape[0], 1))


def grad_nearest_class_mean(clf, X):
    '''
    softmax over -||x - m_c||^2, so logit p1 = 2 x.(m1 - m0) + const and the gradient is
    the constant vector 2(m1 - m0)
    '''
    X = _as_2d(X)
    means = unwrap(clf, 'means_').means_
    return np.tile(2.0*(means[1] - means[0]), (X.shape[0], 1))


# ------------------------------------------------------------- quadratic log-odds

def _grad_gaussian_pair(X, mean0, prec0, mean1, prec1):
    '''
    gradient of the log-likelihood ratio of two Gaussians:
      logit p1 = -0.5 (x-m1)' P1 (x-m1) + 0.5 (x-m0)' P0 (x-m0) + const
    with P the precision (inverse covariance). The constants carry the determinants and
    the priors and drop out under differentiation.
    '''
    return -(X - mean1) @ prec1 + (X - mean0) @ prec0


def grad_qda(clf, X):
    '''
    QDA. sklearn stores the per class covariance by its eigendecomposition:
    rotations_[k] holds the eigenvectors and scalings_[k] the eigenvalues (already
    carrying reg_param if the escalating fallback had to regularise), so the precision is
    R diag(1/s) R'.
    '''
    X = _as_2d(X)
    inner = unwrap(clf, 'rotations_')
    prec = [r @ np.diag(1.0/np.asarray(s)) @ r.T
            for r, s in zip(inner.rotations_, inner.scalings_)]
    return _grad_gaussian_pair(X, inner.means_[0], prec[0], inner.means_[1], prec[1])


def grad_gaussian_nb(clf, X):
    '''Gaussian naive Bayes - the same quadratic form with diagonal covariance'''
    X = _as_2d(X)
    inner = unwrap(clf, 'theta_')
    m, v = np.asarray(inner.theta_), np.asarray(inner.var_)
    return -(X - m[1])/v[1] + (X - m[0])/v[0]


def grad_bayes_optimal(clf, X):
    '''
    the Gaussian class conditional model, which normalises class densities directly - so
    the log-odds are the log density ratio and the gradient is the same quadratic form.
    Its parameters are the generative ones when the loader supplies them, which makes
    this the only black box here whose ground truth is not estimated from a fit.
    '''
    X = _as_2d(X)
    inner = unwrap(clf, 'means')
    prec = [np.linalg.inv(np.asarray(c)) for c in inner.covs]
    return _grad_gaussian_pair(X, np.asarray(inner.means[0]), prec[0],
                               np.asarray(inner.means[1]), prec[1])


# -------------------------------------------------------- log-odds on a feature map

def grad_polynomial_logistic(clf, X):
    '''
    logistic regression on degree 2 features. logit p1 = w.phi(x) + b with
    phi_k(x) = prod_j x_j ** e_kj, so

        d phi_k / d x_j = e_kj * prod_l x_l ** (e_kl - [l == j])

    which is exact including where a feature is zero (the term simply drops out unless
    its exponent is 1).
    '''
    X = _as_2d(X)
    steps = _pipeline_steps(clf)
    powers = steps['poly'].powers_                      # (n_terms, d)
    w = np.atleast_2d(steps['clf'].coef_)[-1, :]        # (n_terms,)
    n, d = X.shape
    out = np.zeros((n, d))
    for j in range(d):
        e = powers[:, j]
        active = e > 0
        if not np.any(active):
            continue
        reduced = powers[active].copy()
        reduced[:, j] -= 1
        # prod_l x_l ** reduced_kl, evaluated for every sample
        terms = np.prod(X[:, None, :]**reduced[None, :, :], axis=2)   # (n, n_active)
        out[:, j] = terms @ (w[active]*e[active])
    return out


def grad_rbf_logistic(clf, X):
    '''
    logistic regression on a Nystroem RBF map. transform is
    phi(x) = k(x, C) @ N', so with v = N' w,

        grad logit p1 = -2 gamma * sum_c v_c k(x, c_c) (x - c_c)
    '''
    X = _as_2d(X)
    steps = _pipeline_steps(clf)
    nys = steps['rbf']
    w = np.atleast_2d(steps['clf'].coef_)[-1, :]
    C = np.asarray(nys.components_, dtype=np.float64)
    gamma = nys.gamma if nys.gamma is not None else 1.0/X.shape[1]
    v = nys.normalization_.T @ w                        # weight per component
    sq = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
    k = np.exp(-gamma*sq)                               # (n, n_components)
    a = k*v[None, :]
    # sum_c a_c (x - c_c), for every sample
    return -2.0*gamma*(a.sum(axis=1)[:, None]*X - a @ C)


def grad_bagged_logistic(clf, X):
    '''
    bagging averages probabilities, so p = mean_i sigma(z_i) and

        grad logit p = grad p / (p (1-p)),   grad p = mean_i sigma'(z_i) w_i

    which is exactly why the ensemble's log-odds are not linear although every member's
    are. Members may have been fitted on a subset of features (max_features), so each
    contribution is scattered back to the columns that member saw.
    '''
    X = _as_2d(X)
    inner = unwrap(clf, 'estimators_')
    n, d = X.shape
    grad_p = np.zeros((n, d))
    p = np.zeros(n)
    features = getattr(inner, 'estimators_features_', None)
    for i, est in enumerate(inner.estimators_):
        cols = np.arange(d) if features is None else np.asarray(features[i])
        w = np.atleast_2d(est.coef_)[-1, :]
        z = X[:, cols] @ w + float(np.ravel(est.intercept_)[-1])
        pi = 1.0/(1.0 + np.exp(-z))
        p += pi
        grad_p[:, cols] += (pi*(1 - pi))[:, None]*w[None, :]
    m = len(inner.estimators_)
    p /= m
    grad_p /= m
    denom = np.clip(p*(1 - p), 1e-12, None)
    return grad_p/denom[:, None]


# ------------------------------------------------------------------ neural network

_ACTIVATION_GRAD = {
    'relu': lambda z, a: (z > 0).astype(np.float64),
    'tanh': lambda z, a: 1.0 - a**2,
    'logistic': lambda z, a: a*(1.0 - a),
    'identity': lambda z, a: np.ones_like(z),
}


def grad_mlp(clf, X):
    '''
    a binary MLPClassifier ends in a logistic output unit, so the final pre-activation
    IS the log-odds and the gradient is a plain backward pass through the fitted weights.
    '''
    X = _as_2d(X)
    inner = unwrap(clf, 'coefs_')
    act = _ACTIVATION_GRAD[inner.activation]
    # forward, keeping what the backward pass needs
    a = X
    cache = []
    for W, b in zip(inner.coefs_[:-1], inner.intercepts_[:-1]):
        z = a @ W + b
        a = _forward(inner.activation, z)
        cache.append((W, z, a))
    # backward from the single output unit
    g = np.tile(np.asarray(inner.coefs_[-1])[:, -1], (X.shape[0], 1))
    for W, z, a_next in reversed(cache):
        g = g*act(z, a_next)
        g = g @ W.T
    return g


def _forward(name, z):
    if name == 'relu':
        return np.maximum(z, 0.0)
    if name == 'tanh':
        return np.tanh(z)
    if name == 'logistic':
        return 1.0/(1.0 + np.exp(-z))
    return z


# -------------------------------------------------------------------------- kernel SVM

def grad_svm(clf, X):
    '''
    SVC with probability=True: Platt scaling maps the decision function to a probability
    as p1 = 1/(1 + exp(A f(x) + B)), so logit p1 = -(A f(x) + B) and only f has to be
    differentiated. For the RBF kernel

        grad f = sum_i dual_i * (-2 gamma) (x - sv_i) exp(-gamma ||x - sv_i||^2)

    The sign convention of A is libsvm's and is checked numerically by the validation
    below rather than trusted.
    '''
    X = _as_2d(X)
    inner = unwrap(clf, 'support_vectors_')
    if inner.kernel != 'rbf':
        raise NotImplementedError(f'no analytic gradient for a {inner.kernel} SVM kernel')
    sv = np.asarray(inner.support_vectors_, dtype=np.float64)
    dual = np.asarray(inner.dual_coef_)[0]
    gamma = inner._gamma if hasattr(inner, '_gamma') else inner.gamma
    sq = ((X[:, None, :] - sv[None, :, :])**2).sum(axis=2)
    k = np.exp(-gamma*sq)*dual[None, :]
    grad_f = -2.0*gamma*(k.sum(axis=1)[:, None]*X - k @ sv)
    A = float(np.ravel(inner.probA_)[0])
    return -A*grad_f


# ---------------------------------------------------------------------- the registry

ANALYTIC_GRADIENTS = {
    # exactly linear log-odds: the gradient is constant and equals coef_
    'Logistic': grad_linear,
    'LDA': grad_linear,
    'Nearest Class Mean': grad_nearest_class_mean,
    # quadratic log-odds
    'QDA': grad_qda,
    'Gaussian Naive Bayes': grad_gaussian_nb,
    'Bayes Optimal': grad_bayes_optimal,
    'Polynomial Logistic (deg 2)': grad_polynomial_logistic,
    # smooth, non-polynomial
    'MLP': grad_mlp,
    'SVM': grad_svm,
    'RBF Logistic (Nystroem)': grad_rbf_logistic,
    # neither, and the point of including it
    'Bagged Logistic': grad_bagged_logistic,
}

# black boxes with no local gradient anywhere - not an omission
NO_GRADIENT = ['Decision Tree', 'Random Forest', 'k Nearest Neighbours',
               'Gradient Boosting', 'Random Forest (Platt calibrated)',
               'Random Forest (isotonic calibrated)']


def has_gradient(model_name):
    return model_name in ANALYTIC_GRADIENTS


def grad_logit(clf, model_name, X):
    '''gradient of logit f at each row of X, in the model's own (standardised) space'''
    return ANALYTIC_GRADIENTS[model_name](clf, X)


# ------------------------------------------------------------------------ validation

def finite_difference(clf, X, h=1e-3, eps=1e-12):
    '''
    central difference reference. Only meaningful where the black box is not saturated -
    logit of a clipped probability is flat - which is why it is a check and not the
    instrument.
    '''
    X = _as_2d(X)

    def logit_p(Z):
        p = np.clip(np.asarray(clf.predict_proba(Z))[:, 1], eps, 1 - eps)
        return np.log(p/(1 - p))

    n, d = X.shape
    out = np.zeros((n, d))
    step = h*np.eye(d)
    for i in range(n):
        out[i] = (logit_p(X[i] + step) - logit_p(X[i] - step))/(2*h)
    return out


def saturated(clf, X, tol=1e-9):
    '''where the finite difference cannot say anything, so a disagreement is expected'''
    p = np.asarray(clf.predict_proba(_as_2d(X)))[:, 1]
    return (p < tol) | (p > 1 - tol)
