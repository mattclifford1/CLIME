'''an image explained in patches: surrogates and ground truths from the black box's logit on every mask'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from math import comb

import numpy as np
import shap
from scipy.special import expit

PATCHES = 16
FULL = 2**PATCHES - 1
POWERS = 1 << np.arange(PATCHES)
KERNEL_WIDTH = 0.25   # lime_image's, on the cosine distance to the full image
ALPHA = 1.0
# target of the fit, and whether CLIME's class weights are applied
EXPLAINERS = {'LIME': ('probability', False), 'CLIME': ('probability', True),
              'Logit-LIME': ('logit', False), 'Logit-CLIME': ('logit', True),
              'Soft logistic': ('soft', False), 'Soft logistic CLIME': ('soft', True)}
SHAP = {'KernelSHAP': 'probability', 'Logit-KernelSHAP': 'logit'}   # target of each


def masks():
    '''(2**16, 16) every mask; bit j of the row index is patch j'''
    index = np.arange(FULL + 1)
    return ((index[:, None] >> np.arange(PATCHES)) & 1).astype(np.float64)


def kernel(M):
    distance = 1 - np.sqrt(M.sum(axis=1)/PATCHES)
    return np.sqrt(np.exp(-distance**2/KERNEL_WIDTH**2))


def leave_one_out(z):
    '''logit of the full image minus its logit with each patch hidden'''
    return z[FULL] - z[FULL ^ (1 << np.arange(PATCHES))]


def shapley(z):
    '''exact Shapley value of each patch for the logit'''
    index = np.arange(FULL + 1)
    size = ((index[:, None] >> np.arange(PATCHES)) & 1).sum(axis=1)
    share = np.array([1/(PATCHES*comb(PATCHES - 1, k)) for k in range(PATCHES)])
    out = np.empty(PATCHES)
    for j in range(PATCHES):
        without = index[(index >> j) & 1 == 0]
        out[j] = (share[size[without]]*(z[without | (1 << j)] - z[without])).sum()
    return out


def class_weights(f_class):
    '''CLIME's weight for each row: the larger class's count over its own class's'''
    counts = np.bincount(f_class, minlength=2)
    if counts.min() == 0:
        return np.ones(len(f_class))
    return (counts.max()/counts)[f_class]


def ridge(X, y, w):
    mean_x, mean_y = np.average(X, axis=0, weights=w), np.average(y, weights=w)
    centred = X - mean_x
    A = (centred*w[:, None]).T @ centred + ALPHA*np.eye(X.shape[1])
    coef = np.linalg.solve(A, (centred*w[:, None]).T @ (y - mean_y))
    return coef, float(mean_y - coef @ mean_x)


def soft_logistic(X, p, w, iterations=200, tolerance=1e-10):
    '''cross-entropy fit to the probabilities p with the ridge's penalty; zeros if p is one certain class'''
    if not ((w*p).sum() > 0 and (w*(1 - p)).sum() > 0):
        return np.zeros(X.shape[1]), float(np.inf if p.mean() >= 0.5 else -np.inf)
    design = np.column_stack([np.ones(len(X)), X])
    penalty = 2*ALPHA*np.diag(np.r_[0.0, np.ones(X.shape[1])])

    def loss(theta):
        eta = design @ theta
        return (w*(np.logaddexp(0, eta) - p*eta)).sum() + ALPHA*(theta[1:]**2).sum()

    theta = np.zeros(design.shape[1])
    theta[0] = np.log((w*p).sum()/(w*(1 - p)).sum())
    value = loss(theta)
    for _ in range(iterations):
        g = expit(design @ theta)
        gradient = design.T @ (w*(g - p)) + penalty @ theta
        hessian = (design*(w*g*(1 - g))[:, None]).T @ design + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        while scale > 1e-8 and loss(theta - scale*step) > value:
            scale /= 2
        theta = theta - scale*step
        new = loss(theta)
        if value - new < tolerance*max(1.0, abs(value)):
            break
        value = new
    return theta[1:], float(theta[0])


def fit_shap(name, z, n, seed):
    '''(coefficients, intercept) of one of SHAP from shap's KernelExplainer on n runs of the black box'''
    y = z if SHAP[name] == 'logit' else expit(z)
    np.random.seed(seed)   # the package draws its masks from numpy's global state
    explainer = shap.KernelExplainer(lambda X: y[X.astype(np.int64) @ POWERS], np.zeros((1, PATCHES)))
    # two of the n runs are the empty and the full image; l1_reg=False keeps all 16 patches
    coef = explainer.shap_values(np.ones(PATCHES), nsamples=n - 2, l1_reg=False, silent=True)
    return np.asarray(coef, dtype=np.float64), float(y[0])


def exact_shap(name, z):
    '''what fit_shap converges to: the Shapley values of its target'''
    y = z if SHAP[name] == 'logit' else expit(z)
    return shapley(y), float(y[0])


def fit(name, X, z, w):
    '''(coefficients, intercept) of one of EXPLAINERS on masks X with black-box logits z'''
    target, balanced = EXPLAINERS[name]
    if balanced:
        w = w*class_weights((z > 0).astype(np.int64))
    if target == 'logit':
        return ridge(X, z, w)
    if target == 'soft':
        return soft_logistic(X, expit(z), w)
    return ridge(X, expit(z), w)


def predict(name, coef, intercept, X):
    '''the surrogate's estimate of P(class 1)'''
    out = intercept + X @ coef
    target = SHAP[name] if name in SHAP else EXPLAINERS[name][0]
    return out if target == 'probability' else expit(out)
