'''gradient of a black box's log-odds, for the model families that have one'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from clime.models.projection import projection_model


def _logistic(model, X):
    return np.tile(np.ravel(model.coef_), (len(X), 1))


def _svm(model, X):
    '''Platt scaling gives logit P(1) = -(A f(x) + B), f the RBF decision function'''
    sv = np.asarray(model.support_vectors_, dtype=np.float64)
    k = np.exp(-model._gamma*((X[:, None, :] - sv[None, :, :])**2).sum(axis=2))
    k = k*np.asarray(model.dual_coef_)[0][None, :]
    grad_f = -2.0*model._gamma*(k.sum(axis=1)[:, None]*X - k @ sv)
    return -float(np.ravel(model.probA_)[0])*grad_f


def _mlp(model, X):
    import torch
    model.net.eval()
    x = torch.tensor(X, dtype=torch.float32, device=model.device, requires_grad=True)
    model.net.get_projection(x).sum().backward()
    return x.grad.cpu().numpy().astype(np.float64)


SMOOTH = {'Logistic': _logistic, 'SVM': _svm, 'MLP': _mlp}


def log_odds(box, X):
    '''(n, d) gradient of logit P(class 1) at each row of X; None for a piecewise-constant family'''
    while not isinstance(box, projection_model):
        box = box.model   # a threshold move or a cache changes no gradient
    if box.family not in SMOOTH:
        return None
    return SMOOTH[box.family](box.model, np.atleast_2d(np.asarray(X, dtype=np.float64)))
