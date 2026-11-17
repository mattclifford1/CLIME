'''agreement between a surrogate and the black box on a weighted evaluation set'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _rate(mask, weight):
    total = weight.sum()
    return float((weight*mask).sum()/total) if total > 0 else np.nan


def ranking(score, f_class, weight):
    '''weighted AUC of the surrogate's score for the black box's class'''
    if weight[f_class == 0].sum() <= 0 or weight[f_class == 1].sum() <= 0:
        return np.nan
    return float(roc_auc_score(f_class, score, sample_weight=weight))


def brier(score, f_prob, weight):
    return _rate((np.clip(score, 0, 1) - f_prob)**2, weight)


def operating_point(score, cut, f_class, weight):
    '''
    fidelity: weighted share of points where the surrogate's class is the black box's
              (the local fidelity of CIKM'23 when `weight` is the locality kernel)
    fidelity_0, fidelity_1: the same within each black-box class
    balanced: their mean
    '''
    agree = (score >= cut).astype(np.int64) == f_class
    f0 = _rate(agree[f_class == 0], weight[f_class == 0])
    f1 = _rate(agree[f_class == 1], weight[f_class == 1])
    return {'fidelity': _rate(agree, weight), 'fidelity_0': f0, 'fidelity_1': f1,
            'balanced': (f0 + f1)/2}


def effective_share(weight):
    '''effective sample size of the weights as a share of the points'''
    return float(weight.sum()**2/(weight**2).sum()/len(weight))


def explanation_agreement(a, b):
    '''cosine, same top feature and rank correlation of two coefficient vectors'''
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan
    rank = np.nan
    if len(a) > 2 and np.ptp(np.abs(a)) > 0 and np.ptp(np.abs(b)) > 0:
        rank = float(spearmanr(np.abs(a), np.abs(b)).statistic)
    return {'cosine': cosine, 'same_top': bool(np.argmax(np.abs(a)) == np.argmax(np.abs(b))),
            'rank': rank, 'norm_ratio': float(nb/na) if na > 0 else np.nan}
