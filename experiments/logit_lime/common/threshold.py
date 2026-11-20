'''A black box at another decision threshold: the same model with a constant added to its log-odds.'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from scipy.special import expit, logit


def shift_for(threshold):
    '''the log-odds shift that makes `p > threshold` the same decision as `shifted p > 0.5`'''
    return -float(logit(threshold))


class Shifted:
    def __init__(self, clf, shift):
        self.clf, self.shift = clf, float(shift)

    def log_odds(self, X):
        p = np.asarray(self.clf.predict_proba(X), dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.log(p[:, 1]) - np.log(p[:, 0]) + self.shift
        return np.where(p.sum(axis=1) == 0, -np.inf, z)   # (0, 0) is read as p = 0

    def predict_proba(self, X):
        z = self.log_odds(X)
        return np.column_stack([expit(-z), expit(z)])

    def predict(self, X):
        return (self.log_odds(X) > 0).astype(int)


def at_threshold(clf, threshold):
    return Shifted(clf, shift_for(threshold))
