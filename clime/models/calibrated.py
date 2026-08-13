'''
calibrated and boosted classifiers

these exist to separate two candidate explanations for when a logit space surrogate
helps: whether it is about the black box's probabilities being *saturated* (which
calibration removes) or about its log-odds being *linear in x* (which calibration does
not create, since Platt scaling composes a sigmoid with the same piecewise constant
score function)
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn.ensemble
from sklearn.calibration import CalibratedClassifierCV
import clime
from clime.data.utils import costs
from clime.models import base_model


class _calibrated(base_model):
    '''
    wrapper around a calibrated classifier - CalibratedClassifierCV is not itself
    constructed from a data dict, so cannot go in the registry directly
    '''
    base_estimator = None

    def __init__(self, data, method='sigmoid', cv=5, balanced_training=False, **kwargs):
        self.balanced_training = balanced_training
        estimator = self.base_estimator(**kwargs)
        cv = self._usable_cv(data['y'], cv)
        if cv is None:
            # not enough examples in some class to hold out a fold - calibration is not
            # possible, so fall back to the uncalibrated estimator rather than raising
            # (the pipeline tests run with as little as one example per class)
            self.model = estimator
        else:
            self.model = CalibratedClassifierCV(estimator, method=method, cv=cv)
        self.train(data)

    @staticmethod
    def _usable_cv(y, cv):
        '''largest usable number of folds, or None if the data is too small to calibrate'''
        counts = np.bincount(np.array(y).astype(np.int64))
        smallest_class = counts[counts > 0].min()
        if smallest_class < 2:
            return None
        return int(min(cv, smallest_class))

    def train(self, data):
        if self.balanced_training is True:
            instance_weights = costs.get_instance_class_weights(data)
        else:
            instance_weights = None
        self.model.fit(data['X'], data['y'], sample_weight=instance_weights)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class calibrated_random_forest(_calibrated):
    @staticmethod
    def base_estimator(**kwargs):
        return sklearn.ensemble.RandomForestClassifier(
            max_depth=None, random_state=clime.RANDOM_SEED, **kwargs)


def random_forest_platt(data, **kwargs):
    '''random forest with Platt (sigmoid) calibration'''
    return calibrated_random_forest(data, method='sigmoid', **kwargs)


def random_forest_isotonic(data, **kwargs):
    '''random forest with isotonic calibration'''
    return calibrated_random_forest(data, method='isotonic', **kwargs)


class gradient_boosting(sklearn.ensemble.GradientBoostingClassifier, base_model):
    '''
    gradient boosting builds its additive model in log-odds space and applies a
    sigmoid, so its logit is a sum of many small trees - a useful intermediate
    between a random forest's vote fraction and a genuinely linear log-odds surface
    '''
    def __init__(self, data, balanced_training=False, **kwargs):
        self.data = data      # sklearn contract: see FINDINGS.md B17
        self.balanced_training = balanced_training
        super().__init__(random_state=clime.RANDOM_SEED, **kwargs)
        self.train(data)

    def train(self, data):
        if self.balanced_training is True:
            instance_weights = costs.get_instance_class_weights(data)
        else:
            instance_weights = None
        self.fit(data['X'], data['y'], sample_weight=instance_weights)
