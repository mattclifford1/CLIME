'''
Random forest classifier (logistic regression)
Inherits from sckit-learn classifiers
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn
import numpy as np
import clime
from clime.data.utils import costs
from clime.models import base_model


class random_forest(sklearn.ensemble.RandomForestClassifier, base_model):
    '''
    train linear reg classifier on dataset - sub class of sklearn.ensemble.RandomForestClassifier
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    # N.B. `fit_intercept` used to be accepted here, copy-pasted from the logistic model.
    # A forest has no intercept, so it was silently ignored - and being in the signature
    # without being stored broke sklearn's get_params() (FINDINGS.md B17). Nothing passed
    # it, so it is simply gone.
    def __init__(self, data, balanced_training=False, **kwargs):
        # sklearn's estimator contract requires every __init__ parameter to be stored
        # unmodified under the same name: get_params() reads the signature and does
        # getattr(self, name) for each. `data` was consumed by train() and never stored,
        # which raised AttributeError once sklearn 1.2 began calling _validate_params()
        # from fit(). FINDINGS.md B17.
        self.data = data
        self.balanced_training = balanced_training
        super().__init__(max_depth=None, random_state=clime.RANDOM_SEED, **kwargs)
        self.train(data)

    def train(self, data):
        if self.balanced_training is True:
            # get class imbalance weights
            instance_weights = costs.get_instance_class_weights(data)
        else:
            instance_weights = None
        self.fit(data['X'], data['y'], sample_weight=instance_weights)


def random_forest_balanced_training(data, **kwargs):
    '''
    wrapper to call balanced training version of random forest
    '''
    return random_forest(data, balanced_training=True, **kwargs)
