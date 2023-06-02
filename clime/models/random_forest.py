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
    def __init__(self, data, balanced_training=False, fit_intercept=True, **kwargs):
        self.balanced_training = balanced_training
        super().__init__(max_depth=3, random_state=clime.RANDOM_SEED, **kwargs)
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
    wrapper to call balanced training version of ridge
    '''
    return random_forest(data, class_weight='balanced', **kwargs)
