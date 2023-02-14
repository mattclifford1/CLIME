'''
Linear classifiers (logistic regression)
Inherits from sckit-learn classifiers
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import sklearn
import numpy as np
from clime.data import costs
from clime.models import base_model


class logistic(sklearn.linear_model.LogisticRegression, base_model):
    '''
    train linear reg classifier on dataset - sub class of sklearn.linear_model.Ridge
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data, balanced_training=False, fit_intercept=True, **kwargs):
        self.balanced_training = balanced_training
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        self.train(data)

    def train(self, data):
        if self.balanced_training is True:
            # get class imbalance weights
            class_weights = costs.weight_based_on_class_imbalance(data)
            # get class labels as a matrix
            y = np.expand_dims(data['y'], axis=1)
            Y = np.concatenate((y, np.abs(1-y)), axis=1)
            class_matrix = data['y']
            # apply to all instances
            instance_weights = np.dot(Y, class_weights.T)
        else:
            instance_weights = None
        self.fit(data['X'], data['y'], sample_weight=instance_weights)


def logistic_balanced_training(data, **kwargs):
    '''
    wrapper to call balanced training version of ridge
    '''
    return logistic(data, balanced_training=True, **kwargs)
