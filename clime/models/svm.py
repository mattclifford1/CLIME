'''
Support vector machine classifiers
Inherits from sckit-learn classifiers
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn.svm
import numpy as np
import clime
from clime.data.utils import costs
from clime.models import base_model


class SVM(sklearn.svm.SVC, base_model):
    '''
    train an SVM on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    *** to train with class balance weighting using the kwarg: class_weight='balanced'
    '''
    def __init__(self, data, gamma=2, C=1, probability=True, balanced_training=False, **kwargs):
        self.balanced_training = balanced_training
        super().__init__(gamma=gamma, C=C, probability=probability, random_state=clime.RANDOM_SEED, **kwargs)
        self.train(data)

    def train(self, data):
        if self.balanced_training is True:
            # get class imbalance weights
            instance_weights = costs.get_instance_class_weights(data)
        else:
            instance_weights = None
        self.fit(data['X'], data['y'], sample_weight=instance_weights)


def SVM_balanced_training(data, **kwargs):
    '''
    wrapper to call balanced training version of SVM
    '''
    return SVM(data, balanced_training=True, **kwargs)
