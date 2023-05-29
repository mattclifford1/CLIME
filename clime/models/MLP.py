'''
Multi layer perceptron
Inherits from sckit-learn classifiers
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from sklearn.neural_network import MLPClassifier
import numpy as np
import clime
from clime.models import base_model


class MLP_simple(MLPClassifier, base_model):
    '''
    train an SVM on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    *** to train with class balance weighting using the kwarg: class_weight='balanced'
    '''

    def __init__(self, data, **kwargs):
        super().__init__(random_state=clime.RANDOM_SEED, 
                         learning_rate='adaptive',
                         **kwargs)
        self.train(data)

    def train(self, data):
        self.fit(data['X'], data['y'])
