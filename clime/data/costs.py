'''
utils for getting weightings/costs for data based on distance and class imbalance
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import warnings
import numpy as np

def weights_based_on_distance(query_point, X):
    '''
    get the weighting of each sample proportional to the distance to the query point
    weights generated using exponential kernel found in the original lime implementation
    '''
    kernel_width = np.sqrt(X.shape[1]) * .75
    euclidean_dist = np.sqrt(np.sum((X - query_point)**2, axis=1))
    weights = np.sqrt(np.exp(-(euclidean_dist ** 2) / kernel_width ** 2))
    return weights

def weight_based_on_class_imbalance(data):
    '''
    get the weight of each class proportional to the number of instances
    '''
    # make sure we have class data (sampled data from LIME won't have this)
    if 'y' not in data.keys():
        warnings.warn("No class data 'y': not use class balanced weightings", Warning)
        return np.array([1, 1])
    y = np.array(data['y']).astype(np.int64)
    n_classes = len(list(np.unique(y)))
    # if we only have one class then we can't say anything about the weighting
    if n_classes == 1:
        return np.array([1, 1])
    # otherwise get the weightings
    bin_count = np.bincount(y)
    weights = 1 / ((n_classes * bin_count) / n_classes)

    weights /= min(weights)
    # weights /= sum(weights)
    return weights

def get_instance_class_weights(data):
    '''
    get cost based on classs imbalance but in a matrix form
    useful when weighting training examples during model training
    '''
    class_weights = weight_based_on_class_imbalance(data)
    # get class labels as a matrix
    y = np.expand_dims(data['y'], axis=1)
    Y = np.concatenate((y, np.abs(1-y)), axis=1)
    # apply to all instances
    instance_weights = np.dot(Y, class_weights.T)
    return instance_weights
