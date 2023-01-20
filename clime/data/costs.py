# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

'''
utils for getting weightings/costs for data based on distance and class imbalance
'''
import numpy as np


def weights_based_on_distance(query_point, X):
    '''
    get the weighting of each sample proportional to the distance to the query point
    weights generated using exponential kernel found in the original lime implementation'''
    euclidean_dist = np.sqrt(np.sum((X - query_point)**2, axis=1))
    kernel_width = np.sqrt(X.shape[1]) * .75
    weights = np.sqrt(np.exp(-(euclidean_dist ** 2) / kernel_width ** 2))
    return weights

def weight_based_on_class_imbalance(data):
    '''
    get the weight of each class proportional to the number of instances
    '''
    n_samples = len(data['y'])
    y = np.array(data['y']).astype(np.int64)
    n_classes = len(list(np.unique(y)))
    bin_count = np.bincount(y)
    weights = 1 / ((n_classes * bin_count) / n_classes)

    weights /= min(weights)
    # weights /= sum(weights)
    return weights
