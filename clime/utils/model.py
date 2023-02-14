# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

from clime.data import costs
import numpy as np

def accuracy(clf, data):
    '''
    get accuracy on test set
    '''
    # get prediction from model
    clf_preds = clf.predict(data['X'])
    same_preds = (clf_preds==data['y']).astype(np.int64)
    # get the accuracy
    acc = sum(same_preds)/len(clf_preds)
    return acc

def bal_accuracy(clf, data):
    '''
    get accuracy on test set but weight based on
    class imbalance - give higher weight to minority class (prop to instances)
    '''
    # get prediction from model
    clf_preds = clf.predict(data['X'])
    same_preds = (clf_preds==data['y']).astype(np.int64)
    # get weights dataset based on class imbalance
    weightings = costs.weight_based_on_class_imbalance(data)
    weights = data['y'].copy()
    masks = {}
    for i in range(len(weightings)):
        masks[i] = (weights==i)
    for i in range(len(weightings)):
        weights[masks[i]] = weightings[i]
    # adjust score with weights
    w_acc = sum(same_preds*weights)/ sum(weights)
    return w_acc

def get_model_stats(clf, train_data, test_data):
    return {
        'train accurracy': accuracy(clf, train_data),
        'test accurracy': accuracy(clf, test_data),
        'train balanced accurracy': bal_accuracy(clf, train_data),
        'test balanced accurracy': bal_accuracy(clf, test_data),
    }
