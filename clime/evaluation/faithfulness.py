# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import warnings
from clime.data import costs
import numpy as np

# from sklearn.metrics import log_loss
from rbig._src.mutual_info import MutualInfoRBIG
from scipy.stats import spearmanr


def rbig_kl(expl, black_box_model, data, query_class=0, **kwargs):
    '''
    calculate KL between the probabilities outputted by both models
    '''
    bb_preds = black_box_model.predict_proba(data['X'])[:, query_class]
    expl_preds = expl.predict_proba(data['X'])[:, 0]
    # Initialize RBIG class
    rbig_model = MutualInfoRBIG(max_layers=100)

    # fit model to the data
    rbig_model.fit(bb_preds[:, np.newaxis], expl_preds[:, np.newaxis])
    MI_rbig = rbig_model.mutual_info()
    return MI_rbig

def spearman(expl, black_box_model, data, query_class=0, **kwargs):
    '''
    calculate Spearman correlation between probabilities outputted by both
    models
    '''
    bb_preds = black_box_model.predict_proba(data['X'])[:, query_class]
    expl_preds = expl.predict_proba(data['X'])[:, 0]
    corr = spearmanr(bb_preds, expl_preds)[0]
    return corr

def log_loss_score(expl, black_box_model, data, **kwargs):
    stabilty_constant = 1e-7
    y = black_box_model.predict_proba(data['X']).astype(np.float64) + stabilty_constant
    p = expl.predict_proba(data['X']).astype(np.float64) + stabilty_constant
    log_loss = - (y[:, 0]*np.log(p[:, 0]) + (y[:, 1])*np.log((p[:, 1])))
    return log_loss.mean()

def local_log_loss_score(expl, black_box_model, data, query_point, **kwargs):
    # get log loss
    stabilty_constant = 1e-7
    y = black_box_model.predict_proba(data['X']).astype(np.float64) + stabilty_constant
    p = expl.predict_proba(data['X']).astype(np.float64) + stabilty_constant
    log_loss = - (y[:, 0]*np.log(p[:, 0]) + (y[:, 1])*np.log((p[:, 1])))
    # weight locally
    weights = costs.weights_based_on_distance(query_point, data['X'])
    mean_loss = sum(log_loss*weights) / sum(weights)
    return mean_loss

def Brier_score(expl, black_box_model, data, **kwargs):
    y = black_box_model.predict_proba(data['X']).astype(np.float64)[:, 0]
    p = expl.predict_proba(data['X']).astype(np.float64)[:, 0]
    scores = (p - y)**2
    return scores.mean()

def local_Brier_score(expl, black_box_model, data, query_point, **kwargs):
    y = black_box_model.predict_proba(data['X']).astype(np.float64)[:, 0]
    p = expl.predict_proba(data['X']).astype(np.float64)[:, 0]
    scores = (p - y)**2
    weights = costs.weights_based_on_distance(query_point, data['X'])
    mean_score = sum(scores*weights) / sum(weights)
    return mean_score

def fidelity(expl, black_box_model, data, **kwargs):
    '''
    get fidelity accuracy between both models
    '''
    same_preds = _get_preds(expl, black_box_model, data)
    # get the accuracy
    fidelity_acc = sum(same_preds) / len(same_preds)
    return fidelity_acc

def local_fidelity(expl, black_box_model, data, query_point, **kwargs):
    '''
    get fidelity accuracy between both models but weight based on
    distance from query point
    '''
    same_preds = _get_preds(expl, black_box_model, data)
    # get weights dataset based on locality
    weights = costs.weights_based_on_distance(query_point, data['X'])
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights) / sum(weights)
    return fidelity_acc

def bal_fidelity(expl, black_box_model, data, **kwargs):
    '''
    get fidelity accuracy between both models but weight based on
    class imbalance - give higher weight to minority class (prop to instances)
    '''
    same_preds = _get_preds(expl, black_box_model, data)
    weights = _get_class_weights(data)
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
    return fidelity_acc

def local_and_bal_fidelity(expl, black_box_model, data, query_point, **kwargs):
    '''
    combine the weights of both bal and local
    '''
    same_preds = _get_preds(expl, black_box_model, data)
    weights = costs.weights_based_on_distance(query_point, data['X'])
    weights *= _get_class_weights(data)
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
    return fidelity_acc

def _get_preds(expl, black_box_model, data):
    # get prediction from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = expl.predict(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    return same_preds

def _get_class_weights(data):
    # make sure we have class data (sampled data from LIME won't have this)
    if 'y' not in data.keys():
        warnings.warn("No class data 'y': not using class balanced weightings", Warning)
        return np.ones(data['X'].shape[0])
    # get weights dataset based on class imbalance
    weightings = costs.weight_based_on_class_imbalance(data)
    weights = data['y'].copy()
    masks = {}
    for i in range(len(weightings)):
        masks[i] = (weights==i)
    for i in range(len(weightings)):
        weights[masks[i]] = weightings[i]
    return weights


# if __name__ == '__main__':
#     import data
#     import models
#     import explainer
#     # get dataset
#     train_data = data.get_moons()
#     # train_data = data.unbalance(train_data,[1,0.5])

#     # train model
#     clf = models.SVM(train_data)

#     # BLIMEY!
#     q_point = 10
#     expl = explainer.bLIMEy(clf, train_data['X'][q_point, :])

#     fid = fidelity(expl, clf, train_data)
#     print(fid)
#     loc_fid = local_fidelity(expl, clf, train_data, train_data['X'][q_point, :])
#     print(loc_fid)
#     bal_fid = bal_fidelity(expl, clf, train_data)
#     print(bal_fid)
