# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

from clime.data import costs
import numpy as np


def fidelity(expl, black_box_model, data, **kwargs):
    '''
    get fidelity accuracy between both models
    '''
    # get prediction from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = expl.predict(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    # get the accuracy
    fidelity_acc = sum(same_preds)/len(bb_preds)
    return fidelity_acc

def local_fidelity(expl, black_box_model, data, query_point):
    '''
    get fidelity accuracy between both models but weight based on
    distance from query point
    '''
    # get predictions from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = expl.predict(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    # get weights dataset based on locality
    weights = costs.weights_based_on_distance(query_point, data['X'])
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
    return fidelity_acc


def bal_fidelity(expl, black_box_model, data, **kwargs):
    '''
    get fidelity accuracy between both models but weight based on
    class imbalance - give higher weight to minority class (prop to instances)
    '''
    # get predictions from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = expl.predict(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    # get weights dataset based on class imbalance
    weightings = costs.weight_based_on_class_imbalance(data)
    weights = data['y'].copy()
    masks = {}
    for i in range(len(weightings)):
        masks[i] = (weights==i)
    for i in range(len(weightings)):
        weights[masks[i]] = weightings[i]
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
    return fidelity_acc


if __name__ == '__main__':
    import data
    import model
    import explainer
    # get dataset
    train_data = data.get_moons()
    train_data = data.unbalance(train_data,[1,0.5])

    # train model
    clf = model.SVM(train_data)

    # BLIMEY!
    q_point = 10
    expl = explainer.bLIMEy(clf, train_data['X'][q_point, :])

    fid = fidelity(expl, clf, train_data)
    print(fid)
    loc_fid = local_fidelity(expl, clf, train_data, train_data['X'][q_point, :])
    print(loc_fid)
    bal_fid = bal_fidelity(expl, clf, train_data)
    print(bal_fid)
