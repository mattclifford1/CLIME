from clime import explainer
import numpy as np


def fidelity(blimey, black_box_model, data):
    '''
    get fidelity accuracy between both models
    '''
    # get prediction from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = blimey.predict_locally(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    # get the accuracy
    fidelity_acc = sum(same_preds)/len(bb_preds)
    return fidelity_acc

def local_fidelity(blimey, black_box_model, data, query_point):
    '''
    get fidelity accuracy between both models but weight based on
    distance from query point
    '''
    # get predictions from both models
    bb_preds = black_box_model.predict(data['X'])
    expl_preds = blimey.predict_locally(data['X'])
    same_preds = (bb_preds==expl_preds).astype(np.int64)
    # get weights dataset based on locality
    weights = explainer.weights_based_on_distance(query_point, data['X'])
    # adjust score with weights
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
    return fidelity_acc


if __name__ == '__main__':
    import data
    import model
    import explainer
    # get dataset
    data = data.get_moons()

    # train model
    clf = model.SVM(data)

    # BLIMEY!
    q_point = 10
    blimey = explainer.bLIMEy(clf, data['X'][q_point, :])

    fid = fidelity(blimey, clf, data)
    print(fid)
    loc_fid = local_fidelity(blimey, clf, data, data['X'][q_point, :])
    print(loc_fid)
