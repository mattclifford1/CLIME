# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from functools import partial
import multiprocessing
from tqdm.autonotebook import tqdm
from clime.data import costs
import numpy as np

class get_avg_score():
    '''
    wrapper of fidelity metrics to loop over the whole dataset when called
    '''
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, explainer_generator, black_box_model, data, run_parallel=False):
        '''
        get avg/std score given an explainer, black_box_model and data to test on
        '''
        _get_explainer_evaluation_wrapper=partial(get_avg_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  data_dict=data,
                                                  clf=black_box_model,
                                                  metric=self.metric)
        data_list = list(range(len(data['y'])))
        if run_parallel == True:
            with multiprocessing.Pool() as pool:
                scores = list(tqdm(pool.imap_unordered(_get_explainer_evaluation_wrapper, data_list), total=len(data_list), leave=False, desc='Evaluation'))
        else:
            scores = list(map(_get_explainer_evaluation_wrapper, data_list))
        scores = np.array(scores)
        return {'avg': np.mean(scores), 'std': np.std(scores)}

    @staticmethod
    def _get_single_score(query_point_ind, explainer_generator, clf, data_dict, metric):
        '''
        wrapper to use with multiprocessing
        '''
        expl = explainer_generator(clf, data_dict, query_point_ind)
        score = metric(expl, black_box_model=clf,
                             data=data_dict,
                             query_point=data_dict['X'][query_point_ind, :])
        return score


def fidelity(expl, black_box_model, data, **kwargs):
    '''
    get fidelity accuracy between both models
    '''
    same_preds = _get_preds(expl, black_box_model, data)
    # get the accuracy
    fidelity_acc = sum(same_preds)/len(same_preds)
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
    fidelity_acc = sum(same_preds*weights)/ sum(weights)
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
    # get weights dataset based on class imbalance
    weightings = costs.weight_based_on_class_imbalance(data)
    weights = data['y'].copy()
    masks = {}
    for i in range(len(weightings)):
        masks[i] = (weights==i)
    for i in range(len(weightings)):
        weights[masks[i]] = weightings[i]
    return weights


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
