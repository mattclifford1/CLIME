'''
Get the score from key points of a dataset for a given evaluation metric
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from functools import partial
import multiprocessing
from tqdm.autonotebook import tqdm
import numpy as np


def get_class_means(data):
    # estimate mean of the data
    classes = len(np.unique(data['y']))
    means = []
    for cl in range(classes):
        X_c = data['X'][data['y']==cl, :]
        means.append(np.mean(X_c, axis=0))
    return means

def get_points_between_class_means(data, num_samples=7):
    # estimate mean of the data and get points between
    # !!! currently only works for 2 classes !!!
    classes = len(np.unique(data['y']))
    means = []
    for cl in range(classes):
        X_c = data['X'][data['y']==cl, :]
        means.append(np.mean(X_c, axis=0))
    if len(means) > 2:
        raise Exception(f"'get_points_between_class_means' only supports 2 classes, was given {len(means)}")
    c_vector = means[1] - means[0]
    points = []
    for i in np.linspace(0, 1, num_samples):
        points.append(means[0] + i*c_vector)
    return points

class get_key_points_score():
    '''
    wrapper of metrics to loop over the whole dataset when called
    '''
    def __init__(self, key_points='means'):
        self.key_points = key_points

    def determine_key_points(self, data):
        if self.key_points is 'means':
            return get_class_means(data)
        elif self.key_points is 'between_means':
            return get_points_between_class_means(data)


    def __call__(self, metric, explainer_generator, black_box_model, data, run_parallel=False):
        '''
        get score given an explainer, black_box_model and data to test on
        '''
        data_points = self.determine_key_points(data)
        _get_explainer_evaluation_wrapper=partial(get_key_points_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  clf=black_box_model,
                                                  data_dict=data,
                                                  query_data_list=data_points,
                                                  metric=metric)
        data_list = list(range(len(data_points)))
        if run_parallel == True:
            with multiprocessing.Pool() as pool:
                scores = list(tqdm(pool.imap_unordered(_get_explainer_evaluation_wrapper, data_list), total=len(data_list), leave=False, desc='Evaluation'))
        else:
            scores = list(map(_get_explainer_evaluation_wrapper, data_list))
        scores = np.array(scores)
        return {'avg': np.mean(scores), 'std': np.std(scores), 'eval_points': data_points}

    @staticmethod
    def _get_single_score(query_point_ind, explainer_generator, clf, data_dict, query_data_list, metric):
        '''
        wrapper to use with multiprocessing
        '''
        query_point = query_data_list[query_point_ind]
        expl = explainer_generator(clf, data_dict, query_point=query_point)
        score = metric(expl, black_box_model=clf,
                             data=data_dict,
                             query_point=query_point)
        return score


def get_key_points_means_score():
    return get_key_points_score(key_points='means')

def get_key_points_points_between_means_score():
    return get_key_points_score(key_points='between_means')
