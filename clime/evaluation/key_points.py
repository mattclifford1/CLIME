'''
Get the score from key points of a dataset for a given evaluation metric
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from functools import partial
import multiprocessing
from tqdm.autonotebook import tqdm
import numpy as np

class get_key_points_score():
    '''
    wrapper of metrics to loop over the whole dataset when called
    '''
    def __init__(self, metric):
        self.metric = metric

    def determine_key_points(self, data):
        # estimate mean and cov from the data
        classes = len(np.unique(data['y']))
        means = []
        for cl in range(classes):
            X_c = data['X'][data['y']==cl, :]
            means.append(np.mean(X_c, axis=0))
        print(means)
        return means


    def __call__(self, explainer_generator, black_box_model, data, run_parallel=False):
        '''
        get score given an explainer, black_box_model and data to test on
        '''
        data_points = self.determine_key_points(data)
        _get_explainer_evaluation_wrapper=partial(get_key_points_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  clf=black_box_model,
                                                  data_dict=data,
                                                  query_data=data_points,
                                                  metric=self.metric)
        data_list = list(range(len(data_points)))
        if run_parallel == True:
            with multiprocessing.Pool() as pool:
                scores = list(tqdm(pool.imap_unordered(_get_explainer_evaluation_wrapper, data_list), total=len(data_list), leave=False, desc='Evaluation'))
        else:
            scores = list(map(_get_explainer_evaluation_wrapper, data_list))
        scores = np.array(scores)
        return {'avg': np.mean(scores), 'std': np.std(scores)}

    @staticmethod
    def _get_single_score(query_point_ind, explainer_generator, clf, data_dict, query_data, metric):
        '''
        wrapper to use with multiprocessing
        '''
        expl = explainer_generator(clf, data_dict, query_point_ind)
        score = metric(expl, black_box_model=clf,
                             data=data_dict,
                             query_point=data_dict['X'][query_point_ind, :])
        return score
