'''
Get the average score over the whole dataset for a given evaluation metric
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from functools import partial
import multiprocessing
from tqdm.autonotebook import tqdm
import numpy as np

class get_avg_score():
    '''
    wrapper of metrics to loop over the whole dataset when called
    '''
    def __call__(self, metric, explainer_generator, black_box_model, data, run_parallel=False):
        '''
        get avg/std score given an explainer, black_box_model and data to test on
        '''
        _get_explainer_evaluation_wrapper=partial(get_avg_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  clf=black_box_model,
                                                  data_dict=data,
                                                  metric=metric)
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
        query_point = data_dict['X'][query_point_ind, :]
        expl = explainer_generator(clf, data_dict, query_point=query_point)
        score = metric(expl, black_box_model=clf,
                             data=data_dict,
                             query_point=query_point)
        return score
