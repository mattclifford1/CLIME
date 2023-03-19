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

def get_points_between_class_means(data, num_samples=10):
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
    for i in np.linspace(-0.75, 1.75, num_samples):
        points.append(means[0] + i*c_vector)
    # data for plotting
    plot_data = {'means': means}
    return points, plot_data

def get_all_points(data):
    points = []
    for i in range(data['X'].shape[0]):
        points.append(data['X'][i, :])
    return points

def get_local_points(data, query_point, samples=20):
    # sample locally around the query point with a smaller variance that the dataset
    data_cov = np.cov(data['X'].T)
    local_sample_cov = data_cov / 5    # maybe justify this?
    samples = np.random.multivariate_normal(query_point, local_sample_cov, samples)
    return {'X': samples}

class get_key_points_score():
    '''
    wrapper of metrics to evaluate on key points when called
     - key_points: which points to build an explainer around
     - test_points: points to test/eval the explainer against
    '''
    def __init__(self, key_points='means', test_points='all'):
        self.key_points = key_points
        self.test_points = test_points

    def determine_key_points(self, data):
        '''
        get points to build the explainer from
         - data: the test or train dataset given from the pipeline
        '''
        if self.key_points == 'means':
            return get_class_means(data), None
        elif self.key_points == 'between_means':
            return get_points_between_class_means(data)
        elif self.key_points == 'all_points':
            return get_all_points(data), None

    @staticmethod
    def get_test_points(data, test_points, query_point):
        '''
        get points to test the explainer against
         - data: the test or train dataset given from the pipeline
        '''
        if test_points == 'all':
            return data
        if test_points == 'local':
            return get_local_points(data, query_point)

    def __call__(self, metric, explainer_generator, black_box_model, data, run_parallel=False):
        '''
        get score given an explainer, black_box_model and data to test on
        '''
        data_points, plot_data = self.determine_key_points(data)
        _get_explainer_evaluation_wrapper=partial(get_key_points_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  clf=black_box_model,
                                                  data_dict=data,
                                                  query_data_list=data_points,
                                                  metric=metric,
                                                  test_points=self.test_points)
        data_list = list(range(len(data_points)))
        if run_parallel == True:
            with multiprocessing.Pool() as pool:
                scores = list(tqdm(pool.imap(_get_explainer_evaluation_wrapper, data_list), total=len(data_list), leave=False, desc='Evaluation'))
        else:
            scores = list(map(_get_explainer_evaluation_wrapper, data_list))
        scores = np.array(scores)
        results = {'avg': np.mean(scores), 'std': np.std(scores)}
        if self.key_points is not 'all_points':
            results['eval_points'] = data_points
        if self.key_points == 'between_means':
            results['scores'] = scores  # structered scores are useful for analysis
        return results

    @staticmethod
    def _get_single_score(query_point_ind, explainer_generator, clf, data_dict, query_data_list, metric, test_points):
        '''
        wrapper to use with multiprocessing
        '''
        query_point = query_data_list[query_point_ind]
        expl = explainer_generator(clf, data_dict, query_point=query_point)

        test_points = get_key_points_score.get_test_points(data_dict, test_points, query_point)
        score = metric(expl, black_box_model=clf,
                             data=test_points,
                             query_point=query_point)
        return score
