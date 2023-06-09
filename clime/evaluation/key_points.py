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
    classes = np.unique(data['y'])
    means = []
    for cl in classes:
        X_c = data['X'][data['y']==cl, :]
        means.append(np.mean(X_c, axis=0))
    return means

def get_points_between_class_means(data, num_samples=20):
    '''
    estimate mean of the data and get points between 
    !!! currently only works for 2 classes !!!
    '''
    # get means for each class
    means = get_class_means(data)
    if len(means) > 2:
        raise Exception(f"'get_points_between_class_means' only supports 2 classes, was given {len(means)}")
    # vector between the two classes
    gradients = means[1] - means[0]
    gradients /= np.sum(gradients)   # noramlise

    # max and min extension of the data along the vector
    min_data = np.min(data['X'], axis=0)
    max_data = np.max(data['X'], axis=0)

    # determine the min and max values for x allowed based on the sign of the gradient
    x_mins = []
    x_maxs = []
    for i, g in enumerate(gradients):
        if g > 0:
            x_mins.append(min_data[i])
            x_maxs.append(max_data[i])
        elif g < 0:
            x_mins.append(max_data[i])
            x_maxs.append(min_data[i])
    x_mins = np.array(x_mins)
    x_maxs = np.array(x_maxs)

    # solve and find closest point (first remove indices with 0 gradient as cannot divide by 0)
    means_0 = np.delete(means[0], np.where(gradients==0))
    non_0_gradients = np.delete(gradients, np.where(gradients==0))
    min_ = np.max((x_mins-means_0)/non_0_gradients)
    max_ = np.min((x_maxs-means_0)/non_0_gradients)

    # min_ = np.median((x_mins-means_0)/non_0_gradients)
    # max_ = np.median((x_maxs-means_0)/non_0_gradients)

    # store query points along the vector
    points = []
    for i in np.linspace(min_, max_, num_samples):
        points.append(means[0] + i*gradients)
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

    def __call__(self, metric, explainer_generator, black_box_model, train_data, test_data, run_parallel=False):
        '''
        get score given an explainer, black_box_model and data to test on
        '''
        data_points, plot_data = self.determine_key_points(test_data)
        _get_explainer_evaluation_wrapper=partial(get_key_points_score._get_single_score,
                                                  explainer_generator=explainer_generator,
                                                  clf=black_box_model,
                                                  train_data=train_data,
                                                  test_data=test_data,
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
    def _get_single_score(query_point_ind, explainer_generator, clf, train_data, test_data, query_data_list, metric, test_points):
        '''
        wrapper to use with multiprocessing
        '''
        query_point = query_data_list[query_point_ind]
        expl = explainer_generator(clf, train_data=train_data, test_data=test_data, query_point=query_point)

        test_points = get_key_points_score.get_test_points(test_data, test_points, query_point)
        score = metric(expl, black_box_model=clf,
                             data=test_points,
                             query_point=query_point)
        return score
