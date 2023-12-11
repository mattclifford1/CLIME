'''
Get the score from key points of a dataset for a given evaluation metric
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from functools import partial
import multiprocessing
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.decomposition import PCA
from clime.data.utils import costs


def get_class_means(data):
    # estimate mean of the data
    classes = np.unique(data['y'])
    means = []
    for cl in classes:
        X_c = data['X'][data['y']==cl, :]
        means.append(np.mean(X_c, axis=0))
    return means

def get_data_edges(data, num_samples=20):
    X = data['X']
    min_data = np.min(X, axis=0)
    max_data = np.max(X, axis=0)
    # sample across
    linspace = np.linspace(min_data, max_data, num_samples)
    return linspace


def get_data_grid(data, num_samples=20):
    '''get data going across the dataset on first two PCA components'''
    # max and min extension of the data along the vector
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(data['X'])
    X = pca.transform(data['X'])

    # find min and max in the first two components
    min_1 = np.min(X[:, 0])
    max_1 = np.max(X[:, 0])
    min_2 = np.min(X[:, 1])
    max_2 = np.max(X[:, 1])
    # sample across first two components
    linspace_1 = np.linspace(min_1, max_1, num_samples)
    linspace_2 = np.linspace(min_2, max_2, num_samples)

    print(linspace_1)
    print(linspace_2)
    X_1, X_2 = np.meshgrid(linspace_1, linspace_2)
    query_points = np.vstack([X_1.ravel(), X_2.ravel()])
    # return to original dataspace
    query_points = pca.inverse_transform(query_points)

    import matplotlib.pyplot as plt
    plt.scatter(query_points[:, 0], query_points[:, 1])
    plt.show()

    return query_points


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

def get_local_points(data, query_point, samples=100):
    # sample locally around the query point with a variance of the dataset
    data_cov = np.cov(data['X'].T)
    local_sample_cov = data_cov #/ 5    # maybe justify this?
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
        elif self.key_points == 'data_edges':
            return get_data_edges(data), None
        elif self.key_points == 'grid':
            return get_data_grid(data), None
        else:
            raise Exception(f'key points not accecpted, was given: {self.key_points}')

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
                results = list(tqdm(pool.imap(_get_explainer_evaluation_wrapper, data_list), total=len(data_list), leave=False, desc='Evaluation'))
        else:
            results = list(map(_get_explainer_evaluation_wrapper, data_list))
        
        results = np.array(results)
        evaluation = results[:, 0]
        class_weights = results[:, 1]
        maj_influ = results[:, 2]
        results = {'avg': np.mean(evaluation), 'std': np.std(evaluation)}
        # if self.key_points is not 'all_points':
        results['eval_points'] = data_points
        if self.key_points in ['between_means', 'data_edges', 'means']:
            results['2D results'] = True
        else:
            results['2D results'] = False
        results['scores'] = evaluation  # structered scores are useful for analysis
        if class_weights[0] != None:
            results['class_weights'] = class_weights
        results['majority influence'] = maj_influ
        return results

    @staticmethod
    def _get_single_score(query_point_ind, explainer_generator, clf, train_data, test_data, query_data_list, metric, test_points):
        '''
        wrapper to use with multiprocessing
        '''
        query_point = query_data_list[query_point_ind]
        expl = explainer_generator(clf, train_data=train_data, test_data=test_data, query_point=query_point)
        if hasattr(expl, 'class_weights'):
            class_weights = 1/(sum(expl.class_weights))
            minority_sample_class = np.argmax(expl.class_weights)
        else:
            class_weights = None
            minority_sample_class = None
        test_points = get_key_points_score.get_test_points(test_data, test_points, query_point)
        score = metric(expl, black_box_model=clf,
                             data=test_points,
                             query_point=query_point)
        min_influ = minority_class_weight_from_black_box(
            clf, test_points, query_point, minority_sample_class)
        return [score, class_weights, min_influ]


def minority_class_weight_from_black_box(black_box_model, data, query_point, minority_sample_class):
    '''get the influence over how balanced the distance is on local eval'''
    if minority_sample_class == None:
        return None
    bb_preds = black_box_model.predict(data['X'])
    distance_weights = costs.weights_based_on_distance(query_point, data['X'])   
    cls_influence = sum(distance_weights[bb_preds == minority_sample_class])
    return cls_influence / sum(distance_weights)