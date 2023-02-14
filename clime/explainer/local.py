'''
Implimentation of build LIME yourself (bLIMEy): https://arxiv.org/abs/1910.13016
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

import numpy as np
import sklearn
import clime
from clime.data import costs


class bLIMEy:
    '''
    create our own version of LIME that has no access to the training data
    current steps:
        - sample around the query point in the feature domain (Guassian)
        - get model probabilities from sampled data
        - train ridge regression on the probabilities (weighted with exp. kernel)

    Currently only work for binary classification tasks

    Input:
        - clf: classifier with .predict_proba() attribute
        - query_point: data point to locally explain
        - data_lim: *not yet implimented*

    Attributes:
        - get_explanation: returns the weights of the surrogate model
                           (feature importance used to predict the prob of that class)
        - predict returns the surrogate model's locally faithful prediction
    '''
    def __init__(self, black_box_model,
                       query_point,
                       data_lims=None,
                       samples=10000,
                       class_weight=False,
                       rebalance_sampled_data=False):
        self.query_point = query_point
        self.data_lims = data_lims
        self.samples = samples
        self.class_weight = class_weight
        self.rebalance_sampled_data = rebalance_sampled_data
        self.data = {}
        self._sample_locally(black_box_model)
        self._train_surrogate(black_box_model)

    def get_explanation(self):
        return self.surrogate_model.coef_[0, :] # just do for one class (is the negative for the other class)

    def predict_proba(self, X):
        return self.surrogate_model.predict(X)

    def predict(self, X):
        probability_class_1 = self.surrogate_model.predict(X)[:, 1]
        class_prediction = np.heaviside(probability_class_1-0.5, 1)   # threshold class prediction at 0.5
        return class_prediction.astype(np.int64)

    def _sample_locally(self, black_box_model):
        cov = self._get_local_sampling_cov()
        self.data['X'] = np.random.multivariate_normal(self.query_point, cov, self.samples)
        self.data['y'] = black_box_model.predict(self.data['X'])

    def _get_local_sampling_cov(self):
        if self.data_lims is None:
            return np.eye(len(self.query_point))
        else:
            # calculate from data lims ?
            return np.eye(len(self.query_point))   # change this to implement var from data lims

    def _train_surrogate(self, black_box_model):
        # option to adjust weights based on class imbalance
        if self.rebalance_sampled_data is True:
            self.data = clime.data.balance_oversample(self.data)
        # get probabilities to regress on
        self.data['p(y)'] = black_box_model.predict_proba(self.data['X'])
        # get sample weighting based on distance
        weights = costs.weights_based_on_distance(self.query_point, self.data['X'])
        if self.class_weight is True:
            # get class imbalance weights
            class_weights = costs.weight_based_on_class_imbalance(self.data)
            class_preds_matrix = np.round(self.data['p(y)'])
            # apply to all instances
            instance_class_imbalance_weights = np.dot(class_preds_matrix, class_weights.T)
            # now combine class imbalance weights with distance based weights
            weights *= instance_class_imbalance_weights
        # regresssion model
        self.surrogate_model = sklearn.linear_model.Ridge(alpha=1, fit_intercept=True,
                                    random_state=clime.RANDOM_SEED)
        self.surrogate_model.fit(self.data['X'],
                                 self.data['p(y)'],
                                 sample_weight=weights,
                                 )

# import fatf.transparency.predictions.surrogate_explainers as surrogates
# import fatf.vis.lime as fatf_vis_lime
#
#
# def print_explanation(exp):
#     for key in exp.keys():
#         print(f'{key} ---')
#         for k in exp[key].keys():
#             print(f'    {k}: {exp[key][k]:.3f}')
#
# class LIME:
#     def __init__(self, data, clf):
#         self.data = data
#         self.clf = clf
#         self.lime = surrogates.TabularBlimeyLime(self.data['X'], self.clf)
#
#     def __call__(self, data_instance, samples_number=500):
#         self.expl = self.lime.explain_instance(data_instance, samples_number=samples_number)
#         return self.expl
#
#     def plot_explanation(self):
#         fatf_vis_lime.plot_lime(self.expl)


if __name__ == '__main__':
    import data
    import model
    import matplotlib.pyplot as plt

    # get dataset
    data = data.get_moons()

    # train model
    clf = model.SVM(data)
    # import pdb; pdb.set_trace()

    # BLIMEY!
    blimey = bLIMEy(clf,
                    data['X'][0, :],
                    class_weight=True)
    print(blimey.get_explanation())
    print(blimey.predict([[1, 2]]))





    # # get lime explainer
    # lime = LIME(data, clf)
    #
    # lime_explanation = lime(data['X'][0, :])
    # print_explanation(lime_explanation)
    #
    # import lime
    # import lime.lime_tabular
    #
    # explainer = lime.lime_tabular.LimeTabularExplainer(data['X'],
    #                                                discretize_continuous=True,
    #                                                )
    # exp = explainer.explain_instance(data['X'][0, :],
    #                              clf.predict_proba,
    #                              # num_features=2,
    #                              # top_labels=1,
    #                              )
    #
    # print(exp.as_list())
    # # lime.plot_explanation()
    # # plt.show()
    #
    # dataset = data['X']
    # feature_names = ['f1', 'f2']
    #
    # import fatf.utils.array.tools as fuat
    # import fatf.utils.data.discretisation as fudd
    # import fatf.utils.data.augmentation as fuda
    #
    # indices = fuat.indices_by_type(dataset)
    # num_indices = set(indices[0])
    # cat_indices = set(indices[1])
    # all_indices = num_indices.union(cat_indices)
    #
    # numerical_indices = list(num_indices)
    # categorical_indices = list(cat_indices)
    # column_indices = list(range(dataset.shape[1]))
    #
    # discretiser = fudd.QuartileDiscretiser(
    #         dataset,
    #         categorical_indices=categorical_indices,
    #         feature_names=feature_names)
    # dataset_discretised = discretiser.discretise(dataset)
    # print(dataset_discretised[:2, :])
    # print(dataset[:2, :])
    # augmenter = fuda.NormalSampling(
    #         dataset_discretised, categorical_indices=column_indices)
    # print(augmenter.sample())
