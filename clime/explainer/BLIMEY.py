'''
Implimentation of build LIME yourself (bLIMEy): https://arxiv.org/abs/1910.13016
Use a simplified version where there is no interpretable domain
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
                       data=None,
                       data_lims=None,
                       samples=10000,
                       class_weight_data=False,
                       class_weight_sampled=False,
                       rebalance_sampled_data=False):
        self.query_point = query_point
        self.data_test = data   # test set to get statistics from
        self.data_lims = data_lims
        self.samples = samples
        self.class_weight_data = class_weight_data
        self.class_weight_sampled = class_weight_sampled
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
        if self.class_weight_data is True:   # weight from given weights
            class_weights = costs.weight_based_on_class_imbalance(self.data_test)
            class_preds_matrix = np.round(self.data['p(y)'])
            # apply to all instances
            instance_class_imbalance_weights = np.dot(class_preds_matrix, class_weights.T)
            # now combine class imbalance weights with distance based weights
            weights *= instance_class_imbalance_weights
        if self.class_weight_sampled is True: # weight based on imbalance of the samples
            # get the class predictions from the sampled data
            self.data['y'] = black_box_model.predict(self.data['X'])
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

if __name__ == '__main__':
    import clime

    # get dataset
    data = clime.data.get_moons()

    # train model
    clf = clime.models.SVM(data)
    # import pdb; pdb.set_trace()

    lime = bLIMEy(clf, data['X'][1, :], data)
    print(lime.predict(data['X'][2:3, :]))
