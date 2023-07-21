'''
Implimentation of build LIME yourself (bLIMEy): https://arxiv.org/abs/1910.13016
Use a simplified version where there is no interpretable domain
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn
import clime
from clime.data.utils import costs


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
                       test_data=None,
                       sampling_cov=None,  # provide covariance to sample test_
                       samples=10000,
                       class_weight_data=False,
                       class_weight_sampled=False,
                       class_weight_sampled_probs=False,
                       weight_locally=True,
                       rebalance_sampled_data=False,
                       train_logits=False,
                       logistic_regression=False,
                       **kwargs
                       ):
        self.query_point = query_point
        self.test_data = test_data   # test set to get statistics from
        self.sampling_cov = sampling_cov
        self.samples = samples
        self.class_weight_data = class_weight_data
        self.class_weight_sampled = class_weight_sampled
        self.class_weight_sampled_probs = class_weight_sampled_probs
        self.weight_locally = weight_locally
        self.rebalance_sampled_data = rebalance_sampled_data
        self.train_logits = train_logits
        self.logistic_regression = logistic_regression

        sampled_data = self._sample_locally(black_box_model)
        self._train_surrogate(sampled_data)

    def get_explanation(self):
        return self.surrogate_model.coef_[0, :] # just do for one class (is the negative for the other class)

    def predict_proba(self, X):
        y_ = self.surrogate_model.predict(X)
        return np.clip(y_, 0, 1)  # make probability (might need to run through logistic function?)

    def predict(self, X):
        probability_class_1 = self.surrogate_model.predict(X)[:, 1]
        class_prediction = np.heaviside(probability_class_1-0.5, 1)   # threshold class prediction at 0.5
        return class_prediction.astype(np.int64)

    def _sample_locally(self, black_box_model):
        cov = self._get_local_sampling_cov()
        sampled_data = {}
        sampled_data['X'] = np.random.multivariate_normal(self.query_point, cov, self.samples)
        # get the class predictions from the sampled data (for use with class balanced learning and metrics)
        sampled_data['y'] = black_box_model.predict(sampled_data['X'])
        # option to adjust weights based on class imbalance
        if self.rebalance_sampled_data is True:
            sampled_data = clime.data.balance_oversample(
                sampled_data)
        ''' get behaviour of the blackbox model at the sampled data '''
        # get probabilities to regress on
        sampled_data['p(y|x)'] = black_box_model.predict_proba(sampled_data['X'])

        self.query_probs = black_box_model.predict_proba([self.query_point])
        return sampled_data


    def _get_local_sampling_cov(self):
        if self.sampling_cov is None:
            if self.test_data is None:
                return np.eye(len(self.query_point)) # dont know anything so assume cov is identity matrix
            else:
                # calc cov of data given
                return np.cov(self.test_data['X'].T)
        else:
            return self.sampling_cov

    def _train_surrogate(self, sampled_data):
        sample_weights = self._get_sampled_weights(sampled_data)
        ''' train surrogate '''
        # regresssion model
        if self.train_logits == True:
            self.surrogate_model = clime.models.logit_ridge(alpha=1, 
                                                         fit_intercept=True,
                                                         random_state=clime.RANDOM_SEED)
        elif self.logistic_regression == True:
            self.surrogate_model = clime.models.logistic_regression(
                                                         random_state=clime.RANDOM_SEED)
        else:
            self.surrogate_model = sklearn.linear_model.Ridge(alpha=1, 
                                                           fit_intercept=True,
                                                           random_state=clime.RANDOM_SEED)
        self.surrogate_model.fit(sampled_data['X'],
                                 sampled_data['p(y|x)'],
                                 sample_weight=sample_weights,
                                 )

    def _get_sampled_weights(self, sampled_data):
        ''' training cost weights '''
        # get sample weighting based on distance
        if self.weight_locally is True:
            weights = costs.weights_based_on_distance(self.query_point, sampled_data['X'])
        else:
            weights = np.ones(sampled_data['X'].shape[0])

        # black box training data class imbalance weights/costs
        if self.class_weight_data is True:
            class_weights = costs.weight_based_on_class_imbalance(self.test_data)
            class_preds_matrix = np.round(sampled_data['p(y|x)'])
            # apply to all instances
            instance_class_imbalance_weights = np.dot(class_preds_matrix, class_weights.T)
            # now combine class imbalance weights with distance based weights
            weights *= instance_class_imbalance_weights

        # weights/costs based on class imbalance of the samples
        self.class_weights = costs.weight_based_on_class_imbalance(sampled_data) # save for plotting
        if self.class_weight_sampled is True: 
            # get class imbalance weights
            class_preds_matrix = np.round(sampled_data['p(y|x)'])
            # apply to all instances
            instance_class_imbalance_weights = np.dot(class_preds_matrix, self.class_weights.T)
            # now combine class imbalance weights with distance based weights
            weights *= instance_class_imbalance_weights

        if self.class_weight_sampled_probs is True:
            # adjust probs based on query point's prob
            # get class imbalance weights
            instance_class_imbalance_weights = costs.weights_based_on_class_either_side_of_prob(sampled_data, self.query_probs)
            # now combine class imbalance weights with distance based weights
            weights *= instance_class_imbalance_weights
        return weights

        

if __name__ == '__main__':
    import clime

    # get dataset
    data = clime.data.get_moons()

    # train model
    clf = clime.models.SVM(data)
    # import pdb; pdb.set_trace()

    lime = bLIMEy(clf, data['X'][1, :], data)
    print(lime.predict(data['X'][2:3, :]))
