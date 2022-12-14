import numpy as np
import sklearn
import clime
import fatf.transparency.predictions.surrogate_explainers as surrogates
import fatf.vis.lime as fatf_vis_lime


def print_explanation(exp):
    for key in exp.keys():
        print(f'{key} ---')
        for k in exp[key].keys():
            print(f'    {k}: {exp[key][k]:.3f}')

class LIME:
    def __init__(self, data, clf):
        self.data = data
        self.clf = clf
        self.lime = surrogates.TabularBlimeyLime(self.data['X'], self.clf)

    def __call__(self, data_instance, samples_number=500):
        self.expl = self.lime.explain_instance(data_instance, samples_number=samples_number)
        return self.expl

    def plot_explanation(self):
        fatf_vis_lime.plot_lime(self.expl)


class bLIMEy:
    '''
    create our own version of LIME that has no access to the training data
    '''
    def __init__(self, clf, query_point, data_lims=None):
        self.clf = clf
        self.query_point = query_point
        self.data_lims = data_lims
        self.data = {}
        self.sample_locally()
        self.train_surrogate()
        print(self.surrogate_model.coef_)

    def sample_locally(self):
        cov = self.get_sampling_cov()
        samples = 10000
        self.data['X'] = np.random.multivariate_normal(self.query_point, cov, samples)
        self.data['y'] = self.clf.predict(self.data['X'])
        self.data['p(y)'] = self.clf.predict_proba(self.data['X'])

    def get_sampling_cov(self):
        if self.data_lims == None:
            return np.eye(len(self.query_point))
        else:
            # calculate from data lims ?
            return np.eye(len(self.query_point))   # change this to implilmet var from data lims

    def get_sample_weights(self):
        '''get the weighting of each sample proportional to the distance to the query point
           weights generated using exponential kernel found in the original lime implementation'''
        euclidean_dist = np.sqrt(np.sum((self.data['X'] - self.query_point)**2, axis=1))
        kernel_width = np.sqrt(self.data['X'].shape[1]) * .75
        self.data['weights'] = np.sqrt(np.exp(-(euclidean_dist ** 2) / kernel_width ** 2))

    def train_surrogate(self):
        self.get_sample_weights()
        self.surrogate_model = sklearn.linear_model.Ridge(alpha=1, fit_intercept=True,
                                    random_state=clime.RANDOM_SEED)
        self.surrogate_model.fit(self.data['X'],
                                 self.data['p(y)'],
                                 sample_weight=self.data['weights'],
                                 )


if __name__ == '__main__':
    import data
    import model
    import matplotlib.pyplot as plt

    # get dataset
    data = data.get_moons()

    # train model
    clf = model.SVM(data)

    # BLIMEY!
    blimey = bLIMEy(clf, data['X'][0, :])





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
