'''
original version of LIME from https://github.com/marcotcr/lime
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
# from lime.lime_tabular import LimeTabularExplainer
import fatf.transparency.predictions.surrogate_explainers as surrogates
import fatf.utils.models.validation as fumv


def print_explanation(exp):
    for key in exp.keys():
        print(f'{key} ---')
        for k in exp[key].keys():
            print(f'    {k}: {exp[key][k]:.3f}')

class LIME_fatf:
    '''
    issue/bug with fatf lime version not accepting our black_box_model
        due to False from fatf fumv.check_model_functionality (import fatf.utils.models.validation as fumv)
            even though the black_box_model hasattr(black_box_model, 'predict_proba'))

    '''
    def __init__(self, black_box_model,
                       query_point,
                       data,
                       samples_number=500,
                       **kwargs):
        self.black_box_model = black_box_model
        self.data = data
        # print(data['X'].shape)
        # print(hasattr(black_box_model, 'predict_proba'))
        # print(fumv.check_model_functionality(black_box_model, True, True))
        # print(self.black_box_model.predict_proba)
        self.lime = surrogates.TabularBlimeyLime(self.data['X'], self.black_box_model)
        expl, surrogate_models = self.lime.explain_instance(query_point, samples_number=samples_number, return_models=True)
        self.surrogate_model = surrogate_models['class 1']

    def _predict(self, X):
        '''
        need to transform input data into the 'explanable' domain before prediction
        '''
        X_discretised = self.lime.discretiser.discretise(X)
        return self.surrogate_model.predict(X)

    def predict_proba(self, X):
        return self._predict(X)

    def predict(self, X):
        probability_class_1 = self._predict(X)
        class_prediction = np.heaviside(probability_class_1-0.5, 1)   # threshold class prediction at 0.5
        return class_prediction.astype(np.int64)

class LIME:
    def __init__(self, black_box_model,
                       query_point,
                       data,
                       samples_number=500,
                       **kwargs):
        self.black_box_model = black_box_model
        self.data = data
        explainer = LimeTabularExplainer(self.data['X'], random_state=clime.RANDOM_SEED)
        expl = explainer.explain_instance(query_point, black_box_model.predict_proba)
        print(query_point)
        print(explainer.convert_and_round(query_point))
        print(explainer.discretizer.discretize(query_point))
        ###
        '''
        need to figure out how to extract the interpretable domain mapper and get linear model (can get weights from expl)
        here helps: https://github.com/marcotcr/lime/blob/master/lime/lime_tabular.py#L427
        '''

    def predict_proba(self, X):
        return self.surrogate_model.predict(X)

    def predict(self, X):
        probability_class_1 = self.surrogate_model.predict(X)
        class_prediction = np.heaviside(probability_class_1-0.5, 1)   # threshold class prediction at 0.5
        return class_prediction.astype(np.int64)


if __name__ == '__main__':
    import clime

    # get dataset
    data = clime.data.get_moons()

    # train model
    clf = clime.models.SVM(data)
    # import pdb; pdb.set_trace()

    # now wrap with the clime.model.balancer to see why we get an error with predict_proba

    lime = LIME_fatf(clf, data['X'][1, :], data)
    # lime = LIME(clf, data['X'][1, :], data)


    print(lime.predict(data['X'][2:3, :]))
    print(lime.predict_proba(data['X'][2:5, :]))
