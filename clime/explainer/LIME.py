'''
original version of LIME from https://github.com/marcotcr/lime
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import fatf.transparency.predictions.surrogate_explainers as surrogates
import fatf.utils.data.transformation as fudt
import fatf.utils.array.tools as fuat
# import fatf.utils.models.validation as fumv
# from lime.lime_tabular import LimeTabularExplainer



def print_explanation(exp):
    for key in exp.keys():
        print(f'{key} ---')
        for k in exp[key].keys():
            print(f'    {k}: {exp[key][k]:.3f}')

class LIME_fatf:
    '''
    version of LIME from fat-forensics so we have access to the models and domain transformers
    '''
    def __init__(self, black_box_model,
                       query_point,
                       test_data,
                       samples_number=500,
                       **kwargs):
        self.black_box_model = black_box_model
        self.data = test_data
        # print(data['X'].shape)
        # print(hasattr(black_box_model, 'predict_proba'))
        # print(fumv.check_model_functionality(black_box_model, True, True))
        # print(self.black_box_model.predict_proba)
        self.lime = surrogates.TabularBlimeyLime(self.data['X'], self.black_box_model)
        expl, surrogate_models = self.lime.explain_instance(query_point, samples_number=samples_number, return_models=True)
        self.query_point_discretised = self.lime.discretiser.discretise(query_point)
        self.surrogate_model = surrogate_models['class 1']

    def _predict(self, X):
        '''
        need to transform input data into the 'explanable' domain before prediction
        using discretiser: https://github.com/fat-forensics/fat-forensics/blob/master/fatf/transparency/predictions/surrogate_explainers.py#L1012
        and binariser: https://github.com/fat-forensics/fat-forensics/blob/master/fatf/transparency/predictions/surrogate_explainers.py#L1365-L1367
        '''
        X_discretised = self.lime.discretiser.discretise(X)
        X_bin = fudt.dataset_row_masking(X_discretised, self.query_point_discretised)
        X_bin = fuat.as_unstructured(X_bin)
        return self.surrogate_model.predict(X_bin)

    def predict_proba(self, X):
        p_class0 = np.expand_dims(self._predict(X), axis=1)
        return np.concatenate([p_class0, 1 - p_class0], axis=1)

    def predict(self, X):
        probability_class_1 = self._predict(X)
        class_prediction = np.heaviside(probability_class_1-0.5, 1)   # threshold class prediction at 0.5
        return class_prediction.astype(np.int64)



'''
original LIME implementation below (use only to validate against the fatf version)
'''
class LIME:
    def __init__(self, black_box_model,
                       query_point,
                       test_data,
                       samples_number=500,
                       **kwargs):
        self.black_box_model = black_box_model
        self.data = test_data
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
