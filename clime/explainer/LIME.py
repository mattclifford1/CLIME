'''
original version of LIME (taken from fat forensics)
'''
import numpy as np
import fatf.transparency.predictions.surrogate_explainers as surrogates


def print_explanation(exp):
    for key in exp.keys():
        print(f'{key} ---')
        for k in exp[key].keys():
            print(f'    {k}: {exp[key][k]:.3f}')

class LIME:
    def __init__(self, black_box_model,
                       query_point,
                       data,
                       samples_number=500,
                       **kwargs):
        self.black_box_model = black_box_model
        self.data = data
        print(data['X'].shape)
        print(self.black_box_model.predict_proba)
        self.lime = surrogates.TabularBlimeyLime(self.data['X'], self.black_box_model)
        expl, surrogate_models = self.lime.explain_instance(query_point, samples_number=samples_number, return_models=True)
        self.surrogate_model = surrogate_models['class 1']

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

    lime = LIME(clf, data['X'][1, :], data)
    print(lime.predict(data['X'][2:3, :]))
