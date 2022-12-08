import fatf.transparency.predictions.surrogate_explainers as surrogates
import fatf.vis.lime as fatf_vis_lime

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



if __name__ == '__main__':
    import data_generation
    import model
    import matplotlib.pyplot as plt

    # get dataset
    train_data, test_data = data.get_moons()

    # train model
    clf = model.SVM(train_data)

    # get lime explainer
    lime = LIME(train_data, clf)

    lime_explanation = lime(test_data['X'][0, :])
    print(lime_explanation)
    lime.plot_explanation()
    plt.show()
