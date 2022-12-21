# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import sklearn.svm
import numpy as np
from clime import costs


class SVM(sklearn.svm.SVC):
    '''
    train a 'black box' model on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    to train with class balance weighting using the kwarg: class_weight='balanced'
    '''
    def __init__(self, data, gamma=2, C=1, probability=True, **kwargs):
        self.data = data    # colab wont work unless we attribute data? (older python version)
        super().__init__(gamma=gamma, C=C, probability=probability, **kwargs)
        self.fit(self.data['X'], self.data['y'])


class SVM_balance_boundary(SVM):
    '''
    train a 'black box' model on dataset but balance the model by pushing its
    boundary away from the minority class since we are less certain about
    predictions in that area, this is achieved by changing the data point before
    being input to the model.
    **currently only works with pushing minority class towards the majority class
    input:
        - data: dictionary with keys 'X', 'y'
        - boundary_weight: scale to push the boundary

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data, boundary_weight=1, **kwargs):
        super().__init__(data, **kwargs)
        self.boundary_weight = boundary_weight
        self._get_vector_to_balance(data)

    def _get_vector_to_balance(self, data):
        '''get the mean value of each class'''
        ys = np.array(data['y'])
        classes = list(np.unique(ys))
        self.means = {}
        self.class_counter = {}
        # set up data storers
        for clas in classes:
            self.means[clas] = np.zeros([data['X'].shape[1]])
            self.class_counter[clas] = 0
        # loop through the dataset
        for i in range(data['X'].shape[0]):
            self.means[data['y'][i]] += data['X'][i, :]
            self.class_counter[data['y'][i]] += 1
        # calc mean
        for clas in classes:
            self.means[clas] /= self.class_counter[clas]
        # get vector from the minority class to the majority class
        min = np.argmin(list(self.class_counter.values()))
        max = np.argmax(list(self.class_counter.values()))
        self.bal_vector = self.means[max] - self.means[min]
        # work out the scale of the class imbalance
        self.diff_scale = self.class_counter[max] - self.class_counter[min]
        # if self.diff_scale == 0:
        #     # make no change when classes are balanced
        #     self.bal_vector = np.zeros(self.bal_vector.shape)
        self.diff_scale /= data['X'].shape[0]

    def _balance_input(self, x):
        return x - (self.bal_vector*self.diff_scale*self.boundary_weight)

    def predict(self, x):
        bal_x = self._balance_input(x)
        return super().predict(bal_x)

    def predict_proba(self, x):
        bal_x = self._balance_input(x)
        return super().predict_proba(bal_x)


class SVM_balance_proba(SVM):
    '''
    train a 'black box' model on dataset but balance the model by changing the
    probabilities output from the model by scaling them with respect to the
    abundance of data points in that class
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._class_weightings(data)

    def _class_weightings(self, data):
        '''
        get the weight of each class proportional to the number of instances
        '''
        self.class_weights = costs.weight_based_on_class_imbalance(data)

    def predict_proba(self, x):
        '''
        reduce the probability of less representative class
        '''
        preds = super().predict_proba(x)
        # adjust probability with the weights
        adjusted = preds * self.class_weights
        # normalise to make a probability again -- is this okay to be doing????
        adjust_probs = adjusted / np.sum(adjusted, axis=1)[:, None]
        return adjust_probs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import data
    import plot_utils

    # get dataset
    train_data = data.get_moons()
    train_data = data.unbalance(train_data,[1,0.5])

    # train model
    # clf = SVM(train_data)
    clf = SVM(train_data)
    clf_bal_prob = SVM_balance_proba(train_data)
    print(clf.predict_proba([[1,2]]), sum(sum(clf.predict_proba([[1,2]]))))
    print(clf_bal_prob.predict_proba([[1,2]]), sum(sum(clf_bal_prob.predict_proba([[1,2]]))))


    # # plot results
    ax = plt.gca()
    plot_utils.plot_decision_boundary(clf, train_data, ax=ax)
    plot_utils.plot_classes(train_data, ax=ax)
    plt.show()
