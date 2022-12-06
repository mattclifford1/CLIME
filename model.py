import sklearn.svm
import numpy as np


def SVM(data, **kwargs):
    '''
    train a 'black box' model on dataset
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    clf = sklearn.svm.SVC(gamma=2, C=1, probability=True, **kwargs)
    clf.fit(data['X'], data['y'])
    return clf

def SVM_weighted_training(data):
    return SVM(data, class_weight='balanced')
    

# todo make a superclass  of sklearn classifier
#   and incorportate input_balancer into:
#       - predict_proba
#       - predict
class SVM_balance_boundary:
    '''
    train a 'black box' model on dataset but balance the model by pushing its
    boundary away from the minority class since we are less certain about
    predictions in that area, this is achieved by changing the data point before
    being input to the model.
    **currently only works with pushing minority class towards the majority class
    input:
        - data: dictionary with keys 'X', 'y'
        - weight: scale to push the boundary

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data, weight=1):
        self.weight = weight
        self.fit(data)
        self._get_vector_to_balance(data)

    def fit(self, data):
        self.clf = SVM(data)
        self.fit_status_ = self.clf.fit_status_

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
        return x - (self.bal_vector*self.diff_scale*self.weight)

    def predict(self, x):
        bal_x = self._balance_input(x)
        return self.clf.predict(bal_x)

    def predict_proba(self, x):
        bal_x = self._balance_input(x)
        return self.clf.predict_proba(bal_x)


class SVM_balance_proba:
    '''
    train a 'black box' model on dataset but balance the model by changing the
    probabilities output from the model by scaling them with respect to the
    abundance of that class
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data):
        self._train_model(data)

    def _train_model(self, data):
        self.clf = SVM(data)
        self.fit_status_ = self.clf.fit_status_







if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import data_generation
    import plot_utils

    # get dataset
    train_data, test_data = data_generation.get_data()

    # train model
    clf = SVM(train_data)

    # push clf boundairy away from the minority class
    clf_bal_inp = input_balancer(train_data)

    # plot results
    ax = plt.gca()
    train_data['X'] = clf_bal_inp(train_data['X'], weight=1)
    plot_utils.plot_decision_boundary(clf, train_data, ax=ax)
    plot_utils.plot_classes(train_data, ax=ax)
    plt.show()
