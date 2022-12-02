import sklearn.svm
import numpy as np

def SVM(data):
    '''
    train a 'black box' model on dataset
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    clf = sklearn.svm.SVC(gamma=2, C=1, probability=True)
    clf.fit(data['X'], data['y'])
    return clf


class input_balancer:
    '''
    balance a model by pushing its boundary away from the minority class since
    we are less certain about predictions in that area, this is achieved by
    changing the data point before being input to the model

    **currently only works with pushing minority class towards the majority class
    '''
    def __init__(self, data):
        '''
            - data: dictionary with keys 'X', 'y'
        '''
        self.get_class_means(data)


    def get_class_means(self, data):
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

    def __call__(self, x, weight=1):
        return x + (self.bal_vector*self.diff_scale*weight)





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
