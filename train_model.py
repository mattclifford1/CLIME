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
        class_counter = {}
        for class in classes:
            self.means[class] = np.zeros([data['X'].shape[0]])
            class_counter[class] = 0
        for i in len(data['X'].shape[1]):
            self.means[data['y'][i]] += data['X'][:, i]
            class_counter[data['y'][i]] += 1
        for class in classes:
            self.means[class] / class_counter[class]





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import data_generation
    import plot_utils

    # get dataset
    train_data, test_data = data_generation.get_data()

    # train model
    clf = train_model(train_data)

    # plot results
    ax = plt.gca()
    plot_utils.plot_decision_boundary(clf, train_data, ax=ax)
    plot_utils.plot_classes(train_data, ax=ax)
    plt.show()
