import sklearn.svm
import data_generation


def train_model(data):
    '''
    train a 'black box' model on dataset
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    clf = sklearn.svm.SVC(gamma=2, C=1)
    clf.fit(data['X'], data['y'])
    return clf



if __name__ == '__main__':
    import matplotlib.pyplot as plt
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
