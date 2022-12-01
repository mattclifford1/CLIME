'''
Generate toy data
Data can be balanced or unbalanced
'''
import sklearn.datasets
import sklearn.model_selection

# test commit

def get_data(class_proportion=0.5,
             random_state=42):
    '''
    make half moon dataset from sklearn
        - class_proportion: balance of the classes (default:0.5 is 50/50 split)
                            classes are unbalanced via undersampling

    returns:
        - train_data: dictionary with keys 'X', 'y'
        - test_data:  dictionary with keys 'X', 'y'
    '''
    X, y = sklearn.datasets.make_moons(noise=0.3, random_state=random_state, n_samples=200)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=random_state)
    train_data = {'X': X_train, 'y':y_train}
    test_data =  {'X': X_test,  'y':y_test}

    # now use the class_proportion param to undersample one of the classes


    return train_data, test_data



def balance_data(data):
    '''
    given a dataset, make the classes balanced
    balancing is done via oversmaplign the minority class
        - data: dictionary with keys 'X', 'y'

    returns:
        - data: dictionary with keys 'X', 'y'
    '''
    # make balanced usign oversampling

    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    train_data, test_data = get_data()
    plt.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.show()
