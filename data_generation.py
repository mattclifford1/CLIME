'''
Generate toy data
Data can be balanced or unbalanced
'''
import sklearn.datasets
import sklearn.model_selection

# test commit

def get_data(class_proportion=0.5):
    '''
    make half moon dataset from sklearn
        - class_proportion: balance of the classes (default:0.5 is 50/50 split)
    '''
    X, y = sklearn.datasets.make_moons(noise=0.3, random_state=0, n_samples=200)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    X_train, X_test, y_train, y_test = get_data()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    plt.show()
