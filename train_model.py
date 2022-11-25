import sklearn.svm
import data_generation

# get toy data
X_train, X_test, y_train, y_test = data_generation.get_data(class_proportion=0.5)
# train 'black box' classifier
clf = sklearn.svm.SVC(gamma=2, C=1)
clf.fit(X_train, y_train)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.inspection import DecisionBoundaryDisplay
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    cm = plt.cm.RdBu

    ax = plt.gca()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    DecisionBoundaryDisplay.from_estimator(clf, X_train, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
    plt.show()
