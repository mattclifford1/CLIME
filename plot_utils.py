import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
# plot colours
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
cm = plt.cm.RdBu

def plot_classes(data, ax=None):
    '''
    plot classes in different colour on an axes
        - data: dictionary with keys 'X', 'y'
    '''
    ax, show = _get_axes(ax)
    ax.scatter(data['X'][:, 0], data['X'][:, 1], c=data['y'], cmap=cm_bright, edgecolors="k")
    if show == True:
        plt.show()

def plot_decision_boundary(clf, data, ax=None):
    '''
    plot a decision boundary on axes
        - clf: sklearn classifier object
    '''
    ax, show = _get_axes(ax)
    # ax.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    DecisionBoundaryDisplay.from_estimator(clf, data['X'], cmap=cm, alpha=0.8, ax=ax, eps=0.5)
    if show == True:
        plt.show()

def _get_axes(ax):
    '''
    determine whether to make an axes or not, making axes also means show them
        - ax: None or matplotlib axes object
    '''
    if ax == None:
        ax = plt.gca()
        show = True
    else:
        show = False
    return ax, show

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay
    import data_generation


    # get dataset
    train_data, test_data = data_generation.get_data()
    plot_classes(train_data)
