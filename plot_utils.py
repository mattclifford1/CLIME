import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
# plot colours
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
cm = plt.cm.RdBu

def plot_classes(data, ax=None):
    '''
    plot classes in different colour on an axes
    input:
        - data: dictionary with keys 'X', 'y'
    '''
    ax, show = _get_axes(ax)
    ax.scatter(data['X'][:, 0], data['X'][:, 1], c=data['y'], cmap=cm_bright, edgecolors="k")
    if show == True:
        plt.show()

def plot_decision_boundary_sklearn(clf, data, ax=None):
    '''
    plot a decision boundary on axes using sklearn built in method 'DecisionBoundaryDisplay'
    input:
        - clf: sklearn classifier object
    '''
    ax, show = _get_axes(ax)
    DecisionBoundaryDisplay.from_estimator(clf, data['X'], cmap=cm, alpha=0.8, ax=ax, eps=0.5)
    if show == True:
        plt.show()

def plot_decision_boundary(clf, data, ax=None):
    '''
    plot a decision boundary on axes
    input:
        - clf: sklearn classifier object
    '''
    ax, show = _get_axes(ax)
    # get X from data
    X = data['X']
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    yhat = clf.predict_proba(grid)
    # keep just the probabilities for class 0
    yhat = yhat[:, 0]
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    c = ax.contourf(xx, yy, zz, cmap=cm, alpha=0.8)
    # add a legend, called a color bar
    plt.colorbar(c)
    if show == True:
        plt.show()

'''
helper functions
'''
def _get_axes(ax):
    '''
    determine whether to make an axes or not, making axes also means show them
    input:
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
    import model

    # ax = plt.gca()
    _, [ax1, ax2] = plt.subplots(2)
    # get dataset
    train_data, test_data = data_generation.get_data()
    train_data = data_generation.unbalance_data(train_data,[1,0.5])
    clf = model.SVM(train_data)
    plot_decision_boundary(clf, train_data, ax=ax1)
    plot_classes(train_data, ax=ax1)

    clf_bal = model.SVM_balance_boundary(train_data)
    plot_decision_boundary(clf_bal, train_data, ax=ax2)
    plot_classes(train_data, ax=ax2)

    plt.show()
