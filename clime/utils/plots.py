# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

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
    ax.grid(False)
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

'''
juypter-notebook plotting
'''
def plot_data_dict(data_dict):
    fig, axs = plt.subplots(1, len(data_dict.keys()))
    for i, key in enumerate(data_dict.keys()):
        plot_classes(data_dict[key], axs[i])
        axs[i].set_title(key)

# def plot_clfs():
#     fig, axs = plt.subplots(1, len(datasets.keys()))
#     for i, key in enumerate(datasets.keys()):
#         plot_utils.plot_classes(datasets[key], axs[i])
#         plot_utils.plot_decision_boundary(models[key], datasets[key], ax=axs[i])
#         axs[i].set_title(key)

def plot_clfs(data_dict, ax_x=2, title=True):
    '''
    data dict has keys:
        - model
        - data
    '''
    ax_y = int(np.ceil(len(data_dict.keys())/ax_x))
    fig, axs = plt.subplots(ax_y, ax_x)
    if ax_y == 1 and ax_x == 1:
        axs = [axs]
    count = 0
    keys = list(data_dict.keys())
    for i in range(ax_x):
        for j in range(ax_y):
            key = keys[count]
            data = data_dict[key]['data']
            if data['X'].shape[1] > 2:
                # TODO: impliment PCA or TSNE to reduce data dims for plotting
                continue
            model = data_dict[key]['model']
            if ax_y > 1:
                ax = axs[i][j]
            else:
                ax = axs[i]
            plot_classes(data, ax)
            plot_decision_boundary(model, data, ax=ax)
            if title is True:
                ax.set_title(key)
            count += 1

def plot_bar_dict(data_dict, title='', ylabel=None, stds=True, ax=None, ylim=None):
    if ax is None:
        fig, ax = plt.subplots()
    keys = list(data_dict.keys())
    x_pos = np.arange(len(keys))
    if stds is True:
        avgs = []
        stds = []
        for key in keys:
            avgs.append(data_dict[key]['avg'])
            stds.append(data_dict[key]['std'])
        ax.bar(x_pos, avgs, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    else:
        ax.bar(x_pos, list(data_dict.values()), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_title(title)
    if ylabel is None:
        ylabel = ''
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_multiple_bar_dicts(data_dicts, ylabels=None, stds=False, ylims=[0, 1], **kwargs):
    '''
    use plot_bar_dict but on sub axes
    data_dicts: dict of data dictionaries
    '''
    # get the max value in the plots
    for plot in data_dicts.values():
        for bar_value in plot.values():
            if isinstance(bar_value, dict):
                for bar_avg_and_std in bar_value.values():
                    ylims[1] = max(ylims[1], bar_avg_and_std)
                    ylims[0] = min(ylims[0], bar_avg_and_std)
            else:
                ylims[1] = max(ylims[1], bar_value)
                ylims[0] = min(ylims[0], bar_value)
    print(ylims)
    # make subplots
    fig, axs = plt.subplots(1, len(data_dicts))
    if len(data_dicts) == 1:
        axs = [axs]
    for i, key in enumerate(data_dicts):
        ylabel = ylabels[i] if ylabels is not None else None
        plot_bar_dict(data_dicts[key], stds=stds, ax=axs[i], ylabel=ylabel, ylim=ylims)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay
    from clime import data, model

    # ax = plt.gca()
    _, [ax1, ax2] = plt.subplots(2)
    # get dataset
    train_data = data.get_moons()
    train_data = data.unbalance(train_data,[1,0.5])
    clf = model.SVM(train_data)
    plot_decision_boundary(clf, train_data, ax=ax1)
    plot_classes(train_data, ax=ax1)

    clf_bal = model.SVM_balance_boundary(train_data)
    plot_decision_boundary(clf_bal, train_data, ax=ax2)
    plot_classes(train_data, ax=ax2)

    plt.show()
