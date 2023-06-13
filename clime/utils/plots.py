'''
plotting functions and helpers for graphs and notebooks
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import numpy as np

# plot colours
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
cm = plt.cm.RdBu
scatter_point_size = 30

def plot_classes(data, ax=None, dim_reducer=None):
    '''
    plot classes in different colour on an axes, duplicate points in the data are enlarged for clarity
    input:
        - data: dictionary with keys 'X', 'y'
    '''
    ax, show = _get_axes(ax)
    if dim_reducer == None:
        x1 = list(data['X'][:, 0])
        x2 = list(data['X'][:, 1])
    else:
        X = dim_reducer.transform(data['X'])
        x1 = list(X[:, 0])
        x2 = list(X[:, 1])
    # count the occurrences of each point
    point_count = Counter(zip(x1, x2))
    # create a list of the sizes, here multiplied by 10 for scale
    size = [scatter_point_size*point_count[(xx1, xx2)] for xx1, xx2 in zip(x1, x2)]

    ax.scatter(x1, x2, s=size, c=data['y'], cmap=cm_bright, edgecolors="k")
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

def plot_decision_boundary(clf, data, ax=None, dim_reducer=None):
    '''
    plot a decision boundary on axes
    input:
        - clf: sklearn classifier object
    '''
    # if dim_reducer != None:
    #     X = dim_reducer.transform(data['X'])
    #     # need to impliment this to encorportant n dimensions
    #     print('non impliment')
    #     return
    
    ax, show = _get_axes(ax)
    # get X from data
    X = data['X']
    if dim_reducer != None:
        X = dim_reducer.transform(X)
    n_features = X.shape[1]

    # define bounds of the domain
    lims = []
    for f in range(n_features):
        min = X[:, f].min()-1
        max = X[:, f].max()+1
        lims.append((min, max))

    # define the x and y scale
    res = 25
    xranges = []
    for f in range(n_features):
        step = (lims[f][1] - lims[f][0])/ res
        xranges.append(np.arange(lims[f][0], lims[f][1], step))

    # create all of the lines and rows of the grid
    xgrids = np.meshgrid(*xranges)
    # flatten each grid to a vector
    flats = []
    for f in range(n_features):
        flats.append(xgrids[f].flatten().reshape(-1, 1))

    # horizontal stack vectors to create x1, x2 input for the model
    flat_grid = np.hstack(flats)
    if dim_reducer != None:
        flat_grid = dim_reducer.inverse_transform(flat_grid)

    # make predictions for the flat_grid
    yhat = clf.predict_proba(flat_grid)
    # keep just the probabilities for class 0
    yhat = yhat[:, 0]

    # reshape the predictions back into a grid
    zz = yhat.reshape(xgrids[0].shape)

    # plot the grid of x, y and z values as a surface
    c = ax.contourf(xgrids[0], xgrids[1], zz, cmap=cm, alpha=0.7)

    # add a legend, called a color bar
    plt.colorbar(c)
    if show == True:
        plt.show()

def plot_query_points(query_points, ax, dim_reducer=None):
    '''
    point eval points as black, are a list of values
    '''
    ax, _ = _get_axes(ax)
    q_nums = list(range(len(query_points)))
    for num, q in zip(q_nums, query_points):
        if dim_reducer == None:
            q0= q[0]
            q1 = q[1]
        else:
            q_ = dim_reducer.transform(q.reshape(1, -1)).reshape(-1, 1)
            q0 = q_[0]
            q1 = q_[1]
        ax.scatter(q0, q1, s=scatter_point_size, color='black')
        ax.annotate(num, (q0, q1))

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

def plot_line_graphs(data_dict, ylabels=None, ylims=[0, 1]):
    # first get the min and max values
    for key, item1 in data_dict.items():
        for key, item2 in item1.items():
            scores = item2['scores']
            for score in scores:
                ylims[1] = max(ylims[1], score)
                ylims[0] = min(ylims[0], score)
    # now plot
    fig, axs = plt.subplots(1, len(data_dict))
    if len(data_dict) == 1:
        axs = [axs]
    for i, key in enumerate(data_dict.keys()):
        if ylabels is not None:
            ylabel = ylabels[i]
        else:
            ylabel = 'Evaluation Score'
        plot_multiple_lines(data_dict[key], axs[i], ylims=ylims, ylabel=ylabel)

    fig.tight_layout()

def plot_mean_std_graphs(data_dict, ylabel=None, ylims=[0, 1], ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    for key, item in data_dict.items():
        x = list(range(len(item[0])))
        scores = np.array(item)
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        p = ax.plot(x, mean,  label=key)
        ax.fill_between(x, mean-std, mean+std, color=p[0].get_color(), alpha=0.3)
        ax.set_xlabel('Query Point')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylims)

    if len(data_dict) > 1:
        ax.legend()

def plot_line_graphs_on_one_graph(data_dict, ylabel=None, ylims=[0, 1], ax=None, query_values=True):
    # first get the min and max values
    for key, item2 in data_dict.items():
        scores = item2['scores']
        for score in scores:
            ylims[1] = max(ylims[1], score)
            ylims[0] = min(ylims[0], score)
    # now plot
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    if ylabel is None:
        ylabel = 'Evaluation Score'
    for key, item in data_dict.items():
        if 'eval_points' in item.keys() and query_values == True:
            x = [f[0] for f in item['eval_points']]
        else:
            x = list(range(len(item['scores'])))
        ax.plot(x, item['scores'],  label=key)
        ax.plot(x, item['scores'], 'ko',  label=None)
        ax.set_xlabel('Query Point')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylims)

    if len(data_dict) > 1:
        ax.legend()
    # fig.tight_layout()

def plot_clfs(data_dict, ax_x=2, title=True):
    '''
    data dict has keys:
        - model
        - data
    '''
    num_plots = len(data_dict.keys())
    ax_y = int(np.ceil(num_plots/ax_x))
    if num_plots == 1:
        ax_x = 1
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
                # get pca with 2 components to visualise in 2D
                pca = PCA(n_components=2, svd_solver='full')
                pca.fit(data['X'])
            else:
                pca = None
            model = data_dict[key]['model']
            if ax_y > 1:
                ax = axs[i][j]
            else:
                ax = axs[i]
            plot_classes(data, ax, dim_reducer=pca)
            plot_decision_boundary(model, data, ax=ax, dim_reducer=pca)
            if 'query_points' in data_dict[key].keys():
                plot_query_points(data_dict[key]['query_points'], ax, dim_reducer=pca)
            if title is True:
                ax.set_title(key)
            count += 1
    fig.tight_layout()

def plot_bar_dict(data_dict, title='', ylabel=None, ax=None, ylim=None):
    if ax is None:
        fig, ax = plt.subplots()
    keys = list(data_dict.keys())
    x_pos = np.arange(len(keys))
    avgs = []
    stds = []
    for key, item in data_dict.items():
        if isinstance(item, dict):
            # get results
            if 'avg' in item.keys():
                avgs.append(item['avg'])
            elif 'result' in item.keys():
                avgs.append(item['result'])
            else:
                raise ValueError(f"plotting dict needs key 'avg' or 'result', got: {item.keys()}")
            # get standard deviations
            if 'std' in item.keys():
                stds.append(item['std'])
            else:
                stds.append(None)
        else:
            avgs.append(item)
            stds.append(None)
    ax.bar(x_pos, avgs, align='center', alpha=0.5)
    # plot error bars
    for pos, avg, std  in zip(x_pos, avgs, stds):
        if std is not None:
            ax.errorbar(pos, avg, std, color='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(keys, rotation=75, ha='right')
    ax.set_title(title)
    if ylabel is None:
        ylabel = ''
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_multiple_bar_dicts(data_dicts, title=None, ylabels=None, ylims=[0, 1], **kwargs):
    '''
    use plot_bar_dict but on sub axes
    data_dicts: dict of data dictionaries
    '''
    # get the max value in the plots
    for plot in data_dicts.values():
        for bar_value in plot.values():
            if isinstance(bar_value, dict):
                for bar_avg_and_std in bar_value.values():
                    if isinstance(bar_avg_and_std, float) or isinstance(bar_avg_and_std, int):
                        ylims[1] = max(ylims[1], bar_avg_and_std)
                        ylims[0] = min(ylims[0], bar_avg_and_std)
            elif isinstance(bar_value, float) or isinstance(bar_value, int):
                ylims[1] = max(ylims[1], bar_value)
                ylims[0] = min(ylims[0], bar_value)
    # make subplots
    fig, axs = plt.subplots(1, len(data_dicts))
    if len(data_dicts) == 1:
        axs = [axs]
    for i, key in enumerate(data_dicts):
        ylabel = ylabels[i] if ylabels is not None else None
        plot_bar_dict(data_dicts[key], ax=axs[i], ylabel=ylabel, ylim=ylims)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

def plot_multiple_lines(data_dict, ax=None, ylims=[0, 1], ylabel='Evaluation Score'):
    '''
    plot multiple line charts
        - data_dict: dictionary with scores/results from pipeline
    '''
    ax, show = _get_axes(ax)
    for key, item in data_dict.items():
        if 'eval_points' in item.keys():
            x = [f[0] for f in item['eval_points']]
        else:
            x = list(range(len(item['scores'])))
        ax.plot(x, item['scores'],  label=key)
        ax.plot(x, item['scores'], 'ko',  label=None)
        ax.set_xlabel('Query Point Value')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylims)

    if len(data_dict) > 1:
        ax.legend()
    if show == True:
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay
    import clime

    params_breast = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}, 
                     'dataset': 'moons', 
                     'dataset': 'breast cancer', 
                     'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (normal)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
    result_breast = clime.pipeline.run_pipeline(params_breast, parallel_eval=True)
    plot_clfs({0: {'data': result_breast['train_data'], 'model':result_breast['clf']}})

    compare_bal = False
    if compare_bal == True:
        params_normal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}, 'dataset': 'moons', 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (normal)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
        params_class_bal = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}, 'dataset': 'moons', 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (cost sensitive sampled)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
        params_SVM = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}, 'dataset': 'moons', 'dataset rebalancing': 'none', 'model': 'SVM', 'model balancer': 'none', 'explainer': 'bLIMEy (cost sensitive sampled)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
    
        params_normal_guass = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[0.5, 0], [0, 0.5]],    [[0.5, 0], [
        0, 0.5]]]}, 'dataset': 'Gaussian', 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (normal)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}
        params_class_bal_guass = {'data params': {'class_samples': (200, 200), 'percent of data': 0.11, 'moons_noise': 0.2, 'gaussian_means': [[0, 0], [1, 1]], 'gaussian_covs': [[[0.5, 0], [0, 0.5]],    [[0.5, 0], [
            0, 0.5]]]}, 'dataset': 'Gaussian', 'dataset rebalancing': 'none', 'model': 'Random Forest', 'model balancer': 'none', 'explainer': 'bLIMEy (cost sensitive sampled)', 'evaluation metric': 'fidelity (local)', 'evaluation run': 'between_class_means'}


        result_noraml = clime.pipeline.run_pipeline(
            params_normal_guass, parallel_eval=True)
        result_bal = clime.pipeline.run_pipeline(
            params_class_bal_guass, parallel_eval=True)




        ax = plt.gca()

        # plot_classes(result['train_data'], ax=ax)
        # plot_decision_boundary(result['clf'], result['train_data'], ax=ax)
        # plot_query_points(result['score']['eval_points'], ax)

        scores = {'class balancing': result_bal['score'],'normal': result_noraml['score']}
        plot_line_graphs_on_one_graph(scores, ylabel='Fidelity (local)', ax=ax)


    plt.show()

    # _, [ax1, ax2] = plt.subplots(2)
    # # get dataset
    # train_data = data.get_moons()
    # train_data = data.unbalance(train_data,[1,0.5])
    # clf = model.SVM(train_data)
    # plot_decision_boundary(clf, train_data, ax=ax1)
    # plot_classes(train_data, ax=ax1)
    #
    # clf_bal = model.SVM_balance_boundary(train_data)
    # plot_decision_boundary(clf_bal, train_data, ax=ax2)
    # plot_classes(train_data, ax=ax2)
    #
    # plt.show()
