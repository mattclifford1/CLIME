'''
plotting functions and helpers for graphs and notebooks
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import scipy
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import numpy as np

# plot colours
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
cm = plt.cm.RdBu
scatter_point_size = 200
font_size = 30
ticks_size = 24
ticks_size_small = 20

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

    ax.scatter(x1, x2, s=scatter_point_size,
               c=data['y'], cmap=cm_bright, edgecolors="k", alpha=0.4,
               linewidths=2)
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

def plot_decision_boundary(clf, data, ax=None, dim_reducer=None, labels=True):
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

    # yhat = clf.predict(flat_grid)
    
    # reshape the predictions back into a grid
    zz = yhat.reshape(xgrids[0].shape)

    # plot the grid of x, y and z values as a surface
    c = ax.contourf(xgrids[0], xgrids[1], zz, cmap=cm,
                    vmin=0, vmax=1, alpha=0.7)
    c.set_clim(0, 1)

    # add a legend, called a color bar
    cbar = plt.colorbar(c, ticks=[0, 0.5, 1])
    if labels == True:
        cbar.ax.tick_params(labelsize=ticks_size)
        cbar.ax.set_ylabel('Probability', size=ticks_size)

    # set labels
    if labels == True:
        if dim_reducer != None:
            ax.set_xlabel('PCA Component 1', fontsize=font_size)
            ax.set_ylabel('PCA Component 2', fontsize=font_size)
        else:
            ax.set_xlabel('Feature 1', fontsize=font_size)
            ax.set_ylabel('Feature 2', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=ticks_size_small)


    if show == True:
        plt.show()

def plot_query_points(query_points, ax, dim_reducer=None):
    '''
    point eval points as black, are a list of values
    '''
    ax, _ = _get_axes(ax)
    q_nums = list(range(len(query_points)))
    # get lims 
    qs = np.array(query_points)
    lim = np.max(qs[:, 1]) - np.min(qs[:, 1])
    for num, q in zip(q_nums, query_points):
        if dim_reducer == None:
            q0= q[0]
            q1 = q[1]
        else:
            q_ = dim_reducer.transform(q.reshape(1, -1)).reshape(-1, 1)
            q0 = q_[0]
            q1 = q_[1]
        ax.scatter(q0, q1, s=scatter_point_size*1.5, color='yellow',
                   edgecolors="k")  # , zorder=100)
        if num % 2 == 0:
            ax.annotate(num, (q0, q1), 
                        xytext=(q0, q1+0.3),
                        # xytext=(q0, q1+lim/5),
                        ha='center', 
                        # va='bottom', 
                        fontsize=font_size,
                        color="yellow",
                        path_effects=[pe.withStroke(linewidth=4, foreground="black")])  # , zorder=100)
            # ax.annotate(num, (q0, q1), ha='right', va='bottom', fontsize=font_size)#, zorder=100)

    # set lims to focus on query points
    qs = np.array(query_points)
    # ax.set_xlim([np.min(qs[:, 0])-1, np.max(qs[:, 0])+1])
    # ax.set_ylim([np.min(qs[:, 1])-1, np.max(qs[:, 1])+1])

'''
helper functions
'''
def _metric_range(metric_name):
    '''
    fixed display range of a metric, or None if the limits should come from the data
    (see clime.evaluation.METRIC_RANGES)
    '''
    if metric_name is None:
        return None
    import clime
    return clime.evaluation.METRIC_RANGES.get(metric_name, None)

def _data_ylims(values, pad=0.05):
    '''axis limits that actually contain the data, with a small margin'''
    values = np.asarray(list(values), dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0, 1]
    low, high = float(values.min()), float(values.max())
    if high == low:
        margin = max(abs(high)*0.1, 0.5)
    else:
        margin = (high - low)*pad
    return [low - margin, high + margin]

def _get_ylims(values, metric_name=None):
    '''
    limits for an axis showing 'metric_name'. bounded metrics (e.g. fidelity) keep a
    fixed scale so plots are comparable; unbounded ones (log loss) or ones with a tiny
    range (Brier score, ~0.02) are scaled to the data - a hard coded [0, 1] makes them
    unreadable
    '''
    fixed = _metric_range(metric_name)
    if fixed is not None:
        return list(fixed)
    return _data_ylims(values)

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

def plot_line_graphs(data_dict, ylabels=None, ylims=None, extra_lines=False):
    keys = list(data_dict.keys())
    # gather every score belonging to each metric, so that subplots showing the same
    # metric share a scale and stay comparable with each other
    scores_per_metric = {}
    for i, key in enumerate(keys):
        ylabel = ylabels[i] if ylabels is not None else 'Evaluation Score'
        for item in data_dict[key].values():
            scores_per_metric.setdefault(ylabel, []).extend(list(item['scores']))
    # now plot
    fig, axs = plt.subplots(1, len(data_dict))
    if len(data_dict) == 1:
        axs = [axs]
    for i, key in enumerate(keys):
        ylabel = ylabels[i] if ylabels is not None else 'Evaluation Score'
        lims = ylims if ylims is not None else _get_ylims(scores_per_metric[ylabel], ylabel)
        plot_multiple_lines(data_dict[key], axs[i], ylims=lims, ylabel=ylabel, extra_lines=extra_lines)

    fig.tight_layout()

def plot_mean_std_graphs(data_dict, ylabel=None, ylims=None, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    if ylims is None:
        ylims = _get_ylims([np.array(item) for item in data_dict.values()], ylabel)
    for key, item in data_dict.items():
        x = list(range(len(item[0])))
        scores = np.array(item)
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        p = ax.plot(x, mean,  label=key)
        ax.fill_between(x, mean-std, mean+std, color=p[0].get_color(), alpha=0.3)
        ax.set_xlabel('Query Point', fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_ylim(ylims)

    if len(data_dict) > 1:
        ax.legend()

def plot_line_graphs_on_one_graph(data_dict, ylabel=None, ylims=None, ax=None, query_values=True, model=None):
    # limits from the metric if it is bounded, otherwise from the data itself
    if ylims is None:
        ylims = _get_ylims([item['scores'] for item in data_dict.values()], ylabel)
    # now plot
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    if ylabel is None:
        ylabel = 'Evaluation Score'
    
    line_style = '-'
    for key, item in data_dict.items():
        if 'eval_points' in item.keys() and query_values == True:
            x = [f[0] for f in item['eval_points']]
        else:
            x = list(range(len(item['scores'])))
        ax.plot(x, item['scores'],  label=key,
                linewidth=10, linestyle=line_style)
        if line_style == '-':
            # uncomment to show dashed on every other
            line_style = '-'
            # line_style = '--'
        else:
            line_style = '-'
        ax.plot(x, item['scores'], 'ko',  label=None,
                markersize=20,
                markeredgecolor='black',
                markerfacecolor='yellow')
        
    ax.set_xlabel('Query Point', fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    xticks = []
    for i in x:
        if i % 2 == 0:
            xticks.append(i)
    if len(xticks) < 15:
        ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)

    ax.set_ylim(ylims)
    if len(data_dict) > 1:
        ax.legend(fontsize=font_size) #, loc='lower right')
    # fig.tight_layout()


def plot_heatmaps(scores, axs=False, fig=None, ylabels=None, pca=None):
    '''scores is from pipeline eval so contains:
        - eval_points
        - scores
    which is what we will construct the heat map from
    '''
    if axs == False:
        ax_x = len(scores.keys())
        ax_y = 1
        # ax_y = int(np.ceil(num_plots/ax_x))
        fig, axs = plt.subplots(ax_y, ax_x)
        if ax_y == 1 and ax_x == 1:
            axs = [axs]
    else:
        ax_x = 1
        ax_y = len(axs)

    # gather the scores for each metric first, so that heatmaps of the same metric
    # share a colour scale (and so are comparable) but are not squashed onto a
    # hard coded [0, 1] that hides small ranges such as Brier score
    scores_per_metric = {}
    count = 0
    for num, runs in scores.items():
        for title, run_data in runs.items():
            ylabel = ylabels[count] if ylabels is not None else None
            scores_per_metric.setdefault(ylabel, []).extend(list(run_data['scores']))
            count += 1

    count = 0
    for num, runs in scores.items():
        for title, run_data in runs.items():
            eval_points = np.array(run_data['eval_points'])
            if pca != None:
                eval_points = pca.transform(eval_points)
            x = eval_points[:, 0]
            y = eval_points[:, 1]
            z = run_data['scores']
            ylabel = ylabels[count] if ylabels is not None else None
            vmin, vmax = _get_ylims(scores_per_metric[ylabel], ylabel)
            heatmap = _heatmap_interpolate(x, y, z, axs[count], vmin=vmin, vmax=vmax)
            axs[count].set_title(title)
            plt.colorbar(heatmap, ax=axs[count], label=ylabel)
            count += 1

    fig.tight_layout()


def _heatmap_interpolate(x, y, z, ax=None, aspect=1, cmap=plt.cm.rainbow, vmin=None, vmax=None):
    '''
    vmin/vmax bound the colour scale and clip the interpolation to it. Pass the
    limits for the metric being plotted - clipping to [0, 1] regardless of metric
    flattens anything with a small range into a single colour
    '''
    # Create regular grid
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(
        y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate missing data
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
    # clip interpolation to the colour scale
    if vmin is not None and vmax is not None:
        zi = np.clip(zi, vmin, vmax)
        kwargs = {'vmin': vmin, 'vmax': vmax}
    else:
        kwargs = {}

    if ax == None:
        _, ax = plt.subplots(figsize=(6, 6))

    hm = ax.imshow(zi, interpolation='nearest', cmap=cmap,
                    extent=[x.min(), x.max(), y.max(), y.min()], **kwargs)
    ax.scatter(x, y, s=0.5)
    ax.set_aspect(aspect)
    return hm


def plot_clfs(data_dict, ax_x=2, title=True, axs=False, fig=None, labels=True):
    '''
    data dict has keys:
        - model
        - data
    '''
    if axs == False:
        num_plots = len(data_dict.keys())
        ax_y = int(np.ceil(num_plots/ax_x))
        if num_plots == 1:
            ax_x = 1
        fig, axs = plt.subplots(ax_y, ax_x)
        if ax_y == 1 and ax_x == 1:
            axs = [axs]
    else:
        ax_x = 1
        ax_y = len(axs)

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
            # flip plot if query points are decending
            if pca != None and 'query_points' in data_dict[key].keys():
                start = data_dict[key]['query_points'][0]
                end = data_dict[key]['query_points'][-1]
                start = pca.transform(start.reshape(1, -1)).reshape(-1, 1)
                end = pca.transform(end.reshape(1, -1)).reshape(-1, 1)
                if start[0] > end[0]:
                    pca = flip_pca(pca)
            # plot all
            plot_decision_boundary(model, data, ax=ax, dim_reducer=pca, labels=labels)
            plot_classes(data, ax, dim_reducer=pca)
            if 'query_points' in data_dict[key].keys():
                plot_query_points(data_dict[key]['query_points'], ax, dim_reducer=pca)
            if title is True:
                ax.set_title(key)
            count += 1
    fig.tight_layout()
    return pca

class flip_pca:
    def __init__(self, pca):
        self.pca = pca

    def transform(self, X):
        X_ = self.pca.transform(X)
        X_[:, 0] = - X_[:, 0]
        return X_
    
    def inverse_transform(self, X_):
        X_[:, 0] = - X_[:, 0]
        return self.pca.inverse_transform(X_)

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

def _bar_values(plot):
    '''pull every plottable number out of one bar plot's data dict'''
    values = []
    for bar_value in plot.values():
        if isinstance(bar_value, dict):
            # only the bar height matters for the scale, not the error bar
            for name in ['avg', 'result']:
                if name in bar_value and isinstance(bar_value[name], (float, int)):
                    values.append(bar_value[name])
        elif isinstance(bar_value, (float, int)):
            values.append(bar_value)
    return values

def plot_multiple_bar_dicts(data_dicts, title=None, ylabels=None, ylims=None, **kwargs):
    '''
    use plot_bar_dict but on sub axes
    data_dicts: dict of data dictionaries
    '''
    keys = list(data_dicts.keys())
    # gather the bar heights for each metric so subplots sharing a metric share a scale
    values_per_metric = {}
    for i, key in enumerate(keys):
        ylabel = ylabels[i] if ylabels is not None else None
        values_per_metric.setdefault(ylabel, []).extend(_bar_values(data_dicts[key]))
    # make subplots
    fig, axs = plt.subplots(1, len(data_dicts))
    if len(data_dicts) == 1:
        axs = [axs]
    for i, key in enumerate(keys):
        ylabel = ylabels[i] if ylabels is not None else None
        lims = ylims if ylims is not None else _get_ylims(values_per_metric[ylabel], ylabel)
        plot_bar_dict(data_dicts[key], ax=axs[i], ylabel=ylabel, ylim=lims)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()


def plot_multiple_lines(data_dict, ax=None, ylims=None, ylabel='Evaluation Score', extra_lines=False, query_values=False):
    '''
    plot multiple line charts
        - data_dict: dictionary with scores/results from pipeline
    '''
    ax, show = _get_axes(ax)
    if ylims is None:
        ylims = _get_ylims([item['scores'] for item in data_dict.values()], ylabel)
    for key, item in data_dict.items():
        if 'eval_points' in item.keys()  and query_values == True:
            x = [f[0] for f in item['eval_points']]
        else:
            x = list(range(len(item['scores'])))
        ax.plot(x, item['scores'],  label=key)
        ax.plot(x, item['scores'], 'ko',  label=None)
        if 'class_weights' in item and extra_lines == True:
            ax.plot(x, item['class_weights'], label='minority class proportion (sampling data)')
            ax.legend()
        if 'majority influence' in item and extra_lines == True:
            ax.plot(x, item['majority influence'],
                    label='minority class evaluation influence (distance weights)')
            ax.legend()
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
