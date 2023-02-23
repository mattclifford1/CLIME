import multiprocessing
import ast
from tqdm.autonotebook import tqdm
import ipywidgets
from IPython.display import display, Javascript
import clime
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 6)

def run_all(ev):
    display(Javascript('IPython.notebook.execute_cells_below()'))

def get_run_button():
    # button = ipywidgets.widgets.Button(description=f"{'*'*10}RUN PIPELINE{'*'*10}")
    button = ipywidgets.widgets.Button(description=f"RUN PIPELINE")
    button.on_click(run_all)
    display(button)

# set up interactive functionality
def get_sliders(interactive_data_store):
    # class samples
    print('synthetic datasets')
    class_samples = ipywidgets.IntRangeSlider(value=[25, 75],
                                              min=1,
                                              max=200,
                                              description='CLASS SAMPLES (synthetic datasets):',
                                              layout=ipywidgets.Layout(width='100%'),
                                              style={'description_width': 'initial'})
    display(class_samples)
    interactive_data_store['data params']['class_samples'] = class_samples
    # percent data
    print('real datasets')
    percent_data = ipywidgets.FloatSlider(value=0.1,
                                          min=0.01,
                                          max=10,
                                          description='PERCENT DATA (real datasets):',
                                          layout=ipywidgets.Layout(width='100%'),
                                          style={'description_width': 'initial'})
    display(percent_data)
    interactive_data_store['data params']['percent of data'] = percent_data
    print('moons noise')
    percent_data = ipywidgets.FloatSlider(value=0.2,
                                          min=0,
                                          max=4,
                                          description='noise var (moons)',
                                          layout=ipywidgets.Layout(width='100%'),
                                          style={'description_width': 'initial'})
    display(percent_data)
    interactive_data_store['data params']['percent_of_data'] = percent_data
    return interactive_data_store

def get_list_input(interactive_data_store):
    # get input to make lists but as a text input
    print('Gaussian Means')
    means = ipywidgets.Text(value='[[0, 0], [1, 1]]',
                            placeholder='Type something',
                            description='MEANS:',
                            disabled=False)
    display(means)
    interactive_data_store['data params']['gaussian_means'] = means
    print('Gaussian Covs')
    covs = ipywidgets.Text(value='[[[1, 0], [0, 1]],    [[2, 1],[1, 2]]]',
                            placeholder='Type something',
                            description='COVS:',
                            disabled=False)
    display(covs)
    interactive_data_store['data params']['gaussian_covs'] = covs
    return interactive_data_store

def get_toggle(pipeline_section, interactive_data_store):
    toggle = ipywidgets.ToggleButtons(options=list(clime.pipeline.AVAILABLE_MODULES[pipeline_section].keys()),
                                      description=f'{pipeline_section.upper()}:',
                                      )
    display(toggle)
    interactive_data_store[pipeline_section] = toggle
    return interactive_data_store

def get_multiple(pipeline_section, interactive_data_store):
    init_value = [list(clime.pipeline.AVAILABLE_MODULES[pipeline_section].keys())[0]]
    toggle = ipywidgets.SelectMultiple(options=list(clime.pipeline.AVAILABLE_MODULES[pipeline_section].keys()),
                                       value=init_value,
                                       description=f'{pipeline_section.upper()}:',
                                       layout={'width': 'max-content'},
                                       style={'description_width': 'initial'}
                                      )
    display(toggle)
    interactive_data_store[pipeline_section] = toggle
    return interactive_data_store

def get_config(interactive_data_store):
    # need to read results from interactive widgets
    config = {}
    for key, val in interactive_data_store.items():
        # dataset params
        if isinstance(interactive_data_store[key], dict):
            config[key] = {}
            for key2, val2 in interactive_data_store[key].items():
                read_widget2 = val2.value
                if not isinstance(read_widget2, str):
                    config[key][key2] = read_widget2
                else:
                    config[key][key2] = ast.literal_eval(read_widget2)
                if isinstance(config[key][key2], float):
                    config[key][key2] = round(config[key][key2], 3)
            config[key] = [config[key]]
        # other params
        else:
            read_widget = val.value
            if not isinstance(read_widget, tuple) or key == 'class_samples':
                config[key] = [val.value]
            else:
                config[key] = read_widget
            if isinstance(config[key], float):
                config[key] = round(config[key], 3)
    return config

def get_pipeline_widgets():
    data_store = {'data params': {}}
    # which dataset
    data_store = get_toggle('dataset', data_store)
    # dataset params (sliders)
    data_store = get_sliders(data_store)
    # dataset params (text/lists)
    data_store = get_list_input(data_store)
    # which balancing data method
    data_store = get_toggle('dataset rebalancing', data_store)
    # which model to use
    data_store = get_multiple('model', data_store)
    # which model to use
    data_store = get_multiple('model balancer', data_store)
    # which explainer to use
    data_store = get_multiple('explainer', data_store)
    # which evaluation metric to use
    data_store = get_multiple('evaluation metric', data_store)
    # which evaluation points to run
    data_store = get_multiple('evaluation run', data_store)
    return data_store

# running the pipeline
def run_experiments(data_store):
    # get all pipeline combinations
    all_opts = get_config(data_store)
    opts_permutations = clime.utils.get_all_dict_permutations(all_opts)
    title, labels = clime.utils.get_opt_differences(opts_permutations)
    # get plot details
    if 'evaluation metric' in list(title.keys()):
        ylabels = [title['evaluation metric']]*len(opts_permutations)
        title.pop('evaluation metric')
    else:
        ylabels = [label.pop('evaluation metric') for label in labels]
    # run pipelines
    scores = {}
    scores_no_label = {}
    model_stats_ = {}
    clfs = {}
    train_datas = {}
    test_datas = {}
    for i, opts in tqdm(enumerate(opts_permutations), total=len(opts_permutations), desc='Pipeline runs', leave=False):
        result = clime.pipeline.run_pipeline(opts, parallel_eval=True)
        scores[i] = {str(labels[i]): result['score']}
        scores_no_label[i] = result['score']
        model_stats_[i] = {'result': result['model_stats']}
        model_stats_[i] = result['model_stats']
        clfs[i] = result['clf']
        train_datas[i] = result['train_data']
        test_datas[i] = result['test_data']
    return model_stats_, clfs, train_datas, test_datas, title, scores, scores_no_label, ylabels

# plot pipeline and results
def plot_exp_results(inp):
    model_stats_, clfs, train_datas, test_datas, title, scores, scores_no_label, ylabels = inp
    print(f'Params: {title}')
    # plot evaluation graphs
    clime.utils.plots.plot_multiple_bar_dicts(scores, title=title, ylabels=ylabels)
    # visualise pipeline
    return model_stats_, clfs, train_datas, test_datas, scores, scores_no_label

def plot_model_and_stats(inp):
    model_stats_, clfs, train_datas, test_datas, scores, scores_no_label = inp
    # get all train data and models in plotable dict
    model_plots = {}
    for run in clfs:
        model_plots[run] = {'model': clfs[run], 'data': train_datas[run]}
        if 'eval_points' in scores_no_label[run].keys():
            model_plots[run]['query_points'] = scores_no_label[run]['eval_points']
    clime.utils.plots.plot_clfs(model_plots, ax_x=len(model_plots), title=False)
    clime.utils.plot_multiple_bar_dicts(model_stats_)

def disp_section_name(section, data_store):
    return f"{section}: {get_config(data_store)[section]}"
