import multiprocessing
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
                                              description='CLASS SAMPLES (synthetic datasets):')
    display(class_samples)
    interactive_data_store['class samples'] = class_samples
    # percent data
    print('real datasets')
    percent_data = ipywidgets.FloatSlider(value=0.1,
                                          min=0.01,
                                          max=10,
                                          description='PERCENT DATA (real datasets):')
    display(percent_data)
    interactive_data_store['percent of data'] = percent_data
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
                                      )
    display(toggle)
    interactive_data_store[pipeline_section] = toggle
    return interactive_data_store

def get_config(interactive_data_store):
    # need to read results from interactive widgets
    config = {}
    for key, val in interactive_data_store.items():
        read_widget = interactive_data_store[key].value
        if not isinstance(read_widget, tuple) or key == 'class samples':
            config[key] = [interactive_data_store[key].value]
        else:
            config[key] = interactive_data_store[key].value
        if isinstance(config[key], float):
            config[key] = round(config[key], 3)
    return config

def get_pipeline_widgets():
    data_store = {}
    # input class proportions
    data_store = get_sliders(data_store)
    # which dataset
    data_store = get_toggle('dataset', data_store)
    # which balancing data method
    data_store = get_toggle('dataset rebalancing', data_store)
    # which model to use
    data_store = get_multiple('model', data_store)
    # which model to use
    data_store = get_multiple('model balancer', data_store)
    # which explainer to use
    data_store = get_multiple('explainer', data_store)
    # which evaluation to use
    data_store = get_multiple('evaluation', data_store)
    return data_store

def run_experiments(data_store):
    # get all pipeline combinations
    all_opts = get_config(data_store)
    opts_permutations = clime.utils.get_all_dict_permutations(all_opts)
    title, labels = clime.utils.get_opt_differences(opts_permutations)
    # run pipelines
    scores = {}
    model_stats_ = {}
    clfs = {}
    train_datas = {}
    test_datas = {}
    for i, opts in enumerate(opts_permutations):
        score_avg, model_stats, clf, train_data, test_data = clime.pipeline.run_pipeline(opts, parallel_eval=True, return_all=True)
        scores[str(labels[i])] = {str(labels[i]): score_avg}
        model_stats_[str(labels[i])] = model_stats
        clfs[str(labels[i])] = clf
        train_datas[str(labels[i])] = train_data
        test_datas[str(labels[i])] = test_data
    # get plot details
    if 'evaluation' in list(title.keys()):
        ylabels = [title['evaluation']]*len(scores)
        title.pop('evaluation')
    else:
        ylabels = [label.pop('evaluation') for label in labels]
    print(f'Params: {title}')
    # plot evaluation graphs
    clime.utils.plots.plot_multiple_bar_dicts(scores, title=title, ylabels=ylabels, stds=True)
    # visualise pipeline
    return model_stats_, clfs, train_datas, test_datas

def plot_model_and_stats(inp):
    model_stats_, clfs, train_datas, test_datas = inp
    # get all train data and models in plotable dict
    model_plots = {}
    for run in clfs:
        model_plots[run] = {'model': clfs[run], 'data': train_datas[run]}
    print('Model probabilities')
    clime.utils.plots.plot_clfs(model_plots, ax_x=len(model_plots), title=False)
    print('Model train/test statistics')
    clime.utils.plot_multiple_bar_dicts(model_stats_)

def disp_section_name(section, data_store):
    return f"{section}: {get_config(data_store)[section]}"

def run_vis_pipeline(data_store):
    ############## DATA #################
    # get dataset
    all_opts = get_config(data_store)
    opts_permutations = clime.utils.get_all_dict_permutations(all_opts)

    opt = opts_permutations[0]
    train_data, test_data = clime.pipeline.construct.run_section('dataset', opt, class_samples=opt['class samples'])
    # balalance data
    dataset_bal = clime.pipeline.construct.run_section('dataset rebalancing', opt, data=train_data)
    # display datasets
    datasets = {disp_section_name('dataset', data_store): train_data,
                disp_section_name('dataset rebalancing', data_store): dataset_bal}
    # clime.utils.plot_data_dict(datasets)


    ############## MODEL #################
    # train model
    clf = clime.pipeline.construct.run_section('model', opt, data=dataset_bal)
    # balance model
    clf_bal = clime.pipeline.construct.run_section('model balancer',
                                    opt,
                                    model=clf,
                                    data=dataset_bal,
                                    weight=1)
    models = {
              disp_section_name('model', data_store): {'model': clf, 'data': dataset_bal},
              disp_section_name('model balancer', data_store): {'model': clf_bal, 'data': dataset_bal}
             }
    clime.utils.plots.plot_clfs(models)

    ############## EVALUATE with EXPLAINER #####################
    # get average of all evaluation over all points in the data set

    eval_scores = {
        disp_section_name('model', data_store): clime.pipeline.construct.get_avg_evaluation(opt, clf, test_data, run_parallel=True),
        disp_section_name('model balancer', data_store):  clime.pipeline.construct.get_avg_evaluation(opt, clf_bal, test_data, run_parallel=True)
    }
    title = f"{opt['evaluation']} avg of all {opt['explainer']}"
    clime.utils.plots.plot_bar_dict(eval_scores, title=title)

# def eval_explainer(opts, clf, data, query_point):
#
#     ############## EXPLAINER #################
#     expl = clime.pipeline.construct.run_section('explainer',
#                                      get_config(data_store),
#                                      black_box_model=clf_bal,
#                                      query_point=dataset_bal['X'][0, :])
#
#     ############## EVALUATION #################
#     evalu = clime.pipeline.construct.run_section('evaluation',
#                                       get_config(data_store),
#                                       expl=expl,
#                                       black_box_model=clf_bal,
#                                       data=dataset_bal,
#                                       query_point=0)
#     print(evalu)
