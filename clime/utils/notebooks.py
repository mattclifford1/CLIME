import multiprocessing
import ipywidgets
from IPython.display import display, Javascript
import clime

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
    model_stats = {}
    for i, opts in enumerate(opts_permutations):
        s, m = clime.pipeline.run_pipeline(opts, parallel_eval=True)
        scores[str(labels[i])] = s
        model_stats[str(labels[i])] = m
    # get plot details
    if 'evaluation' in list(title.keys()):
        ylabel = title['evaluation']
        title.pop('evaluation')
    else:
        ylabel = 'Explainer Evaluation'
    # plot
    clime.utils.plots.plot_bar_dict(scores, title=title, ylabel=ylabel)


def disp_section_name(section, data_store):
    return f"{section}: {get_config(data_store)[section]}"

def run_vis_pipeline(data_store):
    ############## DATA #################
    # get dataset
    train_data, test_data = clime.pipeline.construct.run_section('dataset', get_config(data_store), class_samples=get_config(data_store)['class samples'])
    # balalance data
    dataset_bal = clime.pipeline.construct.run_section('dataset rebalancing', get_config(data_store), data=train_data)
    # display datasets
    datasets = {disp_section_name('dataset', data_store): train_data,
                disp_section_name('dataset rebalancing', data_store): dataset_bal}
    # clime.utils.plot_data_dict(datasets)


    ############## MODEL #################
    # train model
    clf = clime.pipeline.construct.run_section('model', get_config(data_store), data=dataset_bal)
    # balance model
    clf_bal = clime.pipeline.construct.run_section('model balancer',
                                    get_config(data_store),
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
        disp_section_name('model', data_store): clime.pipeline.construct.get_avg_evaluation(get_config(data_store), clf, test_data, run_parallel=True),
        disp_section_name('model balancer', data_store):  clime.pipeline.construct.get_avg_evaluation(get_config(data_store), clf_bal, test_data, run_parallel=True)
    }
    title = f"{get_config(data_store)['evaluation']} avg of all {get_config(data_store)['explainer']}"
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
