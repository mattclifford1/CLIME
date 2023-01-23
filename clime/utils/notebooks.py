import ipywidgets
from IPython.display import display, Javascript
import clime

def run_all(ev):
    display(Javascript('IPython.notebook.execute_cells_below()'))

def get_run_button():
    button = ipywidgets.widgets.Button(description="RUN PIPELINE")
    button.on_click(run_all)
    display(button)

# set up interactive functionality
def _get_class_values(class_1, class_2):
    return [class_1, class_2]

def get_sliders(interactive_data_store):
    print('CLASS SAMPLES:')
    class_samples = ipywidgets.interactive(_get_class_values, class_1=(1,200), class_2=(1,200))
    display(class_samples)
    interactive_data_store['class samples'] = class_samples
    return interactive_data_store

def _get_option_value(x):
    return x

def get_drop_down(pipeline_section, interactive_data_store):
    print(f'{pipeline_section.upper()}:')
    dropdown = ipywidgets.interactive(_get_option_value, x=list(clime.pipeline.AVAILABLE_MODULES[pipeline_section].keys()))
    display(dropdown)
    interactive_data_store[pipeline_section] = dropdown
    return interactive_data_store

def get_config(interactive_data_store):
    # need to read results from interactive widgets
    config = {}
    for key in interactive_data_store.keys():
        config[key] = interactive_data_store[key].result
    return config


def run_vis_pipeline(data_store):
    ############## DATA #################
    # get dataset
    dataset_unbal = clime.pipeline.construct.run_section('dataset', get_config(data_store), class_samples=get_config(data_store)['class samples'])

    # balalance data
    dataset_bal = clime.pipeline.construct.run_section('dataset rebalancing', get_config(data_store), data=dataset_unbal)
    # display datasets
    clime.utils.plot_data_dict({'sampled': dataset_unbal, 'rebalanced':dataset_bal})


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
              'model': {'model': clf, 'data': dataset_bal},
              'model balanced': {'model': clf_bal, 'data': dataset_bal}
             }
    clime.utils.plots.plot_clfs(models)


    ############## EXPLAINER #################
    expl = clime.pipeline.construct.run_section('explainer',
                                     get_config(data_store),
                                     black_box_model=clf_bal,
                                     query_point=dataset_bal['X'][0, :])
    print(expl)

    ############## EVALUATION #################
    evalu = clime.pipeline.construct.run_section('evaluation',
                                      get_config(data_store),
                                      expl=expl,
                                      black_box_model=clf_bal,
                                      data=dataset_bal,
                                      query_point=0)
    print(evalu)
