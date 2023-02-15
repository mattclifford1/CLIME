# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import itertools
import clime

def input_error_msg(options, section):
    '''
    error message when a module method is not available
    '''
    available_keys = list(clime.pipeline.AVAILABLE_MODULES[section].keys())
    error_msg = f"'{section}' needs to be one of: {list(available_keys)} not: {options[section]}"
    return error_msg

def get_all_dict_permutations(dict_):
    '''
    given a dict with list values, return all the possible permuations of
    single values for each key item
    '''
    keys, values = zip(*dict_.items())
    dict_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return dict_permutations

def get_opt_differences(opts):
    '''
    given a list of different pipeline options sort out what is the same and what
    is different, this is useful for plots etc.
    '''
    if len(opts) == 1:
        return opts[0].copy(), [0]
    if isinstance(opts, dict):
        return opts.copy(), [0]
    # get keys that all have the same or different values
    same_values = {}
    list_of_diff_keys = []
    for key in opts[0]:
        values = [opt[key] for opt in opts]
        if len(set(values)) == 1:  # mean no unique items
            same_values[key] = values[0]
        else:
            list_of_diff_keys.append(key)
    diff_values = []
    for i, opt in enumerate(opts):
        diff_values.append({})
        for key in list_of_diff_keys:
            diff_values[i][key] = opt[key]
    return same_values, diff_values


if __name__ == '__main__':
    all_opts = {
        # 'dataset':             'credit scoring 1',
        'dataset':             ['moons'],
        'class samples':       [[25, 75]],    # only for syntheic datasets
        'percent of data':     [ 0.05],       # for real datasets
        'dataset rebalancing': ['none'],
        'model':               ['SVM', 'Ridge'],
        'model balancer':      ['none', 'boundary adjust'],
        'explainer':           ['bLIMEy (cost sensitive training)'],
        'evaluation':          ['fidelity (local)'],
    }
    all_opts = {
        'a':             [1],
        'b':       [2, 3],    # only for syntheic datasets
        'c':     [4, 5],       # for real datasets
    }

    opts = get_all_dict_permutations(all_opts)
    same_values, diff_values = get_opt_differences(opts)
    print(f'{same_values=}')
    print(f'{diff_values=}')
