# author: Matt Clifford <matt.clifford@bristol.ac.uk>

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
    dict_ = dict(reversed(dict_.items()))  # want eval first
    keys, values = zip(*dict_.items())
    dict_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return dict_permutations

def get_one_dict_permutation(dict_):
    ''' for use with pytest as all permutations is too computationally intense
    this way just makes sure we test each method at least once but not against all combinations'''
    combos = []
    generic = {}
    for name, methods in dict_.items():
        generic[name] = methods[0]

    for name, methods in dict_.items():
        for method_ in methods:
            opt = generic.copy()
            opt[name] = method_
            combos.append(opt)
    return combos

def check_unique(values, same_values, list_of_diff_keys, key):
    '''
    check whethere a list of values are all the same, and add to the correct list store
    '''
    try:
        set(values)
    except:
        values = [str(v) for v in values]
    if len(set(values)) == 1:  # mean no unique items
        same_values[key] = values[0]
    else:
        list_of_diff_keys.append(key)

def get_opt_differences(opts):
    '''
    given a list of different pipeline options sort out what is the same and what
    is different, this is useful for plots etc.
    '''
    # give empty name for a single option
    if len(opts) == 1:
        return opts[0].copy(), [0]
    if isinstance(opts, dict):
        return opts.copy(), [0]
    # get keys that all have the same or different values
    same_values = {}
    list_of_diff_keys = []
    diff_inner_keys = {}
    for key in opts[0]:
        if isinstance(opts[0][key], dict):
            # get items out of inner dict e.g. 'data_params' dict
            list_of_diff_inner_keys = []
            for key2 in opts[0][key]:
                values = [opt[key][key2] for opt in opts]
                check_unique(values, same_values, list_of_diff_inner_keys, key2)
            diff_inner_keys[key] = list_of_diff_inner_keys
        else:
            values = [opt[key] for opt in opts]
            check_unique(values, same_values, list_of_diff_keys, key)
    # get the values of the differences in opts as a unique name for each
    diff_values = []
    for i, opt in enumerate(opts):
        diff_values.append({})
        # get all the normal items
        for key in list_of_diff_keys:
            diff_values[i][key] = opt[key]
        # get all the inner keys (e.g. from 'data_params' dict)
        for key, item in diff_inner_keys.items():
            for key2 in item:
                diff_values[i][key2] = opt[key2]
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
