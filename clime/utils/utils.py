# author: Jonny Erskine
# email: jonathan.erskine@bristol.ac.uk

# author2: Matt Clifford
# email2: matt.clifford@bristol.ac.uk
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
