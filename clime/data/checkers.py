'''
utils to check and make sure the data is in the correct format for the pipeline
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

def check_data_dict(data):
    '''
    make sure keys are correct and add feature neames if not given
    '''
    # add feature_names if not given
    if 'feature_names' not in data.keys():
        data['feature_names'] = get_generic_feature_names(data)
    # check required keys in data
    data_keys = ['X', 'y', 'feature_names']
    for key in data_keys:
        if key not in data:
            raise ValueError(f'data dictionary needs to have key {key}')
    return data

def get_generic_feature_names(data):
    '''
    add generic feature names to data
    '''
    names = [f'feature {i}' for i in range(len(data['y']))]
    return names
