'''
utils to check and make sure the data is in the correct format for the pipeline
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np

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
    data['y'] = _as_integer_labels(data['y'])
    return data


def _as_integer_labels(y):
    '''
    Normalise class labels to an integer array.

    Several loaders build y with pandas and hand back dtype=object holding python ints
    (Sonar, Abalone Gender, Ionosphere). scikit-learn used to accept that; from 1.2 it
    reports "Unknown label type: unknown" and refuses to fit. Cast centrally so every
    loader - including any added later - is covered, but only when the values really are
    integral, so a genuinely non-integer target still fails loudly rather than being
    silently truncated.
    '''
    y = np.asarray(y)
    if y.dtype.kind in 'iu':
        return y
    try:
        as_float = y.astype(np.float64)
    except (TypeError, ValueError):
        return y            # not numeric at all - leave it for the caller to deal with
    if np.all(np.isfinite(as_float)) and np.all(as_float == np.round(as_float)):
        return as_float.astype(np.int64)
    return y

def get_generic_feature_names(data):
    '''
    add generic feature names to data
    '''
    if isinstance(data['X'], list):
        num_features = len(data['X'][0])
    else:
        num_features = data['X'].shape[1]
    names = [f'feature {i}' for i in range(num_features)]
    return names
