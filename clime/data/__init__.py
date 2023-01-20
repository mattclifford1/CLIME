from .datasets import *
from .balance import *

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'moons': get_moons,
    'guassian': get_gaussian,
}

AVAILABLE_DATA_UNBALANCING = {
    'none': _identity_data,
    'undersampling': unbalance_undersample,
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}
