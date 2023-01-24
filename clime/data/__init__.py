from .datasets import *
from .balance import *

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'moons': sample_dataset_to_proportions(get_moons),
    'guassian': sample_dataset_to_proportions(get_gaussian),
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}