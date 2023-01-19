from .datasets import *
from .balance import *

AVAILABLE_DATASETS = {
    'moons': get_moons,
    'guassian': get_gaussian,
}

AVAILABLE_DATA_UNBALANCING = {
    'undersampling': unbalance_undersample,
}

AVAILABLE_DATA_BALANCING = {
    'oversampling': balance_oversample,
}
