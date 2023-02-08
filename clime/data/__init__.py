from .synthetic_datasets import *
from .german_credit import *
from .costcla import *
from .balance import *
from .downsample_data import *
from .checkers import *

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'moons': sample_dataset_to_proportions(get_moons),
    'guassian': sample_dataset_to_proportions(get_gaussian),
    'credit scoring 1': costcla_dataset('CreditScoring_Kaggle2011_costcla'),
    'credit scoring 2': costcla_dataset('CreditScoring_PAKDD2009_costcla'),
    'direct marketing': costcla_dataset('DirectMarketing_costcla'),
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}
