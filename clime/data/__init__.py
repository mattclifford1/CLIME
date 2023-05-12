from .synthetic_datasets import sample_dataset_to_proportions
from .gaussian import get_gaussian
from .moons import get_moons
from .costcla import costcla_dataset
from .breast_cancer import get_breast_cancer
from .balance import get_proportions_and_sample_num, unbalance_undersample, balance_oversample
from .downsample_data import shuffle_dataset, proportional_downsample, proportional_split
from .checkers import check_data_dict, get_generic_feature_names

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'moons': sample_dataset_to_proportions(get_moons),
    'Gaussian': sample_dataset_to_proportions(get_gaussian),
    'breast cancer': get_breast_cancer,
    'credit scoring 1': costcla_dataset('CreditScoring_Kaggle2011_costcla'),
    'credit scoring 2': costcla_dataset('CreditScoring_PAKDD2009_costcla'),
    'direct marketing': costcla_dataset('DirectMarketing_costcla'),
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}
