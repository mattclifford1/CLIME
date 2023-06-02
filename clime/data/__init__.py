from .processing.synthetic_datasets import sample_dataset_to_proportions
from .loaders.gaussian import get_gaussian
from .loaders.sklearn_synthetic import get_moons, get_circles, get_blobs
from .loaders.costcla import costcla_dataset
from .loaders.diabetes import get_diabetes_indian
from .loaders.sonar_rocks import get_sonar
from .loaders.sklearn_toy import get_breast_cancer, get_wine, get_iris
from .processing.balance import get_proportions_and_sample_num, unbalance_undersample, balance_oversample
from .processing.downsample_data import shuffle_dataset, proportional_downsample, proportional_split
from .utils.checkers import check_data_dict, get_generic_feature_names
from .processing.normalise import normaliser

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'Gaussian': sample_dataset_to_proportions(get_gaussian),
    'moons': sample_dataset_to_proportions(get_moons),
    'circles': sample_dataset_to_proportions(get_circles),
    'blobs': sample_dataset_to_proportions(get_blobs),
    'breast cancer': get_breast_cancer,
    'iris': get_iris,
    'wine': get_wine,
    'diabetes pima indian': get_diabetes_indian,
    'sonar rocks vs mine': get_sonar,
    'credit scoring 1': costcla_dataset('CreditScoring_Kaggle2011_costcla'),
    'credit scoring 2': costcla_dataset('CreditScoring_PAKDD2009_costcla'),
    'direct marketing': costcla_dataset('DirectMarketing_costcla'),
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}
