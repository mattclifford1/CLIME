from .processing.synthetic_datasets import sample_dataset_to_proportions
from .loaders.gaussian import get_gaussian
from .loaders.sklearn_synthetic import get_moons, get_circles, get_blobs
from .loaders.costcla import costcla_dataset
from .loaders.diabetes import get_diabetes_indian
from .loaders.sonar_rocks import get_sonar
from .loaders.banknote import get_banknote
from .loaders.abalone_gender import get_abalone
from .loaders.ionosphere import get_ionosphere
from .loaders.wheat_seeds import get_wheat_seeds
from .loaders.sklearn_toy import get_breast_cancer, get_wine, get_iris
from .processing.balance import get_proportions_and_sample_num, unbalance_undersample, balance_oversample
from .processing.downsample_data import shuffle_dataset, proportional_downsample, proportional_split
from .utils.checkers import check_data_dict, get_generic_feature_names
from .processing.normalise import normaliser

def _identity_data(data, *args):
    return data

AVAILABLE_DATASETS = {
    'Gaussian': sample_dataset_to_proportions(get_gaussian),
    'Breast Cancer': get_breast_cancer,
    'Banknote Authentication': get_banknote,
    'Pima Indian Diabetes': get_diabetes_indian,
    'Iris': get_iris,
    'Wine': get_wine,
    'Sonar Rocks vs Mines': get_sonar,
    'Abalone Gender': get_abalone,
    'Ionosphere': get_ionosphere,
    'Wheat Seeds': get_wheat_seeds,
    'Credit Scoring 1': costcla_dataset('CreditScoring_Kaggle2011_costcla'),
    'Credit Scoring 2': costcla_dataset('CreditScoring_PAKDD2009_costcla'),
    'Direct Marketing': costcla_dataset('DirectMarketing_costcla'),
    'Moons': sample_dataset_to_proportions(get_moons),
    'Circles': sample_dataset_to_proportions(get_circles),
    'Blobs': sample_dataset_to_proportions(get_blobs),
}

AVAILABLE_DATA_BALANCING = {
    'none': _identity_data,
    'oversampling': balance_oversample,
}
