# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from clime import data

DATA_KEYS = ['X', 'y']

def test_sample_numbers_equal_classes():
    class_samples = [100, 100]
    assert_class_sample_num(class_samples)

def test_sample_numbers_unequal_classes():
    class_samples = [40, 160]
    assert_class_sample_num(class_samples)

def assert_class_sample_num(class_samples):
    class_samples = [40, 160]
    for dataset in [data.sample_dataset_to_proportions(data.get_moons),
                    data.sample_dataset_to_proportions(data.get_gaussian)]:
        sampled_data, _ = dataset(class_samples)
        assert len(sampled_data['y']) == sum(class_samples)

def test_correct_dict_keys():
    for dataset in data.AVAILABLE_DATASETS:
        sampled_data, _ = data.AVAILABLE_DATASETS[dataset](class_samples=[40, 160],
                                                           percentage=1)
        assert set(DATA_KEYS).issubset(set(list(sampled_data.keys())))

def test_correct_data_types():
    for dataset in data.AVAILABLE_DATASETS:
        sampled_data, _ = data.AVAILABLE_DATASETS[dataset](class_samples=[40, 160],
                                                           percentage=1)
        for key in DATA_KEYS:
            assert type(sampled_data[key]) == np.ndarray

def test_both_splits_describe_the_same_features():
    '''
    B18: proportional_split only carried keys that are per-instance numpy arrays, so a
    loader that set feature_names before the split kept them in the train split and lost
    them from the test split, which then silently picked up generic names. Explanations
    are labelled from the test split, so the names were never seen where they were wanted.
    '''
    for dataset in data.AVAILABLE_DATASETS:
        train_data, test_data = data.AVAILABLE_DATASETS[dataset](class_samples=[40, 160],
                                                                 percentage=1)
        train_names = list(data.utils.checkers.check_data_dict(train_data)['feature_names'])
        test_names = list(data.utils.checkers.check_data_dict(test_data)['feature_names'])
        assert train_names == test_names, f'{dataset}: splits disagree about features'
        assert len(test_names) == test_data['X'].shape[1]


def test_named_datasets_keep_their_names():
    '''the same bug, in the form it was noticed: real names replaced by generic ones'''
    for dataset in ['Breast Cancer', 'Wine', 'Iris', 'Banknote Authentication']:
        for split in data.AVAILABLE_DATASETS[dataset]():
            names = data.utils.checkers.check_data_dict(split)['feature_names']
            assert not any(str(n).startswith('feature ') for n in names), dataset


### write tests to check raise error with check_data_dict
