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

### write tests to check raise error with check_data_dict
