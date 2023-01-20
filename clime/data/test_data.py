# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from clime import data

def test_sample_numbers_equal_classes():
    class_samples = [100, 100]
    assert_class_sample_num(class_samples)

def test_sample_numbers_unequal_classes():
    class_samples = [40, 160]
    assert_class_sample_num(class_samples)

def assert_class_sample_num(class_samples):
    class_samples = [40, 160]
    for dataset in data.AVAILABLE_DATASETS.keys():
        sampled_data = data.AVAILABLE_DATASETS[dataset](class_samples)
        assert len(sampled_data['y']) == sum(class_samples)
