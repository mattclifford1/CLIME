'''
sample synthetic datasets (e.g. moons/gaussian) to the desired class proportions
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import clime

class sample_dataset_to_proportions():
    '''
    dataset wrapper to get desired unbalanced classes from any dataset
    via undersampling the minority class

    train_data and test_data both have the same class samples
    '''
    def __init__(self, dataset):
        '''
        inputs:
            - dataset: which dataset to sample from
        '''
        self.dataset = dataset

    def get_data(self, test_set=False, **kwargs):
        data = self.dataset(self.total_samples, test=test_set, **kwargs)
        # undersample to get correct class proportions according to class_samples
        data = clime.data.balance.unbalance_undersample(data, self.class_samples)
        return data

    def __call__(self, class_samples=[5, 10], test_set=False, **kwargs):
        '''
        - class_samples: how many points to samples in each class eg. [30, 50]
        '''
        self.class_samples = class_samples
        self.total_samples, _ = clime.data.get_proportions_and_sample_num(self.class_samples)
        # sample the datasets
        train_data = self.get_data(test_set=False, **kwargs)
        test_data = self.get_data(test_set=True, **kwargs)
        return train_data, test_data
