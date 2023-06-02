'''
Generate toy data from the breast cancer dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import clime


def get_breast_cancer(**kwargs):
    '''
    breast cancer dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_breast_cancer()
    data = {'X': data.data, 'y': data.target}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # reduce the size of the dataset
    # data = clime.data.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data

def get_wine(**kwargs):
    '''
    wine dataset (0 vs 1,2)
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_wine()
    # convert to binary datatset (0 vs 1,2)
    y = data.target
    y[np.where(y>1)] = 1
    data = {'X': data.data, 'y': y}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # reduce the size of the dataset
    # data = clime.data.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data

def get_iris(**kwargs):
    '''
    iris dataset (0 vs 1,2)
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_iris()
    # convert to binary datatset (0 vs 1,2)
    y = data.target
    y[np.where(y > 1)] = 1
    data = {'X': data.data, 'y': y}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # reduce the size of the dataset
    # data = clime.data.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data


if __name__ == '__main__':
    get_breast_cancer()