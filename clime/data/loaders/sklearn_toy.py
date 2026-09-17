'''
Generate toy data from the breast cancer dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
import clime


def get_breast_cancer(**kwargs):
    '''
    breast cancer dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_breast_cancer()
    data = {'X': data.data, 'y': data.target,
            'feature_names': [str(n) for n in data.feature_names]}
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
    data = {'X': data.data, 'y': y,
            'feature_names': [str(n) for n in data.feature_names]}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # reduce the size of the dataset
    # data = clime.data.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data

def get_iris(**kwargs):
    '''
    iris dataset (0,2 vs 1)
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_iris()
    # convert to binary datatset (0 vs 1,2)
    y = data.target
    y[np.where(y == 2)] = 0
    data = {'X': data.data, 'y': y}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # add the feature names
    data['feature_names'] = ['Sepal length',
                             'Sepal width',
                             'Petal length',
                             'Petal width']
    # reduce the size of the dataset
    # data = clime.data.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data


def get_digits_3v8(**kwargs):
    '''
    handwritten digits, 3 vs 8, as 8x8 = 64 raw pixel features

    The only dataset here whose features have a spatial layout, which lets an explanation
    be shown as an image rather than a bar chart. 3 against 8 is the useful pair: they
    differ in a small localised region (the left of the two loops), so a linear black
    box's coefficients are spatially concentrated and a surrogate that puts its weight
    elsewhere is visibly, not just numerically, wrong.

    Pixels are kept in grid order and none are dropped, including the border pixels that
    are constant across the dataset - the 8x8 layout is what makes the figure legible, and
    a feature of zero variance carries no explanation weight anyway.

    returns:
        - data: dict containing 'X', 'y'
    '''
    digits = load_digits()
    keep = np.isin(digits.target, [3, 8])
    X = digits.data[keep]
    y = (digits.target[keep] == 8).astype(int)      # class 1 = the digit 8
    data = {'X': X, 'y': y,
            'feature_names': [f'pixel {i//8},{i % 8}' for i in range(X.shape[1])]}
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)
    # split into train, test
    train_data, test_data = clime.data.proportional_split(data, size=0.8)
    return train_data, test_data


if __name__ == '__main__':
    get_breast_cancer()
