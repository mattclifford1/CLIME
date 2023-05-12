'''
Generate toy data from the breast cancer dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from sklearn.datasets import load_breast_cancer
import clime


def get_breast_cancer(**kwargs):
    '''
    sample from the half moons data distribution
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


if __name__ == '__main__':
    get_breast_cancer()