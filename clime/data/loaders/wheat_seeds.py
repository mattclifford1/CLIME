'''
Wheat seed type predcition dataset (3 classes)
UCI dataset: https://archive.ics.uci.edu/ml/datasets/seeds#
instances: 210
attributes: 7
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_wheat_seeds(**kwargs):
    '''
    we remove class 3 and make it a binary problem
    '''
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'wheat_seeds', 'data.csv'), header=None)
    df.drop(df[df[7] == 3].index, inplace=True)
    df = df.replace({7: {2: 0}})
    data['y'] = df.pop(7).to_numpy()  # type: ignore
    data['X'] = df.to_numpy()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'wheat_seeds', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    data['feature_names'] = ['area',
                             'perimeter',
                             'compactness',
                             'length of kernel',
                             'width of kernel',
                             'asymmetry coefficient',
                             'length of kernel groove']
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split(  # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data
