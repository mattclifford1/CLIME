'''
banknote authentication UCI dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
instances: 1372
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_banknote(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'banknote_authentication', 'data.csv'), header=None)
    data['y'] = df.pop(4).to_numpy()  # type: ignore
    data['X'] = df.to_numpy()
    data['feature_names'] = ['variance of Wavelet Transformed image',
                             'skewness of Wavelet Transformed image',
                             'curtosis of Wavelet Transformed image',
                             'entropy of image']
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'banknote_authentication', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split(  # type: ignore
        data, size=0.8)  # type: ignore
    return train_data, test_data
