'''
Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
UCI dataset: https://archive.ics.uci.edu/ml/datasets/Ionosphere
instances: 351
attributes: 34
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_ionosphere(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'Ionosphere', 'data.csv'), header=None)
    df.drop(df[df[0] == 'I'].index, inplace=True)
    df = df.replace({34: {'b': 0, 'g': 1}})
    data['y'] = df.pop(34).to_numpy()  # type: ignore
    data['X'] = df.to_numpy()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'Ionosphere', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split(  # type: ignore
        data, size=0.8)  # type: ignore
    return train_data, test_data
