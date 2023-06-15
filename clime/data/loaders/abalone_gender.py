'''
abalone gender UCI dataset: https://archive.ics.uci.edu/ml/datasets/Abalone
instances: 4177
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_abalone(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'abalone', 'data.csv'), header=None)
    df.drop(df[df[0] == 'I'].index, inplace=True)
    df = df.replace({0: {'M': 0, 'F': 1}})
    data['y'] = df.pop(0).to_numpy()  # type: ignore
    data['X'] = df.to_numpy()
    data['feature_names'] = ['Length',
                             'Diameter',
                             'Height',
                             'Whole weight',
                             'Shucked weight',
                             'Viscera weight',
                             'Shell weight',
                             'Rings']
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'abalone', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split(  # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data
