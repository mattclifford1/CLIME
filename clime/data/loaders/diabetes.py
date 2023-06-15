'''
loader for the Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

def get_diabetes_indian(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'diabetes_pima_indians', 'data.csv'))
    data['y'] = df.pop('Outcome').to_numpy()
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'diabetes_pima_indians', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
        data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split( # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data
