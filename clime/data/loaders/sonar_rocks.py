'''
Sonar rocks vs mines UCI dataset: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
instances: 208

Each pattern is a set of 60 numbers in the range 0.0 to 1.0. 
Each number represents the energy within a particular frequency band, 
integrated over a certain period of time. 
The integration aperture for higher frequencies occur later in time, 
since these frequencies are transmitted later during the chirp.

The label associated with each record contains the letter "R" if the object is a rock 
and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing 
order of aspect angle, but they do not encode the angle directly.

post processed data: https://www.kaggle.com/datasets/mattcarter865/mines-vs-rocks
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_sonar(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'sonar_rocks_mines', 'data.csv'), header=None)
    df = df.replace({60: {'R': 0, 'M': 1}})
    data['y'] = df.pop(60).to_numpy() # type: ignore
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'sonar_rocks_mines', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = clime.data.shuffle_dataset(data)  # type: ignore
    # split into train, test
    train_data, test_data = clime.data.proportional_split(  # type: ignore
        data, size=0.8)  # type: ignore
    return train_data, test_data
