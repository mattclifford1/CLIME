'''
Get datasets from the costcla package
    - credit scoring and direct marketing
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import os
import pandas as pd
import clime

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

class costcla_dataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, **kwargs):
        # get dataset
        data = _get_costcla_dataset()
        # shuffle the dataset
        data = clime.data.shuffle_dataset(data)
        # reduce the size of the dataset
        data = clime.data.proportional_downsample(data, **kwargs)
        # split into train, test
        train_data, test_data = clime.data.proportional_split(data, size=0.8)
        return train_data, test_data



def _get_costcla_dataset(dataset="CreditScoring_Kaggle2011_costcla"):
    '''
    load the costcla csv dataset files
    available datasets:
        - CreditScoring_Kaggle2011_costcla
        - CreditScoring_PAKDD2009_costcla
        - DirectMarketing_costcla
    '''
    data = {}
    csvs = ['X', 'y', 'cost_matrix']
    # read and store all csv data
    for csv in csvs:
        df = pd.read_csv(os.path.join(CURRENT_FILE, 'datasets', dataset, f'{csv}.csv'))
        # split into train and test
        data[csv] = df.to_numpy()
        if data[csv].shape[1] == 1:
            data[csv] = data[csv].ravel()
    # add name and description
    with open(os.path.join(CURRENT_FILE, 'datasets', dataset, 'description.txt'), 'r') as f:
        data['description'] = f.read()
    return data


if __name__ == '__main__':
    train_data, test_data = _get_costcla_dataset()
    print(train_data['X'].shape)
    print(train_data['y'].shape)
