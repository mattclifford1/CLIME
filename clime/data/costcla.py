'''
Get datasets from the costcla package
    - credit scoring and direct marketing
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import os
import pandas as pd

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

def get_costcla_dataset(dataset="CreditScoring_Kaggle2011_costcla", **kwargs):
    '''
    load the costcla csv dataset files
    available datasets:
        - CreditScoring_Kaggle2011_costcla
        - CreditScoring_PAKDD2009_costcla
        - DirectMarketing_costcla
    '''
    train_data = {}
    test_data = {}
    csvs = ['X', 'y', 'cost_matrix']

    # read and store all csv data
    for csv in csvs:
        df = pd.read_csv(os.path.join(CURRENT_FILE, 'datasets', dataset, f'{csv}.csv'))
        # split into train and test
        split_point = df.shape[0]//2
        df1 = df.iloc[:split_point, :]
        df2 = df.iloc[split_point:, :]
        train_data[csv] = df1.to_numpy()
        test_data[csv] = df2.to_numpy()

    # add name and description
    for data, name in zip([train_data, test_data], ['train', 'test']):
        with open(os.path.join(CURRENT_FILE, 'datasets', dataset, 'description.txt'), 'r') as f:
            data['description'] = f.read()
            data['name'] = f'{csv}-{name}'

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = get_costcla_dataset()
    print(train_data['X'].shape)
    print(train_data['y'].shape)
