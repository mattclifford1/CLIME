'''
take datasets from costcla package https://albahnsen.github.io/CostSensitiveClassification/Datasets.html

costcla only works with python 3.7 or lower and old versions of numpy/scikit-learn
so we need to extract the data locally from it to use with newer versions

NOTE: this script needs to be run with python 3.7 and scikit-learn==0.22
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
from costcla.datasets import load_bankmarketing, load_creditscoring1, load_creditscoring2

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

def save_data_locally(costcla_dataset):
    data = costcla_dataset()
    dataset_dir = os.path.join(CURRENT_FILE, data['name']+'_costcla')
    os.makedirs(dataset_dir, exist_ok=True)
    # features
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df.to_csv(os.path.join(dataset_dir, 'X.csv'), index=False)
    # target varable
    df = pd.DataFrame(data['target'])
    df.to_csv(os.path.join(dataset_dir, 'y.csv'), index=False)
    # costs
    df = pd.DataFrame(data['cost_mat'])
    df.to_csv(os.path.join(dataset_dir, 'cost_matrix.csv'), index=False)
    # dataset description
    with open(os.path.join(dataset_dir, 'description.txt'), 'w') as f:
        f.write(data['DESCR'])

if __name__ == '__main__':
    for dataset in [load_bankmarketing, load_creditscoring1, load_creditscoring2]:
        save_data_locally(dataset)
