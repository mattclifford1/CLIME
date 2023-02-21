'''
Make the German credit dataset in the CLIME format
LINKS:
    - https://www.kaggle.com/datasets/uciml/german-credit
    - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd

def main():
    current_file = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_file, 'datasets', 'german_credit_data.csv')
    df = pd.read_csv(dataset_file)
    print(df.head())



if __name__ == '__main__':
    main()
