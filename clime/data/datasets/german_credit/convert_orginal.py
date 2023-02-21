'''
convert the german credit original dataset (german.data) to a readable csv
refer to 'german.doc' for details about the dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd

def main():
    current_file = os.path.dirname(os.path.abspath(__file__))
    ''' normal dataset '''
    dataset_file = os.path.join(current_file, 'german.data')
    df = pd.read_table(dataset_file, header=None)
    # loop through and refromat dataset (probs better with a .apply() - but nvmd)
    data_ = {}
    for i, row in df.iterrows():
        str_data = list(row)[0]  # data stored in a single column format
        row_clean = str_data.split(' ')
        data_[i] = row_clean
    cols = ['Status of existing checking account',
            'Duration in month',
            'Credit history',
            'Purpose',
            'Credit amount',
            'Savings account/bonds',
            'Present employment since',
            'Installment rate in percentage of disposable income',
            'Personal status and sex',
            'Other debtors / guarantors',
            'Present residence since',
            'Property',
            'Age in years',
            'Other installment plans',
            'Housing',
            'Number of existing credits at this bank',
            'Job',
            'Number of people being liable to provide maintenance for',
            'Telephone',
            'foreign worker',
            'label']
    df = pd.DataFrame.from_dict(data_,
                                orient='index',
                                columns=cols)
    df.to_csv(os.path.join(current_file, 'german_credit.csv'))

    ''' numeric dataset '''
    dataset_file = os.path.join(current_file, 'german.data-numeric')
    df = pd.read_table(dataset_file, header=None)
    # loop through and refromat dataset (probs better with a .apply() - but nvmd)
    data_ = {}
    for i, row in df.iterrows():
        str_data = list(row)[0]  # data stored in a single column format
        row_clean = []
        for item in str_data.split(' '):
            if item != '':
                row_clean.append(item)
        for j, num in enumerate(row_clean):
            row_clean[j] = int(num)
        data_[i] = row_clean
    cols = ['Status of existing checking account',
            'Duration in month',
            'Credit history',
            'Purpose',
            'Credit amount',
            'Savings account/bonds',
            'Present employment since',
            'Installment rate in percentage of disposable income',
            'Personal status and sex',
            'Other debtors / guarantors',
            'Present residence since',
            'Property',
            'Age in years',
            'Other installment plans',
            'Housing',
            'Number of existing credits at this bank',
            'Job',
            'Number of people being liable to provide maintenance for',
            'Telephone',
            'foreign worker',
            'label']
    df = pd.DataFrame.from_dict(data_,
                                orient='index',
                                # columns=cols
                                )
    df.to_csv(os.path.join(current_file, 'german_credit_numeric.csv'))

if __name__ == '__main__':
    main()
