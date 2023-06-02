'''
class for standardising data based on a training set (uses sklearn)
normalising transform based on train set can be used on other data e.g. test set
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk> <matt.clifford@bristol.ac.uk>
from sklearn import preprocessing

class normaliser:
    def __init__(self, train_data):
        self.scaler = preprocessing.StandardScaler().fit(train_data['X'])

    def __call__(self, data):
        '''expect data as a dict with 'X', 'y' keys'''
        data['X'] = self.scaler.transform(data['X'])
        return data
