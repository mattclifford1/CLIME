# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
'''
Balancing models post training
models  --- todo: generic input as superclas?
'''
import numpy as np
from clime.data import costs


class base_balance():
    '''dummy class to not adjust the model but still have as an identity wrapper in pipeline'''
    def __init__(self, model, data, weight):
        self.model = model
        self.weight = weight

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, X, y):
        '''
        need to keep the class have all attr of a sklean model
        '''
        self.model.fit(X, y)

class adjust_boundary(base_balance):
    '''
    balance the model by pushing its
    boundary away from the minority class since we are less certain about
    predictions in that area, this is achieved by changing the data point before
    being input to the model.
    **currently only works with pushing minority class towards the majority class
    input:
        - model: clime.models.base_model
        - data: dictionary with keys 'X', 'y'
        - weight: scale to push the boundary

    returns:
        - model: model trained on the dataset
    '''
    def __init__(self, model, data, weight=1):
        super().__init__(model, data, weight)
        self._get_vector_to_balance(data)

    def _get_vector_to_balance(self, data):
        '''get the mean value of each class'''
        ys = np.array(data['y'])
        classes = list(np.unique(ys))
        self.means = {}
        self.class_counter = {}
        # set up data storers
        for clas in classes:
            self.means[clas] = np.zeros([data['X'].shape[1]])
            self.class_counter[clas] = 0
        # loop through the dataset
        for i in range(data['X'].shape[0]):
            self.means[data['y'][i]] += data['X'][i, :]
            self.class_counter[data['y'][i]] += 1
        # calc mean
        for clas in classes:
            self.means[clas] /= self.class_counter[clas]
        # get vector from the minority class to the majority class
        min = np.argmin(list(self.class_counter.values()))
        max = np.argmax(list(self.class_counter.values()))
        self.bal_vector = self.means[max] - self.means[min]
        # work out the scale of the class imbalance
        self.diff_scale = self.class_counter[max] - self.class_counter[min]
        # if self.diff_scale == 0:
        #     # make no change when classes are balanced
        #     self.bal_vector = np.zeros(self.bal_vector.shape)
        self.diff_scale /= data['X'].shape[0]

    def _balance_input(self, x):
        return x - (self.bal_vector*self.diff_scale*self.weight)

    def predict(self, x):
        bal_x = self._balance_input(x)
        return self.model.predict(bal_x)

    def predict_proba(self, x):
        bal_x = self._balance_input(x)
        return self.model.predict_proba(bal_x)


class adjust_proba(base_balance):
    '''
    balance the model by changing the
    probabilities output from the model by scaling them with respect to the
    abundance of data points in that class
    input:
        - model: clime.models.base_model
        - data: dictionary with keys 'X', 'y'
        - weight: scale probabilty weight adjustment

    returns:
        - model: model trained on the dataset
    '''
    def __init__(self, model, data, weight=1):
        super().__init__(model, data, weight)
        self._class_weightings(data)

    def _class_weightings(self, data):
        '''
        get the weight of each class proportional to the number of instances
        '''
        self.class_weights = costs.weight_based_on_class_imbalance(data)

    def predict(self, x):
        '''
        no change to predictions
        '''
        return self.model.predict(x)

    def predict_proba(self, x):
        '''
        reduce the probability of less representative class
        '''
        pred_proba = self.model.predict_proba(x)
        # adjust probability with the weights
        adjusted = pred_proba * self.class_weights * self.weight
        # normalise to make a probability again -- is this okay to be doing????
        adjust_probs = adjusted / np.sum(adjusted, axis=1)[:, None]
        return adjust_probs
