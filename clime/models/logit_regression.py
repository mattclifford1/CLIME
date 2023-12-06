'''
regression in the logit space 

https://aws.amazon.com/what-is/logistic-regression/#:~:text=Logistic%20regression%20is%20a%20data,outcomes%2C%20like%20yes%20or%20no.
https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
https://www.w3schools.com/python/python_ml_logistic_regression.asp
'''
import sklearn
import numpy as np
from scipy.special import logit, expit

MIN_P = 0.000000001
MAX_P = 0.99999999

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     soft = np.exp(x) / np.sum(np.exp(x), axis=0)
#     soft = expit(x)
#     # convert back into 0, 1 range after numberical stability transform
#     soft_scaled = soft - MIN_P
#     return soft_scaled / (MAX_P - MIN_P)

def logits(x):
    # first make in the range 0.0000001 and 0.9999999 for numerical stability
    x = x * (MAX_P - MIN_P)
    x += MIN_P
    return np.log(x/(1-x))
    # return 1/(1-x)
    # return logit(x)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# def logits(x):
#     # return 1/(1+np.exp(-x))
    # return 1/(1-x)   # float overload



class logit_ridge(sklearn.linear_model.Ridge):
    '''
    N.B probabilites cannot be 0 or 1 when using logits so we scale them to range [0.000001, 0.9999999]
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        y = y[:, 1]
        # print(logits(y))
        # print(y)
        # print(len(y))
        # print(sum(y))
        super().fit(X, logits(y), **kwargs)

    def predict(self, X):
        preds = softmax(super().predict(X))
        preds = np.stack([preds, 1-preds], axis=1)
        return preds