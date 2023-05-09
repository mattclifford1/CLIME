'''
versions of SHAP and KernalSHAP from https://github.com/slundberg/shap
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import shap

class kernal_SHAP:
    '''
    Kernal SHAP - requires access to the training dataset
    '''
    def __init__(self, 
                 black_box_model,
                 train_data,
                 link="logit",
                 **kwargs):
        self.explainer = shap.KernelExplainer(
            black_box_model.predict_proba, train_data['X'], link=link)
        
    def predict_proba(self, X):
        return self.explainer.model.f(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
