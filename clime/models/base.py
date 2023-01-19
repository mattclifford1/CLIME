# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
'''
API blueprint for how models should be defined in CLIME
'''
from abc import ABC, abstractmethod

class base_model(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass
