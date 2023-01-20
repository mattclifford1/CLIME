# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from clime import models

def test_model_attribues():
    for model in models.AVAILABLE_MODELS.keys():
        assert issubclass(models.AVAILABLE_MODELS[model], models.base_model)
