# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import numpy as np
from clime import data

def test_get_data():
    train_data, test_data = clime.data.get_costcla_dataset()
    
