from clime.data import *
import numpy as np

samples = 1000
data = get_gaussian(samples=samples)
# data = get_moons(samples=samples)
balanced_data = unbalance(data, [1, 1])
unbalanced_data = unbalance(data, [1,0.5])

print(np.bincount(data['y']))

print(np.bincount(balanced_data['y']))
print(np.bincount(unbalanced_data['y']))
