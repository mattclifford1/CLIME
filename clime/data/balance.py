'''
Generate toy data
Data can be balanced or unbalanced
'''
# author: Jonny Erskine
# email: jonathan.erskine@bristol.ac.uk

# author2: Matt Clifford
# email2: matt.clifford@bristol.ac.uk
import os
import sklearn.datasets
import sklearn.model_selection
import sklearn.utils
import random
import numpy as np
import clime
import clime.utils


def get_proportions_and_sample_num(class_samples):
    ''' give class sample num eg. [25, 75] and will return proportions and total
    number of a balanced dataset needed
    '''
    max_class = max(class_samples)
    n_samples = max_class*len(class_samples)   # samples for balanced dataset
    class_proportions = np.array(class_samples)/max_class # normalise
    return n_samples, list(class_proportions)

def unbalance_undersample(data, class_proportions=None, verbose=False):
    '''
    Transfrom balanced dataset into unbalanced dataset
    Classes are unbalanced via undersampling (random sampling without replacement)
        - data: dictionary with keys 'X', 'y' (must be balanced? - would need to implement an assertion)

        - class_proportions: list of values indicating percentage of each class to keep.
                             if values are greater than 1, majority class won't be reduced but the rest will
                             if values are less than 1, all class will be reduced

                             Default: Full representation [1, 1,...] i.e. [100% of class 1, 100% of class 2,...]


    returns:
        - data: dictionary with keys 'X', 'y'
    '''

    clime.utils.out('\n rebalancing classes... \n',verbose)
    # If class proportions left blank, 100% of each class included
    if class_proportions == None:
        class_proportions=[1.0]*len(np.unique(data['y']))
        clime.utils.out("unbalance warning: No class proportions provided.",verbose)

    # make sure class_proportions not higher than 1:
    if max(class_proportions) > 1:
        class_proportions = list(np.array(class_proportions) / max(class_proportions))

    labels = np.unique(data['y'][:])   # List of unique class labels
    unbalanced_i = []                  # List for appending sampling indices

     # For each class:
    #   Return index of every class instance
    #   Count class size
    #   shuffle data and take n samples where n = class proporition * class size

    for  l in range(0,len(labels)):
        label = labels[l]
        proportion = class_proportions[l]  # Moving beyond non-binary really should make this a dictionary

        label_i = [i for i, x in enumerate(data['y']) if x== label]
        class_size = len(label_i)
        unbalanced_class_size = int(class_size*proportion)

        random.seed(int(clime.RANDOM_SEED+label))
        unbalanced_i = [int(i) for i in np.append(unbalanced_i,random.sample(label_i,unbalanced_class_size))]

        clime.utils.out('-'*50,verbose)
        clime.utils.out('Class '+ str(label) + ' | Balanced = ' + str(class_size) + ' , Unbalanced = ' + str(unbalanced_class_size),verbose)

    random.seed(clime.RANDOM_SEED-1)
    random.shuffle(unbalanced_i)
    # now create new data sizes
    instances = data['X'].shape[0]
    for key in data.keys():
        # apply to all numpy arrays that are data rows
        if type(data[key]) == np.ndarray and data[key].shape[0] == instances:
            data[key] = data[key][unbalanced_i]
    return data



def balance_oversample(data, verbose=False):
    '''
    given a dataset, make the classes balanced
    balancing is done via oversampling the minority class
        - data: dictionary with keys 'X', 'y'

    returns:
        - data: dictionary with keys 'X', 'y'
    '''
    labels, counts = np.unique(data['y'][:], return_counts=True)
    # check that classes aren't already balanced
    if len(np.unique(counts)) == 1:
        return data

    balanced_i = []    # List for appending sampling indices
    # create dict for counting class frequencies
    class_freq = {}
    for y in data['y']:
        if y in class_freq:
            class_freq[y]+=1
        else:
            class_freq[y] = 1

    max_freq = 0
    for key, value in class_freq.items():
        if value > max_freq:
            max_freq = value
        clime.utils.out('Class '+f"{int(key)} | {value}",verbose)

     # For each class:
    #   Return index of every class instance
    #   Count class size and determine majority class
    #   for all other classes oversample from observations [NOT DISTRIBUTION]
    #       - shuffle data and take n samples where n = class proporition * class size

    for  l in range(0,len(labels)):
        label = labels[l]
        label_i = [i for i, x in enumerate(data['y']) if x== label]
        class_size = len(label_i)
        if class_size < max_freq:
            random.seed(int(clime.RANDOM_SEED+label))
            balanced_i = [int(i) for i in np.append(balanced_i,random.choices(label_i,k=(max_freq-class_size)))]
            # random.choices => random sampling with replacement
            balanced_i = np.append(label_i,balanced_i)
        else:
            balanced_i = np.append(balanced_i,label_i)
        clime.utils.out('-'*50,verbose)
        clime.utils.out('Class '+ str(label) + ' | Unbalanced = ' + str(class_size) + ' , Balanced = ' + str(max_freq),verbose)
    random.seed(clime.RANDOM_SEED-1)
    random.shuffle(balanced_i)
    data['X'] = data['X'][balanced_i]
    data['y'] = data['y'][balanced_i]
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    train_data, test_data = get_moons()
    unbalanced_train_data = unbalance(train_data,[1,0.5])
    balanced_train_data = balance(unbalanced_train_data,verbose=True)
    plt.subplot(3,1,1)
    plt.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(3,1,2)
    plt.scatter(unbalanced_train_data['X'][:, 0], unbalanced_train_data['X'][:, 1], c=unbalanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(3,1,3)
    plt.scatter(balanced_train_data['X'][:, 0], balanced_train_data['X'][:, 1], c=balanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.show()
