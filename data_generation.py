'''
Generate toy data
Data can be balanced or unbalanced
'''
import sklearn.datasets
import sklearn.model_selection
import random
import numpy as np


random_seed = 42

def get_data(random_state=random_seed):
    '''
    make half moon dataset from sklearn
        - class_proportion: balance of the classes (default:0.5 is 50/50 split)
                            classes are unbalanced via undersampling

    returns:
        - train_data: dictionary with keys 'X', 'y'
        - test_data:  dictionary with keys 'X', 'y'
    '''
    X, y = sklearn.datasets.make_moons(noise=0.3, random_state=random_state, n_samples=200)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=random_state)
    train_data = {'X': X_train, 'y':y_train}
    test_data =  {'X': X_test,  'y':y_test}

    # now use the class_proportion param to undersample one of the classes


    return train_data, test_data


def unbalance(data,class_proportions=None, verbose=False):
    '''
    Transfrom balanced dataset into unbalanced dataset
        - data: dictionary with keys 'X', 'y' (must be balanced? - would need to implement an assertion)
    
        - class_proportions: Decimal value indicating proportional representation of each class.  
                             Default: Full representation [1,1,...] i.e. [100% of class 1, 100% of class 2,...]
                             Classes are unbalanced via undersampling (random sampling without replacement)

    returns:
        - data: dictionary with keys 'X', 'y'
    '''

    print('\n rebalancing classes... \n' if verbose else None)
    # If class proportions left blank, 100% of each class included
    if class_proportions == None:           
        class_proportions=[1.0]*len(np.unique(data['y']))
        print("unbalance warning: No class proportions provided. This is a very expensive way to copy and paste your original dataset." if verbose else None)


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

        random.seed(int(random_seed+label))
        unbalanced_i = [int(i) for i in np.append(unbalanced_i,random.sample(label_i,unbalanced_class_size))]

        print('-'*50)
        print('Class ',label,' | Balanced = ',class_size,' , Unbalanced = ',unbalanced_class_size)
        
        


    random.seed(random_seed-1)
    random.shuffle(unbalanced_i)

    unbalanced_data = {'X': data['X'][unbalanced_i],'y': data['y'][unbalanced_i]}

    return unbalanced_data



def balance_data(data):
    '''
    given a dataset, make the classes balanced
    balancing is done via oversmaplign the minority class
        - data: dictionary with keys 'X', 'y'

    returns:
        - data: dictionary with keys 'X', 'y'
    '''
    # make balanced usign oversampling

    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    train_data, test_data = get_data()
    class_proportions = [1,0.5]
    unbalanced_train_data = unbalance(train_data,class_proportions)
    
    unbalanced_test_data = unbalance(test_data)
    plt.subplot(2,1,1)
    plt.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(2,1,2)
    plt.scatter(unbalanced_train_data['X'][:, 0], unbalanced_train_data['X'][:, 1], c=unbalanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.show()
