'''
Generate toy data
Data can be balanced or unbalanced
'''
import sklearn.datasets
import sklearn.model_selection
import random
import numpy as np
from utils import out
from datasets import GaussClass


random_seed = 42

def get_data(random_state=random_seed):
    '''
    Make two gaussian dataset 
                  half moon dataset from sklearn (deprecated)

    returns:
        - train_data: dictionary with keys 'X', 'y'
        - test_data:  dictionary with keys 'X', 'y'
    '''
    
    X = np.empty([0,2])
    y = []
    label = 0
    size = 200
    class_means = [[0,0],[1,1]] # X and Y cooridnates of mean 
    for m in class_means:
        gaussclass = GaussClass(m[0],m[1],variance=0.5,covariance=np.array([[2,-1],[-1,2]]))
        gaussclass.gen_data(random_seed+label,size)
        X = np.vstack([X,gaussclass.data])
        y = np.append(y,[label]*size)
        label += 1

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.4, random_state=random_state)
    train_data = {'X': X_train, 'y':y_train}
    test_data =  {'X': X_test,  'y':y_test}

    return train_data, test_data


def unbalance_data(data,class_proportions=None, verbose=False):
    '''
    Transfrom balanced dataset into unbalanced dataset
        - data: dictionary with keys 'X', 'y' (must be balanced? - would need to implement an assertion)
    
        - class_proportions: Decimal value indicating proportional representation of each class.  
                             Default: Full representation [1,1,...] i.e. [100% of class 1, 100% of class 2,...]
                             Classes are unbalanced via undersampling (random sampling without replacement)

    returns:
        - data: dictionary with keys 'X', 'y'
    '''

    out('\n rebalancing classes... \n',verbose)
    # If class proportions left blank, 100% of each class included
    if class_proportions == None:           
        class_proportions=[1.0]*len(np.unique(data['y']))
        out("unbalance warning: No class proportions provided.",verbose)


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

        out('-'*50,verbose)
        out('Class '+ str(label) + ' | Balanced = ' + str(class_size) + ' , Unbalanced = ' + str(unbalanced_class_size),verbose)
        
        


    random.seed(random_seed-1)
    random.shuffle(unbalanced_i)

    unbalanced_data = {'X': data['X'][unbalanced_i],'y': data['y'][unbalanced_i]}

    return unbalanced_data



def balance_data(data, verbose=False):
    '''
    given a dataset, make the classes balanced
    balancing is done via oversmaplign the minority class
        - data: dictionary with keys 'X', 'y'

    returns:
        - data: dictionary with keys 'X', 'y'
    '''
    # make balanced usign oversampling

    labels = np.unique(data['y'][:])   # List of unique class labels
    balanced_i = []                  # List for appending sampling indices

    # create dict for counting cl;ass frequencies
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
        
        out('Class '+f"{int(key)} | {value}",verbose)

     # For each class:
    #   Return index of every class instance
    #   Count class size and determine majority class
    #   for all other classes oversample from obsrevations [NOT DISTRIBUTION]
    #       - shuffle data and take n samples where n = class proporition * class size
    
    for  l in range(0,len(labels)):
        label = labels[l]
        
        label_i = [i for i, x in enumerate(data['y']) if x== label]
        class_size = len(label_i)

        if class_size < max_freq:
            random.seed(int(random_seed+label))
            balanced_i = [int(i) for i in np.append(balanced_i,random.sample(label_i,max_freq-class_size))]
            balanced_i = np.append(label_i,balanced_i)
        else:
            balanced_i = np.append(balanced_i,label_i)

        out('-'*50,verbose)
        out('Class '+ str(label) + ' | Unbalanced = ' + str(class_size) + ' , Balanced = ' + str(max_freq),verbose)

    random.seed(random_seed-1)
    random.shuffle(balanced_i)

    balanced_data = {'X': data['X'][balanced_i],'y': data['y'][balanced_i]}

    return balanced_data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    train_data, test_data = get_data()
    unbalanced_train_data = unbalance_data(train_data,[1,0.5])
    balanced_train_data = balance_data(unbalanced_train_data,verbose=True)
    plt.subplot(3,1,1)
    plt.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(3,1,2)
    plt.scatter(unbalanced_train_data['X'][:, 0], unbalanced_train_data['X'][:, 1], c=unbalanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(3,1,3)
    plt.scatter(balanced_train_data['X'][:, 0], balanced_train_data['X'][:, 1], c=balanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.show()
