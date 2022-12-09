'''
old version that works with current pipeline calls
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


def unbalance(data,class_proportion=0.5):
    '''
    Transfrom balanced dataset into unbalanced dataset
        - data: dictionary with keys 'X', 'y' (must be balanced? - would need to implement an assertion)

        - class_proportion: balance of the classes (default:0.5 is 50/50 split)
                            classes are unbalanced via undersampling (random sampling without replacement)
                            Assuming binary classification: class_proportion applies to class 1, with class_proportion - 1 applied to class 2
    returns:
        - data: dictionary with keys 'X', 'y'
    '''


    # For each class:
    #   count class size
    #   shuffle data and take n samples where n = class proporition * class size
    labels = np.unique(data['y'][:])
    print(labels)
    unbalanced_i = []
    for label in labels:
        print('N: ',label)
        label_i = [i for i, x in enumerate(data['y']) if x== label]
        class_size = len(label_i)
        unbalanced_class_size = int(class_size*abs(label - class_proportion))
        print('Class ',label,' | Balanced = ',class_size,' , Unbalanced = ',unbalanced_class_size)
        random.seed(random_seed+label)
        unbalanced_i = [int(i) for i in np.append(unbalanced_i,random.sample(label_i,unbalanced_class_size))]


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
    class_proportion = 0.2
    unbalanced_train_data = unbalance(train_data,class_proportion)

    unbalanced_test_data = unbalance(test_data,class_proportion)
    plt.subplot(2,1,1)
    plt.scatter(train_data['X'][:, 0], train_data['X'][:, 1], c=train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.subplot(2,1,2)
    plt.scatter(unbalanced_train_data['X'][:, 0], unbalanced_train_data['X'][:, 1], c=unbalanced_train_data['y'], cmap=cm_bright, edgecolors="k")
    plt.show()
