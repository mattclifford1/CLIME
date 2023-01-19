'''
put together all aspects of the training/explaination/evaluation pipeline
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk

from clime import data, model, models, explainer, evaluation, utils

def run(opts):
    '''
      ___   _ _____ _
     |   \ /_\_   _/_\
     | |) / _ \| |/ _ \
     |___/_/ \_\_/_/ \_\
    '''
    # work out how much data to sample
    n_samples, class_proportions = data.get_proportions_and_sample_num(opts['class samples'])
    #  get dataset
    if opts['dataset'] == 'moons':
        train_data = data.get_moons(samples=n_samples)
    elif opts['dataset'] == 'guassian':
        train_data = data.get_gaussian(samples=n_samples)
    else:
        raise ValueError(f"'dataset' needs to be 'moons' or 'guassian' not {opts['dataset']}")
    # unbalance data
    train_data = data.unbalance(train_data, opts['class samples'])
    # option to balance the data
    if opts['rebalance data'] == True:
        train_data = data.balance(train_data)

    '''
      __  __  ___  ___  ___ _
     |  \/  |/ _ \|   \| __| |
     | |\/| | (_) | |) | _|| |__
     |_|  |_|\___/|___/|___|____|
    '''
    # what model to use
    if opts['model'] not in models.AVAILABLE_MODELS.keys():
        raise ValueError(utils.input_error_msg(opts['model'], models.AVAILABLE_MODELS.keys(), 'model'))
    else:
        clf = models.AVAILABLE_MODELS[opts['model']](train_data)

    # adjust model post training
    if opts['model balancer'] not in models.AVAILABLE_MODEL_BALANCING.keys():
        raise ValueError(utils.input_error_msg(opts['model balancer'], models.AVAILABLE_MODELS.keys(), 'model balancer'))
    else:
        clf = models.AVAILABLE_MODEL_BALANCING[opts['model balancer']](clf, train_data, weight=1)

    '''
      _____  _____ _      _   ___ _  _ ___ ___
     | __\ \/ / _ \ |    /_\ |_ _| \| | __| _ \
     | _| >  <|  _/ |__ / _ \ | || .` | _||   /
     |___/_/\_\_| |____/_/ \_\___|_|\_|___|_|_\
    '''
    if opts['explainer'] == 'normal':
        blimey = explainer.bLIMEy(clf, train_data['X'][opts['query point'], :])
    elif opts['explainer'] == 'cost sensitive training':
        blimey = explainer.bLIMEy(clf, train_data['X'][opts['query point'], :], class_weight=True)
    elif opts['explainer'] == 'training data rebalance':
        blimey = explainer.bLIMEy(clf, train_data['X'][opts['query point'], :], class_weight=True)
    else:
        raise ValueError(f"explainer needs to be 'normal' or 'cost sensitive training' not {opts['explainer']}")

    '''
      _____   ___   _   _   _  _ _____ ___ ___  _  _
     | __\ \ / /_\ | | | | | |/_\_   _|_ _/ _ \| \| |
     | _| \ V / _ \| |_| |_| / _ \| |  | | (_) | .` |
     |___| \_/_/ \_\____\___/_/ \_\_| |___\___/|_|\_|
    '''
    if opts['evaluation'] == 'normal fidelity':
        fid = evaluation.fidelity(blimey, clf, train_data)
    elif opts['evaluation'] == 'local fidelity':
        fid = evaluation.local_fidelity(blimey, clf, train_data, opts['query point'])
    elif opts['evaluation'] == 'class balanced fidelity':
        fid = evaluation.bal_fidelity(blimey, clf, train_data)
    else:
        raise ValueError(f"'evaluation' needs to be 'normal fidelity', 'local fidelity' or 'class balanced fidelity' not: {opts['evaluation']}")

    return fid


if __name__ == '__main__':
    opts = {
        'dataset':        'moons',
        'class samples':  [25, 75],
        'rebalance data':  False,
        'model':          'SVM',
        'model balancer': 'none',
        'explainer':      'normal',
        'query point':    10,
        'evaluation':     'normal fidelity',
    }
    f = run(opts)
    print(f)
