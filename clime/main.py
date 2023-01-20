'''
main entry point to run the pipeline/experiments
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import clime

if __name__ == '__main__':
    opts = {
        'dataset':             'moons',
        'class samples':       [25, 75],
        'dataset rebalancing': 'none',
        'model':               'SVM',
        'model balancer':      'none',
        'explainer':           'normal',
        'evaluation':          'normal',
    }
    print(clime.pipeline.get_avg(opts))
