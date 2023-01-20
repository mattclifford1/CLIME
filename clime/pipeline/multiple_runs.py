'''
run the pipeline multiple times to get avg/std of aspects
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import numpy as np
from clime import pipeline, utils


def get_avg(opts, limited_data=False):
    '''
    run the pipeline for all the query points in the dataset
    input:
        - opts: config dict for the pipeline
        - limited_data: whether to only run on a few examples (dev purposes)
    '''
    samples = sum(opts['class samples'])
    # first make into lists for permuation generation
    for key in opts.keys():
        opts[key] = [opts[key]]
    # add all data points
    opts['query point'] = list(range(samples))
    if limited_data == True:
        end = min(len(opts['query point']), 10)
        opts['query point'] = opts['query point'][:end]
    opts_permutations = utils.get_all_dict_permutations(opts)
    scores = []
    for opts in opts_permutations:
        p = pipeline.construct(opts)
        scores.append(p.run())
    scores = np.array(scores)
    return {'avg': np.mean(scores), 'std': np.std(scores)}
