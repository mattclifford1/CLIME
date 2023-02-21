'''
run the pipeline multiple times to get avg/std of aspects
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import multiprocessing
import numpy as np
from clime import pipeline, utils


def _run_pipeline(opts):
    p = pipeline.construct(opts)
    return p.run()

def get_avg(opts, limited_data=False, n_cpus=int(multiprocessing.cpu_count()/2)):
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
    # get all pipeline configs with each query point in the dataset
    opts_permutations = utils.get_all_dict_permutations(opts)
    with multiprocessing.Pool(processes=n_cpus) as pool:
            scores = pool.map(_run_pipeline, opts_permutations)
    scores = np.array(scores)
    return {'avg': np.mean(scores), 'std': np.std(scores)}
