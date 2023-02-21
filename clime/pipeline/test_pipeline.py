'''
test all configurations of the pipeline to make sure all combos will run
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import multiprocessing
import numpy as np
from tqdm import tqdm
from clime import pipeline, utils


def test_all_pipeline_configs():
    all_opts = {
        'class samples': [[1, 1]],   # keep low to reduce comp time
        'percent of data': [0.005]    # super small proportion of real dataset
    }
    # add all available methods for each part of the pipeline
    for name in pipeline.AVAILABLE_MODULE_NAMES:
        all_opts[name] = list(pipeline.AVAILABLE_MODULES[name].keys())
    # get all variations/permuations of the pipeline options
    opts_permutations = utils.get_all_dict_permutations(all_opts)
    # now test all variations of methods
    with multiprocessing.Pool() as pool:
            results = list(pool.imap_unordered(pipeline.run_pipeline, opts_permutations))
    for result in results:
        assert type(result['score']['avg']) == np.float64
        assert type(result['score']['std']) == np.float64

# make one to test running in parrelel p.run()



# for bugging when you want to get nice error output and opts

def run_and_print(opts):
    # use for debugging to get opts used
    print(f'{opts=}')
    return pipeline.run_pipeline

def all_pipeline_configs():
    all_opts = {
        'class samples': [[1, 1]],   # keep low to reduce comp time
        'percent of data': [0.005]    # super small proportion of real dataset
    }
    # add all available methods for each part of the pipeline
    for name in pipeline.AVAILABLE_MODULE_NAMES:
        all_opts[name] = list(pipeline.AVAILABLE_MODULES[name].keys())
    # get all variations/permuations of the pipeline options
    opts_permutations = utils.get_all_dict_permutations(all_opts)
    # now test all variations of methods
    with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap_unordered(pipeline.run_pipeline, opts_permutations), total=len(opts_permutations), leave=False, desc='all opts'))
    for result in results:
        assert type(result['score']['avg']) == np.float64
        assert type(result['score']['std']) == np.float64

if __name__ == '__main__':
    # test_all_pipeline_configs()
    all_pipeline_configs()
