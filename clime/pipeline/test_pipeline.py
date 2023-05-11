'''
test all configurations of the pipeline to make sure all combos will run
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import multiprocessing
import numpy as np
from tqdm import tqdm
from clime import pipeline, utils

opts = {
    'data params': [{'class_samples': [20, 20],  # keep low to reduce comp time
                    'percent_of_data': 0.01},  # super small proportion of real dataset
                    {'class_samples': [1, 1],
                    'percent_of_data': 0.01},
                    {'class_samples': [201, 49],
                    'percent_of_data': 0.01}]
}

def test_all_pipeline_configs(dev=False):
    # add all available methods for each part of the pipeline
    for name in pipeline.AVAILABLE_MODULE_NAMES:
        opts[name] = list(pipeline.AVAILABLE_MODULES[name].keys())
    # get all variations/permuations of the pipeline options
    opts_permutations = utils.get_one_dict_permutation(opts)
    # now test all variations of methods
    if dev == False:
        with multiprocessing.Pool() as pool:
                results = list(pool.imap_unordered(run_and_test, opts_permutations))
    else:
        with multiprocessing.Pool(1) as pool:
            results = list(tqdm(pool.imap_unordered(
                run_and_test, opts_permutations), total=len(opts_permutations), desc='all opts'))

# make one to test running in parrelel p.run()

# for bugging when you want to get nice error output and opts
def run_and_test(opts):
    # print for debugging to get opts used
    # print(f'{opts=}')
    result = pipeline.run_pipeline(opts)
    if isinstance(result, dict):
        assert isinstance(result['score']['avg'], np.float64)
        assert isinstance(result['score']['std'], np.float64)
    else:
        assert isinstance(result, np.float64)
    # print(result)
    return result


if __name__ == '__main__':
    test_all_pipeline_configs(dev=True)
