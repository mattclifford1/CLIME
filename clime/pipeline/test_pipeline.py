# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import itertools
from tqdm import tqdm
from clime import pipeline

def test_all_pipeline_configs():
    all_opts = {
        'class samples': [[10, 15]],   # keep low to reduce comp time
        'query point': [1],
    }
    # add all available methods for each part of the pipeline
    for name in pipeline.AVAILABLE_MODULE_NAMES:
        all_opts[name] = list(pipeline.AVAILABLE_MODULES[name].keys())
    # get all variations/permuations of the pipeline options
    keys, values = zip(*all_opts.items())
    opts_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # now test all variations of methods
    for opts in tqdm(opts_permutations, leave=False):
        p = pipeline.construct(opts)
        p.run()

if __name__ == '__main__':
    test_all_pipeline_configs()
