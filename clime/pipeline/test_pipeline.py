# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import multiprocessing
from clime import pipeline, utils


def run_pipeline(opts):
    p = pipeline.construct(opts)
    p.run()

def test_all_pipeline_configs(n_cpus=int(multiprocessing.cpu_count())):
    # change n_cpus to 1 if a memory/cpu intensive model training method is used

    all_opts = {
        'class samples': [[2, 3]],   # keep low to reduce comp time
    }
    # add all available methods for each part of the pipeline
    for name in pipeline.AVAILABLE_MODULE_NAMES:
        all_opts[name] = list(pipeline.AVAILABLE_MODULES[name].keys())
    # get all variations/permuations of the pipeline options
    opts_permutations = utils.get_all_dict_permutations(all_opts)
    # now test all variations of methods
    with multiprocessing.Pool(processes=n_cpus) as pool:
            pool.map(run_pipeline, opts_permutations)

if __name__ == '__main__':
    test_all_pipeline_configs()
