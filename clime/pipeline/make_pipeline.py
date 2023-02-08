'''
put together all aspects of the training/explaination/evaluation pipeline

'score' returned is avg/std over all local explainers from all data points in the dataset
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from dataclasses import dataclass
from functools import partial
import multiprocessing
import numpy as np
from tqdm import tqdm
import clime
from clime import data, models, explainer, evaluation, utils


@dataclass
class construct:
    opts: dict

    def run(self, parallel_eval=False):
        train_data, test_data, clf = self.get_data_model()
        score_avg = self.get_avg_evaluation(self.opts, clf, test_data, run_parallel=parallel_eval)
        return score_avg

    def get_data_model(self):
        '''DATA'''
        #  get dataset
        train_data, test_data = self.run_section('dataset',
                                       self.opts,
                                       class_samples=self.opts['class samples'],
                                       percentage=self.opts['percent of data'])
        train_data = data.check_data_dict(train_data)
        test_data = data.check_data_dict(test_data)
        # option to rebalance the data
        train_data = self.run_section('dataset rebalancing',
                                       self.opts,
                                       data=train_data)
        '''MODEL'''
        # what model to use
        clf = self.run_section('model',
                                self.opts,
                                data=train_data)
        # adjust model post training
        clf = self.run_section('model balancer',
                                self.opts,
                                model=clf,
                                data=train_data,
                                weight=1)
        return train_data, test_data, clf

    @staticmethod
    def get_avg_evaluation(opts, clf, data_dict, run_parallel=False):
        _get_explainer_evaluation_wrapper=partial(construct.get_explainer_evaluation,
                                                  opts=opts,
                                                  data_dict=data_dict,
                                                  clf=clf)
        data_list = list(range(len(data_dict['y'])))
        if run_parallel == True:
            n_cpus = int(multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=n_cpus) as pool:
                scores = pool.map(_get_explainer_evaluation_wrapper, data_list)
        else:
            scores = list(map(_get_explainer_evaluation_wrapper, data_list))
        scores = np.array(scores)
        return {'avg': np.mean(scores), 'std': np.std(scores)}


    @staticmethod
    def get_explainer_evaluation(query_point_ind, opts, data_dict, clf, get_expl=False):
        '''EXPLAINER'''
        expl = construct.run_section('explainer',
                                 opts,
                                 black_box_model=clf,
                                 query_point=data_dict['X'][query_point_ind, :])
        '''EVALUATION'''
        score = construct.run_section('evaluation',
                                  opts,
                                  expl=expl,
                                  black_box_model=clf,
                                  data=data_dict,
                                  query_point=query_point_ind)
        if get_expl == True:
            return score, expl
        else:
            return score

    @staticmethod  # make static so this can be called from outside the pipeline
    def run_section(section, options, **kwargs):
        '''
        run a portion of the pipeline
        inputs:
            - section: which part of the pipeline to run eg. 'model'
            - options: config for the pipeline
            - kwargs: extra inputs to pass to that section of the pipeline eg. train_data
        raises:
            - ValueError: if the requested option isn't available
        '''
        available_modules = clime.pipeline.AVAILABLE_MODULES[section]
        if options[section] not in available_modules.keys():
            raise ValueError(utils.input_error_msg(options, section))
        else:
            return available_modules[options[section]](**kwargs)


def _get_explainer_evaluation_wrapper(args):
    # for use with pool.starmap to unpack all the args (but keep defualt args)
    return get_explainer_evaluation(*args)

if __name__ == '__main__':
    opts = {
        'dataset':             'credit scoring 1',
        # 'dataset':             'moons',
        'class samples':       [25, 75],    # only for syntheic datasets
        'percent of data':      0.05,       # for real datasets
        'dataset rebalancing': 'none',
        'model':               'SVM',
        'model balancer':      'none',
        'explainer':           'bLIMEy (cost sensitive training)',
        'evaluation':          'fidelity (class balanced)',
    }
    p = construct(opts)
    print(p.run(parallel_eval=False))
