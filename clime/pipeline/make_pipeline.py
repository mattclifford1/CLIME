'''
put together all aspects of the training/explaination/evaluation pipeline

'score' returned is avg/std over all local explainers from all data points in the dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from dataclasses import dataclass
import logging
import numpy as np
import clime
from clime import data, models, explainer, evaluation, utils


@dataclass
class construct:
    opts: dict

    def run(self, parallel_eval=False):
        '''
        run the whole pipeline
        '''
        train_data, test_data, clf = self.get_data_and_model()
        model_stats = utils.get_model_stats(clf, train_data, test_data)
        score = self.get_evaluation(self.opts, clf, test_data, run_parallel=parallel_eval)
        return {'score': score,
                'model_stats': model_stats,
                'clf': clf,
                'train_data': train_data,
                'test_data': test_data}

    def get_data_and_model(self):
        '''DATA'''
        #  get dataset
        train_data, test_data = self.run_section('dataset',
                                       self.opts,
                                       **self.opts['data params']
                                       # class_samples=self.opts['class samples'],
                                       # percentage=self.opts['percent of data']
                                       )
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
    def get_evaluation(opts, clf, data_dict, run_parallel=False):
        '''
        get an evaluation of a given model and data
        '''
        # object to generate explanations
        expl_gen = explainer_generator(opts)
        # evaluation
        score = construct.run_section('evaluation',
                                  opts,
                                  explainer_generator=expl_gen,
                                  black_box_model=clf,
                                  data=data_dict,
                                  run_parallel=run_parallel)
        return score   # score is either a dict with keys: 'avg', 'std' or a number

    @staticmethod
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


class explainer_generator():
    '''
    object that can be called to generate an explainer from given set of options
    '''
    def __init__(self, opts):
        self.opts = opts

    def __call__(self, clf, data_dict, query_point_ind):
        '''
        get an explainer given a query point, data, and model
        '''
        expl = construct.run_section('explainer',
                                 self.opts,
                                 black_box_model=clf,
                                 query_point=data_dict['X'][query_point_ind, :],
                                 data=data_dict)
        return expl

def _get_explainer_evaluation_wrapper(args):
    # for use with pool.starmap to unpack all the args (but keep default args)
    return get_explainer_evaluation(*args)

def run_pipeline(opts, **kwargs):
    p = construct(opts)
    return p.run(**kwargs)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--single', '-s', action='store_false')
    args=parser.parse_args()
    # logging.basicConfig(level=logging.INFO)
    opts = {
        # 'dataset':             'credit scoring 1',
        'dataset':             'moons',
        # 'dataset':             'Gaussian',
        'data params': {'class samples':  [25, 75], # only for syntheic datasets
                        'percent of data': 0.05,    # for real datasets
                        'moons_noise': 0.2
                        },
        'dataset rebalancing': 'none',
        # 'model':               'SVM',
        # 'model':               'Bayes Optimal',
        'model':               'Random Forest',
        'model balancer':      'none',
        'explainer':           'bLIMEy (cost sensitive sampled)',
        # 'explainer':           'LIME (original)',
        'evaluation':          'fidelity (local)',
    }

    p = construct(opts)
    result = p.run(parallel_eval=args.single)
    print(result['score'])
