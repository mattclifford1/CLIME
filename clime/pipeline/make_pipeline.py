'''
put together all aspects of the training/explaination/evaluation pipeline

'score' is avg/std over all local explainers from all data points in the dataset
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from dataclasses import dataclass
import numpy as np
import clime
from clime import data, models, explainer, evaluation, utils


@dataclass
class construct:
    opts: dict

    def run(self):
        train_data, clf = self.get_data_model()
        # expl, score = self.get_explainer_evaluation(train_data, clf)
        score_avg = self.get_avg_evaluation(self.opts, clf, train_data)
        return score_avg

    def get_data_model(self):
        '''
          ___   _ _____ _
         |   \ /_\_   _/_\
         | |) / _ \| |/ _ \
         |___/_/ \_\_/_/ \_\
        '''
        #  get dataset
        train_data = self.run_section('dataset',
                                       self.opts,
                                       class_samples=self.opts['class samples'])
        # option to rebalance the data
        train_data = self.run_section('dataset rebalancing',
                                       self.opts,
                                       data=train_data)
        '''
          __  __  ___  ___  ___ _
         |  \/  |/ _ \|   \| __| |
         | |\/| | (_) | |) | _|| |__
         |_|  |_|\___/|___/|___|____|
        '''
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
        return train_data, clf

    @staticmethod
    def get_avg_evaluation(opts, clf, train_data):
        scores = []
        for i in range(len(train_data['y'])):
            _, score = construct.get_explainer_evaluation(opts, train_data, clf, i)
            scores.append(score)
        scores = np.array(scores)
        return {'avg': np.mean(scores), 'std': np.std(scores)}


    @staticmethod
    def get_explainer_evaluation(opts, train_data, clf, query_point_ind):
        '''
          _____  _____ _      _   ___ _  _ ___ ___
         | __\ \/ / _ \ |    /_\ |_ _| \| | __| _ \
         | _| >  <|  _/ |__ / _ \ | || .` | _||   /
         |___/_/\_\_| |____/_/ \_\___|_|\_|___|_|_\
        '''
        expl = construct.run_section('explainer',
                                 opts,
                                 black_box_model=clf,
                                 query_point=train_data['X'][query_point_ind, :])
        '''
          _____   ___   _   _   _  _ _____ ___ ___  _  _
         | __\ \ / /_\ | | | | | |/_\_   _|_ _/ _ \| \| |
         | _| \ V / _ \| |_| |_| / _ \| |  | | (_) | .` |
         |___| \_/_/ \_\____\___/_/ \_\_| |___\___/|_|\_|
        '''
        score = construct.run_section('evaluation',
                                  opts,
                                  expl=expl,
                                  black_box_model=clf,
                                  data=train_data,
                                  query_point=query_point_ind)

        return expl, score

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
            raise ValueError(utils.input_error_msg(options[section], section))
        else:
            return available_modules[options[section]](**kwargs)


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
    p = construct(opts)
    print(p.run())
