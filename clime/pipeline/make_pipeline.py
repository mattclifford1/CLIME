'''
put together all aspects of the training/explaination/evaluation pipeline
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from dataclasses import dataclass
import clime
from clime import data, models, explainer, evaluation, utils


@dataclass
class construct:
    opts: dict

    def run(self):
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

        '''
          _____  _____ _      _   ___ _  _ ___ ___
         | __\ \/ / _ \ |    /_\ |_ _| \| | __| _ \
         | _| >  <|  _/ |__ / _ \ | || .` | _||   /
         |___/_/\_\_| |____/_/ \_\___|_|\_|___|_|_\
        '''
        expl = self.run_section('explainer',
                                 self.opts,
                                 black_box_model=clf,
                                 query_point=train_data['X'][self.opts['query point'], :])

        '''
          _____   ___   _   _   _  _ _____ ___ ___  _  _
         | __\ \ / /_\ | | | | | |/_\_   _|_ _/ _ \| \| |
         | _| \ V / _ \| |_| |_| / _ \| |  | | (_) | .` |
         |___| \_/_/ \_\____\___/_/ \_\_| |___\___/|_|\_|
        '''
        score = self.run_section('evaluation',
                                  self.opts,
                                  expl=expl,
                                  black_box_model=clf,
                                  data=train_data,
                                  query_point=self.opts['query point'])

        return score

    @staticmethod     # make static so this can be called from outside the pipeline
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
        'number of samples':   1000,
        'dataset rebalancing': 'none',
        'model':               'SVM',
        'model balancer':      'none',
        'explainer':           'normal',
        'query point':         15,
        'evaluation':          'normal',
    }
    p = contruct(opts)
    print(p.run())
