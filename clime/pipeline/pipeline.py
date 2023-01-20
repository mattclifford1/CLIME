'''
put together all aspects of the training/explaination/evaluation pipeline
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
from dataclasses import dataclass
from clime import data, models, explainer, evaluation, utils


@dataclass
class pipeline:
    opts: dict

    def run(self):
        '''
          ___   _ _____ _
         |   \ /_\_   _/_\
         | |) / _ \| |/ _ \
         |___/_/ \_\_/_/ \_\
        '''
        # work out how much data to sample
        n_samples, class_proportions = data.get_proportions_and_sample_num(self.opts['class samples'])
        #  get dataset
        train_data = self._run_section('dataset',
                                       data.AVAILABLE_DATASETS,
                                       samples=n_samples)
        # unbalance data
        train_data = self._run_section('dataset unbalancing',
                                       data.AVAILABLE_DATA_UNBALANCING,
                                       data=train_data,
                                       class_proportions=self.opts['class samples'])
        # option to balance the data
        train_data = self._run_section('dataset rebalancing',
                                       data.AVAILABLE_DATA_BALANCING,
                                       data=train_data)

        '''
          __  __  ___  ___  ___ _
         |  \/  |/ _ \|   \| __| |
         | |\/| | (_) | |) | _|| |__
         |_|  |_|\___/|___/|___|____|
        '''
        # what model to use
        clf = self._run_section('model', models.AVAILABLE_MODELS, data=train_data)
        # adjust model post training
        clf = self._run_section('model balancer',
                                models.AVAILABLE_MODEL_BALANCING,
                                model=clf,
                                data=train_data,
                                weight=1)

        '''
          _____  _____ _      _   ___ _  _ ___ ___
         | __\ \/ / _ \ |    /_\ |_ _| \| | __| _ \
         | _| >  <|  _/ |__ / _ \ | || .` | _||   /
         |___/_/\_\_| |____/_/ \_\___|_|\_|___|_|_\
        '''
        expl = self._run_section('explainer',
                                 explainer.AVAILABLE_EXPLAINERS,
                                 black_box_model=clf,
                                 query_point=train_data['X'][self.opts['query point'], :])

        '''
          _____   ___   _   _   _  _ _____ ___ ___  _  _
         | __\ \ / /_\ | | | | | |/_\_   _|_ _/ _ \| \| |
         | _| \ V / _ \| |_| |_| / _ \| |  | | (_) | .` |
         |___| \_/_/ \_\____\___/_/ \_\_| |___\___/|_|\_|
        '''
        score = self._run_section('evaluation',
                                  evaluation.AVAILABLE_FIDELITY_METRICS,
                                  expl=expl,
                                  black_box_model=clf,
                                  data=train_data,
                                  query_point=self.opts['query point'])

        return score

    def _run_section(self, section, available_modules, **kwargs):
        '''
        run a portion of the pipeline
        inputs:
            - section: which part of the pipeline to run eg. 'model'
            - available_modules: dict holding all available methods from that sections of the pipeline
            - kwargs: extra inputs to pass to that section of the pipeline eg. train_data
        raises:
            - ValueError: if the requested option isn't available
        '''
        if self.opts[section] not in available_modules.keys():
            raise ValueError(utils.input_error_msg(self.opts[section], available_modules.keys(), section))
        else:
            return available_modules[self.opts[section]](**kwargs)


if __name__ == '__main__':
    opts = {
        'dataset':             'moons',
        'dataset unbalancing': 'undersampling',
        'class samples':       [25, 75],
        'dataset rebalancing': 'none',
        'model':               'SVM',
        'model balancer':      'none',
        'explainer':           'normal',
        'query point':         15,
        'evaluation':          'normal',
    }
    p = pipeline(opts)
    print(p.run())
