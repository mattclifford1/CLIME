# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import clime

def test_model_attribues():
    '''
    test that the model is a sub class of the abstract base_model class
    '''
    train_data, _ = clime.pipeline.construct.run_section('dataset',
                                                         {'dataset': 'Moons'},
                                                         class_samples=[2, 3])
    for model in clime.models.AVAILABLE_MODELS:
        clf = clime.pipeline.construct.run_section('model',
                                                   {'model': model},
                                                   data=train_data)
        assert issubclass(type(clf), clime.models.base_model)


if __name__ == '__main__':
    test_model_attribues()
