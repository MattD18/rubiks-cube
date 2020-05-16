import pytest

import rubiks_cube
from rubiks_cube.training.utils import parse_config_file, get_optimizer, get_model


def test_parse_config_file():
    test_config_file = 'tests/fixtures/test_config.yaml'
    config = parse_config_file(test_config_file)
    assert type(config['optimizer']['params']['learning_rate']) == float

def test_get_optimizer_default():
    opt = get_optimizer(name='Default', learning_rate=.00001)
    opt_config = opt.get_config()
    assert (opt_config['name'] == 'SGD') and (opt_config['learning_rate'] == 0.01)

def test_get_optimizer_adam():
    opt = get_optimizer(name='Adam', learning_rate=.00001)
    opt_config = opt.get_config()
    assert (opt_config['name'] == 'Adam') and (opt_config['learning_rate'] == .00001)


def test_get_model_default():
    model = get_model(name='Default', fc_dim=10)
    fc_dim = model.fc_layers.get_config()['layers'][0]['config']['units']
    assert (type(model) == rubiks_cube.agent.small_cnn.CNN) \
        & (fc_dim == 50)

def test_get_model_cnn():
    model = get_model(name='small_cnn', fc_dim=10)
    fc_dim = model.fc_layers.get_config()['layers'][0]['config']['units']
    assert (type(model) == rubiks_cube.agent.small_cnn.CNN) \
        & (fc_dim == 10)