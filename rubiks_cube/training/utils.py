'''
helper functions for training
'''

import yaml
import warnings
import tensorflow as tf

from rubiks_cube.agent.small_cnn import CNN

def parse_config_file(config_file_path):
    '''
    parses a training config yaml file

    Parameters:
    -----------
    config_file_path : str

    Returns:
    ---------
    confg : dict
    '''
    with open(config_file_path,'rb') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    config_dict = convert_parameter(config, 
                                    float_params=config['float_params'], 
                                    int_params=config['int_params'])
    return config_dict


def convert_parameter(config, float_params, int_params):
    '''
    recursively parse yaml file, storing value under a 'params' header
    as params

    Parameters:
    ------------
    config : yaml
    float_params : list
    int_params : list

    '''
    if type(config) != dict:
        #BASECASE 1: Return if at a parameter value
        return config
    else:
        for key in config:
            if key == 'params':
                if config[key]:
                    for param in config[key]:
                        if param in float_params:
                            config[key][param] = float(config[key][param])
                        elif param in int_params:
                            config[key][param] = int(config[key][param])
                else:
                    #BASECASE 2
                    config[key] = {}
            else:
                config[key] = convert_parameter(config[key], float_params, int_params)
        return config

def get_optimizer(name=None, **optimizer_kwargs):
    '''
    return an optimizer for training specified by parameters

    Parameters:
    ----------
    name : str
    optimizer_kwargs : dict
        init args for object of class tf.keras.optimizers.Optimizer

    Returns:
    ---------
    optimizer : tf.keras.optimizers.Optimizer
    '''
    optimizer = None
    if name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
    else:
        warnings.warn("Optimizer specified incorrectly, using default: tf.keras.optimizers.SGD() with default params")
        optimizer = tf.keras.optimizers.SGD()
    return optimizer


def get_model(name=None, **model_kwargs):
    '''
    return a model with specified model parametes

    Parameters:
    -----------
    name : str
    model_kwargs : dict
        init args for object of class tf.keras.Model
    '''
    model = None
    if name == 'small_cnn':
        model = CNN(**model_kwargs)
    else:
        warnings.warn("Optimizer specified incorrectly, using default: rubiks_cube.agent.small_cnn.CNN with default params")
        model = CNN()
    return model