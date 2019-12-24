import os
import datetime
import pickle
import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf

from lib.cube import Cube
from lib.solver import CubeSolver
from lib.models import WideNet
from lib.utils import linear_decay_constant, exponential_decay, constant_rate


### BASELINE ARCHITECTURE ###
## TODO: pass baseline config in as argument
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
session_name = 'base_model_v2_{}'.format(now)


logs_base_dir = "../logs/"
config_save_path = os.path.join(logs_base_dir,
                                'configs',
                                session_name + '.pickle')


config = {}


config['model_params'] = {'regularization_constant':.05}

config['training_params'] = {'exploration_rate_func':linear_decay_constant,
                             'num_shuffles':15,
                             'num_episodes':100000,
                             'max_time_steps':20,
                             'epsilon':.1,
                             'decay_constant':10000,
                             'end_state_reward':1,
                             'replay_buffer_capacity':1024,
                             'learning_rate':0.000001,
                             'clipnorm':0,
                             'batch_size':128,
                             'discount_factor':.9,
                             'validation_count':500,
                             'val_step':500,
                             'vary_shuffle':True,
                             'val_shuffles':7,
                             'train_log_name':session_name,
                             'logging':True,
                             'checkpointing':True,
                             'stop_on_solve':True}


if __name__ == "__main__":

    solver = CubeSolver()
    solver.model = WideNet(**config['model_params'])
    model_path = '../models/base_model_v2_20191214134359/weights'
    solver.load_model_weights(model_path)
    solver.train(**config['training_params'])
    solver.save_model_weights(weights_dir_name=session_name)
    with open(config_save_path ,'wb') as f:
        pickle.dump(config, f)
