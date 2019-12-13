import os
import datetime
import pickle
import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf

from lib.cube import Cube
from lib.solver import CubeSolver
from lib.models import CNN
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


config['model_params'] = {'embed_dim':100,
                          'num_filters':50,
                          'num_conv_layers':3,
                          'kernel_size':2,
                          'regularization_constant':.05,
                          'num_dense_layers':3,
                          'dense_activation':'elu',
                          'conv_activation':'elu'}

config['training_params'] = {'exploration_rate_func':linear_decay_constant,
                             'num_shuffles':7,
                             'num_episodes':50000,
                             'max_time_steps':10,
                             'epsilon':.1,
                             'decay_constant':5000,
                             'end_state_reward':1,
                             'replay_buffer_capacity':512,
                             'learning_rate':0.00001,
                             'clipnorm':0,
                             'batch_size':128,
                             'discount_factor':.9,
                             'validation_count':400,
                             'val_step':500,
                             'train_log_name':session_name,
                             'logging':True,
                             'checkpointing':True,
                             'stop_on_solve':True}

if __name__ == "__main__":

    solver = CubeSolver()
    solver.model = CNN(**config['model_params'])
    solver.load_model_weights('../models/base_model_v2_20191208165135/weights')
    solver.train(**config['training_params'])
    solver.save_model_weights(weights_dir_name=session_name)
    with open(config_save_path ,'wb') as f:
        pickle.dump(config, f)
