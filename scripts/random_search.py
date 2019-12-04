import os
import datetime
import pickle
import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from lib.cube import Cube
from lib.solver import CubeSolver
from lib.models import CNN
from lib.utils import linear_decay_constant, exponential_decay, constant_rate

RANDOM_SEARCH_DIR='../logs/random_search/'
NUM_SEARCH = 5
VALIDATION_COUNT = 100


### BASELINE ARCHITECTURE ###
## TODO: pass baseline config in as argument

CONFIG_NAME = 'base_model_v2'
CONFIG_SAVE_PATH = os.path.join(RANDOM_SEARCH_DIR,
                                'configs',
                                CONFIG_NAME + '.pickle')
CONFIG = {}

CONFIG['model_params'] = {'embed_dim':100,
                          'num_filters':50,
                          'num_conv_layers':3,
                          'kernel_size':2,
                          'regularization_constant':None,
                          'num_dense_layers':3,
                           'dense_activation':'elu'}

CONFIG['training_params'] = {'exploration_rate_func':linear_decay_constant,
                             'num_shuffles':3,
                             'num_episodes':10000,
                             'max_time_steps':None,
                             'epsilon':None,
                             'decay_constant':None,
                             'end_state_reward':1,
                             'replay_buffer_capacity':None,
                             'learning_rate':None,
                             'clipnorm':None,
                             'batch_size':None,
                             'discount_factor':None,
                             'validation_count':400,
                             'val_step':500,
                             'train_log_name':None,
                             'logging':True,
                             'stop_on_solve':True}

with open(CONFIG_SAVE_PATH ,'wb') as f:
    pickle.dump(CONFIG, f)


### SEARCHABLE HYPER PARAMETERS ###
HP_REG_CONSTANT = hp.HParam('regularization_constant', hp.Discrete([0.01,0.05,0.1,0.5,1.0]))
HP_EXP_CONSTANT = hp.HParam('epsilon', hp.Discrete([0.01,0.05,0.1,.2,.3,.5]))
HP_BUFFER = hp.HParam('replay_buffer_capacity', hp.Discrete([64,128,256,512]))
HP_DISCOUNT = hp.HParam('discount_factor', hp.Discrete([.1,.5,.75,.9,.95,.99]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1,4,16,32,64]))
HP_DECAY_CONSTANT = hp.HParam('decay_constant', hp.Discrete([100,200,300,400,500,1000]))
HP_MAX_TIME_STEPS = hp.HParam('max_time_steps', hp.Discrete([3,5,8,10]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([.001,.0001,.00005,.00001,.000005]))
HP_CLIPNORM = hp.HParam('clipnorm', hp.Discrete([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,.01,.1]))
hparams = [HP_REG_CONSTANT, HP_EXP_CONSTANT, HP_BUFFER, HP_DISCOUNT,
           HP_BATCH_SIZE, HP_DECAY_CONSTANT, HP_MAX_TIME_STEPS,
           HP_LEARNING_RATE, HP_CLIPNORM]


#PERFORM RANDOM PARAMETER SEARCH
#http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
for i in range(NUM_SEARCH):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    search_params = {hparam.name:np.random.choice(hparam.domain.values) for hparam in hparams}
    for sp in search_params:
        if type(search_params[sp]) == np.int64:
            search_params[sp] = int(search_params[sp])

    for search_parameter in search_params:
        if search_parameter in CONFIG['model_params']:
            CONFIG['model_params'][search_parameter] = search_params[search_parameter]
        elif search_parameter in CONFIG['training_params']:
            CONFIG['training_params'][search_parameter] = search_params[search_parameter]

    CONFIG['training_params']['train_log_name'] = now
    search_dir = os.path.join(RANDOM_SEARCH_DIR,now)
    with tf.summary.create_file_writer(search_dir).as_default():
        hp.hparams(search_params)
        solver = CubeSolver(train_log_dir=RANDOM_SEARCH_DIR)

        solver.model = CNN(**CONFIG['model_params'])
        try:
            solver.train(**CONFIG['training_params'])
        except ValueError:
            print(search_params)

        solve_count = 0
        for val_trial in range(VALIDATION_COUNT):
            val_trial_cube = Cube()
            val_trial_cube.shuffle(CONFIG['training_params']['num_shuffles'])
            solve_count += solver.solve(val_trial_cube, CONFIG['training_params']['max_time_steps'])[0]
            val_acc = float(solve_count) / float(VALIDATION_COUNT)
        tf.summary.scalar('accuracy', val_acc, step=1)
