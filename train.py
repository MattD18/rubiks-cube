
import os
import argparse
import datetime

import tensorflow as tf

from rubiks_cube.training.experience_replay import train_via_experience_replay
from rubiks_cube.training.utils import parse_config_file, get_optimizer, get_model
from rubiks_cube.training.exploration_rate import ExplorationRateSchedule
from rubiks_cube.agent.small_cnn import CNN
from rubiks_cube.environment.cube import Cube

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help='path to config yaml file')

if __name__ == "__main__":

    #Parse command line arguments
    args = parser.parse_args()
    config_path = args.config_path
    config = parse_config_file(config_path)
    config_name = os.path.split(config_path)[-1].replace('.yaml','')
    
    #Set up training directories
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir= os.path.join('logs/training/gradient_tape', config_name, current_time)
    model_weights_dir = os.path.join('logs/training/model_weights', config_name, current_time, current_time)

    # load model
    model = get_model(**config['model']['params'])
    # use squared loss
    loss_object = tf.keras.losses.MeanSquaredError()
    # load optimizer defined by config file
    optimizer = get_optimizer(**config['optimizer']['params'])
    # initialize exploration rate schduler
    exploration_rate_scheduler = ExplorationRateSchedule(**config['exploration_rate']['params'])

    try:
        # Train the model
        train_via_experience_replay(model, loss_object, optimizer, exploration_rate_scheduler, 
                                    logging=True, train_log_dir=train_log_dir, **config['training_loop']['params'])
    except KeyboardInterrupt:
        print("Training got interrupted")
        #save weights as a fail safe


        save_state = Cube().state
        save_state_tensor = tf.expand_dims(tf.convert_to_tensor(save_state), 0)
        model.predict(save_state_tensor)
        model.save(model_weights_dir)

    # Save trained model weights
    save_state = Cube().state
    save_state_tensor = tf.expand_dims(tf.convert_to_tensor(save_state), 0)
    model.predict(save_state_tensor)
    model.save(model_weights_dir)
    

    # module MCTS
    #gpu/ container
