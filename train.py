
import os
import argparse

import tensorflow as tf

from rubiks_cube.training.experience_replay import train_via_experience_replay
from rubiks_cube.training.utils import parse_config_file, get_optimizer, get_model
from rubiks_cube.agent.small_cnn import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help='path to config yaml file')

if __name__ == "__main__":

    #Parse command line arguments
    args = parser.parse_args()
    config_path = args.config_path
    config = parse_config_file(config_path)
    config_name = os.path.split(config_path)[-1].replace('.yaml','')
    

    # load model
    model = get_model(**config['model']['params'])
    # use squared loss
    loss_object = tf.keras.losses.MeanSquaredError()
    # load optimizer per config file
    optimizer = get_optimizer(**config['optimizer']['params'])

    try:
        # Train the model
        train_log_dir= os.path.join('logs/training/gradient_tape', config_name)
        train_via_experience_replay(model, loss_object, optimizer, 
                                    logging=True, train_log_dir=train_log_dir,  **config['training_loop']['params'])
    except KeyboardInterrupt:
        print("Training got interrupted")

    #add model saving
    # model config/ exploration rate class
    #gpu/ container
    #MCTS