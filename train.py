
import argparse

import tensorflow as tf

from rubiks_cube.training.experience_replay import train_via_experience_replay
from rubiks_cube.agent.small_cnn import CNN


if __name__ == "__main__":

    #Parse command line arguments

    # load model
    model = CNN()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    try:
        # Train the model
        train_via_experience_replay(model, loss_object, optimizer, num_episodes=50, logging=True)
    except KeyboardInterrupt:
        print("Training got interrupted")

    #add model saving/config system
    # config helpers
    #gpu/ container