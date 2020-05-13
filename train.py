
import argparse

from rubiks_cube.training.experience_replay import train_via_experience_replay
from rubiks_cube.agent import small_cnn


def load_model():
    model = small_cnn.CNN()
    return model


if __name__ == "__main__":

    #Parse command line arguments

    # load model
    model = load_model()

    try:
        # Train the model
        train_via_experience_replay(model)
    except KeyboardInterrupt:
        print("Training got interrupted")
