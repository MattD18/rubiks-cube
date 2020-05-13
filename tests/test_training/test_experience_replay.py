import pytest

import numpy as np
import random
import tensorflow as tf

from rubiks_cube.agent import small_cnn
from rubiks_cube.agent import replay_buffer
from rubiks_cube.training.experience_replay import play_episode
from rubiks_cube.environment.cube import Cube


def test_experience_replay_play_episode():
    np.random.seed(5)
    random.seed(5)
    tf.random.set_seed(5)
    model = small_cnn.CNN()
    rb = replay_buffer.ReplayBuffer()
    cube = play_episode(model, rb, training=False)
    assert (cube.state == np.array([[[ 0,  1,  6],
                                    [ 3,  4, 19],
                                    [20, 23, 18]],

                                    [[9, 10, 15],
                                    [12, 13, 14],
                                    [17, 16, 11]],

                                    [[24, 21, 26],
                                    [25, 22,  5],
                                    [ 8,  7,  2]]])).all()