import pytest

import tensorflow as tf
import numpy as np

from rubiks_cube.agent.small_cnn import CNN
from rubiks_cube.environment.cube import Cube


def test_cnn_call_shape():
    model = CNN()
    x = tf.constant(np.stack([Cube().state, Cube().state]))
    assert model(x).shape == tf.TensorShape([2,12])

