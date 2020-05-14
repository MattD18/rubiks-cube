import pytest

import numpy as np
import random
import tensorflow as tf

from rubiks_cube.agent.small_cnn import CNN
from rubiks_cube.environment.cube import Cube
from rubiks_cube.agent import replay_buffer
from rubiks_cube.inference.greedy import greedy_solve


def test_greedy_solve():
    model = CNN()
    c = Cube()
    c.shuffle(5)
    solved, solved_cube, _ = greedy_solve(model, c, 5, verbose=False)
    if solved:
        assert solved_cube == Cube()
    else: 
        assert solved_cube != Cube()