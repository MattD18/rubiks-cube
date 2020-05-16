import pytest

import random

import numpy as np

from rubiks_cube.environment import cube


def test_cube_init():
    c = cube.Cube()
    assert (c.state == np.arange(0, 27).reshape(3, 3, 3)).all()


def test_cube_rotation():
    c = cube.Cube()
    for rotation in c.func_list:
        rotation()
    assert (c.state == np.arange(0, 27).reshape(3, 3, 3)).all()

def test_cube_shuffle():
    random.seed(5)
    c = cube.Cube()
    c.shuffle(4)
    shuffled_state = np.array([[
        [ 0,  1,  2],
        [ 3,  4,  5],
        [ 8, 17, 20]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [ 7, 16, 23]],

       [[24, 21, 18],
        [25, 22, 19],
        [ 6, 15, 26]
    ]])
    assert (c.state == shuffled_state).all()

def test_cube_equal():
    c1 = cube.Cube()
    c2 = cube.Cube()
    c2.back()
    c2.back_p()
    assert c1 == c2