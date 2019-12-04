'''
Module for Cube object

'''
import random
import copy

import numpy as np


class Cube():
    """
    Class that represents a rubix cube

    Attributes
    ----------
    state : np.ndarray
        represents a 3x3x3 rubix cube as a 3x3x3 numpy array where each piece
        has a unique integer id
    func_list : list
        list of possible moves that can be performed to cube

    """

    def __init__(self):
        self.state = np.arange(0, 27).reshape(3, 3, 3)
        self.func_list = [self.front, self.front_p, self.right, self.right_p,
                          self.up, self.up_p, self.left, self.left_p,
                          self.back, self.back_p, self.down, self.down_p]

    def __eq__(self, other):
        equal = False
        if isinstance(other, self.__class__):
            equal = (self.state == other.state).all()
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def front(self):
        '''Front cube rotation'''
        self.state[0] = np.rot90(self.state[0], k=-1)
        return copy.deepcopy(self.state)

    def front_p(self):
        '''Reverse front cube rotation'''
        self.state[0] = np.rot90(self.state[0], k=1)
        return copy.deepcopy(self.state)

    def right(self):
        '''Right cube rotation'''
        self.state[:, :, 2] = np.rot90(self.state[:, :, 2], k=1)
        return copy.deepcopy(self.state)

    def right_p(self):
        '''Reverse right cube rotation'''
        self.state[:, :, 2] = np.rot90(self.state[:, :, 2], k=-1)
        return copy.deepcopy(self.state)

    def up(self):
        '''Up cube rotation'''
        self.state[:, 0, :] = np.rot90(self.state[:, 0, :], k=1)
        return copy.deepcopy(self.state)

    def up_p(self):
        '''Reverse up cube rotation'''
        self.state[:, 0, :] = np.rot90(self.state[:, 0, :], k=-1)
        return copy.deepcopy(self.state)

    def left(self):
        '''Left cube rotation'''
        self.state[:, :, 0] = np.rot90(self.state[:, :, 0], k=-1)
        return copy.deepcopy(self.state)

    def left_p(self):
        '''Reverse left cube rotation'''
        self.state[:, :, 0] = np.rot90(self.state[:, :, 0], k=1)
        return copy.deepcopy(self.state)

    def back(self):
        '''Back cube rotation'''
        self.state[2] = np.rot90(self.state[2], k=1)
        return copy.deepcopy(self.state)

    def back_p(self):
        '''Reverse back cube rotation'''
        self.state[2] = np.rot90(self.state[2], k=-1)
        return copy.deepcopy(self.state)

    def down(self):
        '''Down cube rotation'''
        self.state[:, 2, :] = np.rot90(self.state[:, 2, :], k=-1)
        return copy.deepcopy(self.state)

    def down_p(self):
        '''Reverse down cube rotation'''
        self.state[:, 2, :] = np.rot90(self.state[:, 2, :], k=1)
        return copy.deepcopy(self.state)

    def shuffle(self, num_shuffles, verbose=False):
        '''
        Shuffles internal state num_shuffles times
        '''
        for i in range(num_shuffles):
            move = random.choice(self.func_list)
            move()
            if verbose:
                print(move)
        return copy.deepcopy(self.state)

    def set_state(self, new_state):
        '''
        sets state of cube as a copy of new_state param

        PreC: new_state is a valid cube state np.array
        '''
        self.state = copy.deepcopy(new_state)
