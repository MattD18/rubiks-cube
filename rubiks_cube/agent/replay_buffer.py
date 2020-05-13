'''
Module for replay buffer
'''
import random

class ReplayBuffer():
    '''
    Replay buffer to store past experience use to train agent via experience replay

    Attributes:
    --------------
    buffer_size : int
    buffer : list
    '''

    def __init__(self, buffer_size=128):
        self.buffer_size = buffer_size
        self.buffer = [None] * self.buffer_size
        self.counter = 0

    
    def add(self, transition):
        '''
        Adds transition to buffer. 
        If buffer is full, new transition will replace oldest transition
        currently in buffer

        Parameters:
        ------------
        transition : tuple
        '''
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_size:
            self.counter = 0

    def get_minibatch(self, batch_size):
        '''
        Returns a random subset of buffer

        Parameters:
        ------------
        batch_size : int (> 0)

        Returns:
        ---------
        minibatch : list of tuples
        '''
        return random.sample(self.buffer, batch_size)

    def is_full(self):
        '''
        check if replay buffer is full
        '''
        return not (None in self.buffer)