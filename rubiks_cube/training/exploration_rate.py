'''
Modules defining exploration strategy during training
'''

import warnings

class ExplorationRateSchedule():
    '''
    class to adjust episode exploration rate over training

    Attributes:
    -----------
    method : str
    method_kwargs : dict
    '''

    def __init__(self, method='linear', **method_kwargs):
        self.method = method
        self.method_kwargs = method_kwargs

        if self.method == 'linear':
            self.get_rate_func = self.get_rate_linear
        else:
            warnings.warn("Using initial exploration rate")
            self.get_rate_func = self.get_rate_constant


    def get_rate(self, episode=0):
        '''

        Parameters:
        -----------
        episode : int

        Returns:
        ---------
        episode_rate : float
        '''
        episode_rate = self.get_rate_func(episode, **self.method_kwargs)
        return episode_rate

        
    def get_rate_linear(self, episode, cutoff_rate=.2, cutoff_episode=50):
        '''
        linearly decays exploration until given point, then held constant
        as described in section 5 of https://arxiv.org/pdf/1312.5602v1.pdf

        Parameters:
        ------------
        episode : int
        cutoff_rate : float [0,1]
        cutoff_episode : float

        Returns:
        ---------
        episode_rate : float
        '''
        episode_rate = None
        # linear decay
        if episode <= cutoff_episode:
            decay_slope = (cutoff_rate - 1) / cutoff_episode
            decay_intercept = 1
            episode_rate = (decay_slope * episode) + decay_intercept
        # constant rate
        else:
            episode_rate = cutoff_rate
        # check assert rate exists
        assert episode_rate
        return episode_rate

    def get_rate_constant(self, episode, constant_rate=.5):
        '''
        get default rate

        Parameters:
        ------------
        episode : int
        constant_rate : float

        Returns:
        ---------
        episode_rate : float
        '''
        return constant_rate