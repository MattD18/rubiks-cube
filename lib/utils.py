'''
contains util functions for rubix cube training
'''


def exponential_decay(base_epsilon, decay_constant, episode):
    '''
    exponentially decayse exploration rate over training

    Parameters:
    ------------
    base_epsilon : float [0,1]
    decay_constant : float
    episode : int

    Returns:
    -------------
    exploration_rate : float [0,1]
    '''
    return base_epsilon**(episode/decay_constant)

def linear_decay_constant(base_epsilon, decay_constant, episode):
    '''
    linearly decays exploration until given point, then held constant
    as described in section 5 of https://arxiv.org/pdf/1312.5602v1.pdf

    Parameters:
    ------------
    base_epsilon : float [0,1]
    decay_constant : float
    episode : int

    Returns:
    -------------
    exploration_rate : float [0,1]
    '''
    if episode < decay_constant:
        exploration_rate = ((base_epsilon-1) /decay_constant)*episode + 1
    else:
        exploration_rate = base_epsilon

    return exploration_rate

def constant_rate(base_epsilon, decay_constant, episode):
    '''
    exponentially decayse exploration rate over training

    Parameters:
    ------------
    base_epsilon : float [0,1]
    decay_constant : float
    episode : int

    Returns:
    -------------
    exploration_rate : float [0,1]
    '''
    return base_epsilon
