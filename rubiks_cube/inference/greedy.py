'''
module for greedy cube solving
'''
import copy
import tensorflow as tf

from rubiks_cube.environment.cube import Cube

def greedy_solve(model, shuffled_cube, max_time_steps, verbose=False):
    '''
    attempt to solve cube greedily by taking action
    with highest q value in each state

    Parameters:
    -----------
    model : tf.keras.Model
        Q function approximator
    shuffled_cube : rubiks_cube.environment.cube.Cube()
        rubix cube object to be solved
    max_time_steps : int
        maximum number of time steps allowed to solve cube
    verbose : boolean
        whether to print steps taken to solve

    Returns:
    --------
    solved : boolean
    cube : rubiks_cube.environment.cube.Cube()
    solver_steps : list of action function names
    '''
    # initialize solution conditions
    solved_cube = Cube()
    solved = False
    solver_steps = []

    s0 = copy.deepcopy(shuffled_cube.state)
    st = tf.expand_dims(tf.convert_to_tensor(s0), 0) # (1, 3, 3, 3)
    # at each step takes argmax_a Q(a,s)
    for t in range(max_time_steps):
        at_index = tf.math.argmax(model(st), 1).numpy()[0]
        at = shuffled_cube.func_list[at_index]
        solver_steps.append(at.__name__)
        if verbose:
            # print action taken
            print(at)
        st1 = at()
        if shuffled_cube == solved_cube:
            # break on solve
            solved = True
            break
        st = tf.expand_dims(tf.convert_to_tensor(st1), 0)
    return solved, shuffled_cube, solver_steps