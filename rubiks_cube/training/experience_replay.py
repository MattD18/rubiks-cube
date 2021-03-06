''' Experience replay algorithm as described in
Algorithm 1 of https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
'''
import os
import copy

import numpy as np
import tensorflow as tf

from rubiks_cube.environment.cube import Cube
from rubiks_cube.agent.replay_buffer import ReplayBuffer
from rubiks_cube.inference.greedy import greedy_solve
from rubiks_cube.inference.mcts import mcts_solve

def train_via_experience_replay(model, loss_object, optimizer, exploration_rate_scheduler,
                                num_episodes=20, 
                                buffer_size=128,
                                val_num_shuffles=3,
                                val_max_time_steps=5,
                                val_solve_method='greedy',
                                val_num_cubes=100,
                                mcts_num_search=10,
                                autodidactic=False,
                                train_log_dir='logs/training/gradient_tape',
                                logging=False,
                                logging_freq=100,
                                **episode_kwargs):
    '''
    Wrapper function to train model

    TO DO: Implement model checkpointing

    Parameters:
    -----------
    model : tf.keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    exploration_rate_scheduler : rubiks_cube.training.exploration_rate.ExplorationRateSchedule
    num_episodes : int
    buffer_size : int
    val_num_shuffles : int
    val_max_time_steps : int
    val_solve_method : str
    val_num_cubes : int
    mcts_num_search : int
    autodidactic : boolean
    train_log_dir : str
    logging : boolean
    logging_freq : int
    **episode_kwargs : keyword arguments to play_episde
    '''
    #set up training performance variables
    if logging:
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        validation_cubes = get_validation_cubes(val_num_shuffles, validation_count=val_num_cubes)

    print("Training model")
    rb = ReplayBuffer(buffer_size=buffer_size)
    for episode in range(num_episodes):
        # get episode specific exploration rate
        episode_exploration_rate = exploration_rate_scheduler.get_rate(episode)
        # play episode
        if bool(autodidactic):
            _, episode_loss = play_autodidactic_episode(model, loss_object, optimizer, rb, exploration_rate=episode_exploration_rate, **episode_kwargs)
        else:
            _, episode_loss = play_episode(model, loss_object, optimizer, rb, exploration_rate=episode_exploration_rate, **episode_kwargs)
        if logging:
            # write training loss
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', episode_loss, step=episode)
            # perform validation assessments and write results
            if (episode % logging_freq) == 0:
                avg_max_q = get_val_avg_max_q(model, validation_cubes)
                val_acc = get_val_acc(model, validation_cubes, 
                                     val_max_time_steps=val_max_time_steps, 
                                     val_solve_method=val_solve_method,
                                     mcts_num_search=mcts_num_search)
                with train_summary_writer.as_default():
                    tf.summary.scalar('avg_max_q', avg_max_q, step=episode)
                    tf.summary.scalar('val_acc', val_acc, step=episode)
    

def play_episode(model, loss_object, optimizer, replay_buffer, 
                    num_shuffles=5, 
                    max_time_steps=10, 
                    exploration_rate=.1, 
                    end_state_reward=1.0, 
                    batch_size=16, 
                    discount_factor=.9,
                    training=True):
    '''
    In a single episode, agent has max_time_steps moves to try to solve a cube randomly shuffled num_shuffle times

    Parameters:
    ------------
    model : tf.keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    replay_buffer : rubiks_cube.agent.replay_buffer.ReplayBuffer
    num_shuffles : int (>= 0)
    max_time_steps : int (>= 1)
    exploration_rate : float [0, 1]
    end_state_reward: float
    batch_size : int (>= 1)
    discount_factor: float
    training : boolean
    '''
    #Set up training episode loss
    episode_loss = tf.keras.metrics.Mean()
    #Initialize cube state
    cube = Cube()
    solved_cube = Cube()
    s0 = cube.shuffle(num_shuffles)
    # convert cube state into tensor to feed into model
    st = tf.expand_dims(tf.convert_to_tensor(s0), 0) # (1, 3, 3, 3)
    # Play episode until solved or max_time_steps is reached
    for t in range(max_time_steps):
        #with some probability select a random action a_t
        if np.random.rand() < exploration_rate:
            at_index = np.random.randint(0, 12) #WARNING: Number of possible otations
        #otherwise select a_t = max_a Q(s_t,a)
        else:
            at_index = tf.math.argmax(model(st), 1).numpy()[0]
        # Execute action a_t and observe state s_t+1 and reward r_t
        at = cube.func_list[at_index]
        st1 = at()
        if cube == solved_cube:
            rt = end_state_reward
        else:
            rt = 0.
        # Store transition in replay buffer, convert state to numpy for convenience
        st_numpy = st.numpy()[0] # (3, 3, 3)
        transition = (st_numpy, at_index, rt, st1) # (np.array, int, float, np.array)
        replay_buffer.add(transition)
        #if training is enabled, update q function
        if training:
            loss = update_q_function(model, loss_object, optimizer, replay_buffer, end_state_reward, batch_size, discount_factor)
        else:
            loss = 0
        episode_loss(loss)
        #if reward state has been reached, stop episode early
        if (rt == end_state_reward):
            break
        # convert next cube state into tensor to feed into model
        st = tf.expand_dims(tf.convert_to_tensor(st1), 0) # (1, 3, 3, 3)
    episode_loss_result = episode_loss.result()
    episode_loss.reset_states()
    return cube, episode_loss_result

def play_autodidactic_episode(model, loss_object, optimizer, replay_buffer, 
                    num_shuffles=5, 
                    max_time_steps=10, 
                    exploration_rate=.1, 
                    end_state_reward=1.0, 
                    batch_size=16, 
                    discount_factor=.9,
                    training=True):
    '''
    In a single episode, cube is shuffled up to num_shuffle times, however agent tries to solve cube at every shuffle
    and has 2 x current number of shuffles + 1 to solve

    Parameters:
    ------------
    model : tf.keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    replay_buffer : rubiks_cube.agent.replay_buffer.ReplayBuffer
    num_shuffles : int (>= 0)
    max_time_steps : int (>= 1)
    exploration_rate : float [0, 1]
    end_state_reward: float
    batch_size : int (>= 1)
    discount_factor: float
    training : boolean
    '''
    #Initialize episode cube
    episode_cube = Cube()
    #Initialize episode loss
    episode_loss = tf.keras.metrics.Mean()
    # Initialize solved cube
    solved_cube = Cube()
    for shuffle_step in range(num_shuffles):
        # Initialze shuffle step cube state
        episode_cube.shuffle(1)
        shuffle_step_cube = Cube()
        shuffle_step_cube.state = copy.deepcopy(episode_cube.state)
        #Set up training shuffle_step loss
        shuffle_step_loss = tf.keras.metrics.Mean()
        # regular training loop
        s0 = shuffle_step_cube.state
        # convert cube state into tensor to feed into model
        st = tf.expand_dims(tf.convert_to_tensor(s0), 0) # (1, 3, 3, 3)
        # Play shuffle_step until solved or shuffle_max_time_steps is reached
        shuffle_max_time_steps = 2*shuffle_step + 1
        for t in range(shuffle_max_time_steps):
            #with some probability select a random action a_t
            if np.random.rand() < exploration_rate:
                at_index = np.random.randint(0, 12) #WARNING: Number of possible otations
            #otherwise select a_t = max_a Q(s_t,a)
            else:
                at_index = tf.math.argmax(model(st), 1).numpy()[0]
            # Execute action a_t and observe state s_t+1 and reward r_t
            at = shuffle_step_cube.func_list[at_index]
            st1 = at()
            if shuffle_step_cube == solved_cube:
                rt = end_state_reward
            else:
                rt = 0.
            # Store transition in replay buffer, convert state to numpy for convenience
            st_numpy = st.numpy()[0] # (3, 3, 3)
            transition = (st_numpy, at_index, rt, st1) # (np.array, int, float, np.array)
            replay_buffer.add(transition)
            #if training is enabled, update q function
            if training:
                loss = update_q_function(model, loss_object, optimizer, replay_buffer, end_state_reward, batch_size, discount_factor)
            else:
                loss = 0
            shuffle_step_loss(loss)
            #if reward state has been reached, stop shuffle_step early
            if (rt == end_state_reward):
                break
            # convert next cube state into tensor to feed into model
            st = tf.expand_dims(tf.convert_to_tensor(st1), 0) # (1, 3, 3, 3)
        shuffle_step_loss_result = shuffle_step_loss.result()
        episode_loss(shuffle_step_loss_result )
        shuffle_step_loss.reset_states()
    episode_loss_result = episode_loss.result()
    episode_loss.reset_states()
    return episode_cube, episode_loss_result

def update_q_function(model, loss_object, optimizer, replay_buffer, end_state_reward, batch_size=16, discount_factor=.9):
    '''
    update weights of model with a minibatch samples from replay_buffer

    model : tf.Keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    replay_buffer : rubiks_cube.agent.replay_buffer.ReplayBuffer
    end_state_reward: float
    batch_size : int (>= 1)
    discount_factor: float
    '''
    loss = 0
    if replay_buffer.is_full():
        #grab batch values from replay buffer
        minibatch = replay_buffer.get_minibatch(batch_size)
        states, actions, rewards, next_states = unpack_minibatch(minibatch)
        #perform update
        with tf.GradientTape() as tape:
            # select max predicted q value out of 12 actions
            rows = model(states, training=True)
            selector = tf.expand_dims(actions, 1)
            idx = tf.stack([tf.reshape(tf.range(rows.shape[0]), (-1, 1)), selector], axis=-1)
            predicted_q_vals = tf.gather_nd(rows, idx) # (bs, 1)
            # get y_vals
            y_vals = get_y_vals(rewards, next_states, model, discount_factor, end_state_reward) #(bs, 1)
            loss = loss_object(y_vals, predicted_q_vals)
        #perform gradient step
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def get_y_vals(rewards, next_states, model, discount_factor, end_state_reward):
        '''
        gets approximate reward values based on past experience
        y_j = r_j + I[st1 is not terminal] * gamma * max_a Q(st1, a)
        as in https://arxiv.org/pdf/1312.5602v1.pdf algorithm 1

        Parameters:
        ------------

        rewards : tf.Tensor (bs, )
        next_states : tf.Tensor (bs, 3, 3, 3)
        model : tf.keras.Model
        discount_factor : float
            discount applied to future values in Q function
        end_state_reward : float
        '''
        # convert tensor to np.array for easier processing
        rewards_numpy = rewards.numpy()
        next_state_q_numpy = model(next_states).numpy()
        # add additional term to non_terminal transitions
        non_terminal_mask = (rewards_numpy != end_state_reward)
        rewards_numpy[non_terminal_mask] += discount_factor * np.max(next_state_q_numpy[non_terminal_mask], 1)
        # return value as tensor
        y_vals = tf.expand_dims(tf.constant(rewards_numpy), 1)
        return y_vals


def unpack_minibatch(minibatch):
    '''
    unpacks minibatch into tensors for gradient update

    Parameters:
    mb : list
        list of transition tuples
    '''
    #gets current states, actions, and rewards from batch as tensors
    states = tf.stack([transition[0] for transition in minibatch], 0) # (bs, 3, 3, 3)
    actions = tf.constant([transition[1] for transition in minibatch]) # (bs, )
    rewards = tf.constant([transition[2] for transition in minibatch]) # (bs, )
    next_states = tf.stack([transition[3] for transition in minibatch], 0) # (bs, 3, 3, 3)
    return states, actions, rewards, next_states


def get_val_acc(model, validation_cubes, val_max_time_steps=5, val_solve_method='greedy', mcts_c=.1, mcts_v=.1, mcts_num_search=10):
    '''
    Assess training progress on ability to solve validation cubes

    Parameters:
    -------------
    model : tf.keras.Model
    validation_cubes : list
        list of rubiks_cube.environment.cube.Cube() objects
    val_max_time_steps : int
    val_solve_method : str
        'greedy' or 'mcts'
    mcts_c : float
    mcts_v : float
    mcts_num_search : int
    Returns:
    ----------
    val_acc : float
    '''
    assert val_solve_method in ['greedy', 'mcts']
    solve_count = 0
    for val_cube in validation_cubes:
        val_cube_trial = Cube()
        val_cube_trial.state = np.copy(val_cube.state)
        if val_solve_method == 'greedy':
            solved, _, _ = greedy_solve(model, val_cube_trial, val_max_time_steps)
            solve_count += solved
        elif val_solve_method == 'mcts':
            solved, _ = mcts_solve(model, val_cube_trial, mcts_c, mcts_v, mcts_num_search)
            solve_count += solved
    return solve_count / len(validation_cubes)
  

def get_val_avg_max_q(model, validation_cubes):
    '''
    One measure to assess training progress in
    Playing Atari with Deep Reinforcement Learning is
    the avg maximum Q over all actions in any given state.

    Paper examines improvement over fixed set of states,
    reflected here by validaiton_cubes

    Parameters:
    ----------
    model : tf.keras.model
    validation_cubes : list
        list of rubiks_cube.environment.cube.Cube() objects

    Returns:
    --------
    avg_max_q : float
    '''
    avg_max_q = np.mean([
        model(
            tf.expand_dims(tf.convert_to_tensor(val_cube.state), 0)
        ).numpy().max() for val_cube in validation_cubes
    ])
    return avg_max_q

def get_validation_cubes(val_num_shuffles=1, validation_count=100):
    '''
    Get set of validation cubes that will remain consistent over training period

    Parameters:
    ------------
    val_num_shuffles : int
        number of times validation cube is shuffled
    validation_count : int
        number of validation cubes

    Returns:
    ---------
    validation_cubes : list
        list of rubiks_cube.environment.cube.Cube() objects
    '''
    validation_cubes = []
    for i in range(validation_count):
        val_cube = Cube()
        val_cube.shuffle(val_num_shuffles)
        validation_cubes.append(val_cube)
    return validation_cubes
    